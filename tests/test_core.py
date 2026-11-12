import unittest
from types import SimpleNamespace
import torch

from supermimic.algo.supermimic.config import TrainConfig
from supermimic.algo.supermimic.normalization import VecNorm, build_vecnorms
from supermimic.algo.supermimic.replay import ReplayBuffer
from supermimic.env.rollout.interface import RolloutEnv, TrackingState, Transition
from supermimic.env.rollout.policy import rollout
from supermimic.algo.supermimic.wm.env import WorldModelEnv
from supermimic.env.mdp.tracking import TrackingTask
from supermimic.env.mdp.state import PhysicalState
from supermimic.env.mdp.observation import goal
from supermimic.algo.supermimic.wm import (
    MaxCoordRepresentation,
    build_world_model,
    WMBatch,
)
from supermimic.env.mdp.geometry.state import build_maxcoord_full_state, decompose_maxcoord_state
from supermimic.utils.math import quat_to_mat


def fixture(batch=3, bodies=14):
    torch.manual_seed(17)
    p = torch.randn(batch, bodies, 3)
    q = torch.nn.functional.normalize(torch.randn(batch, bodies, 4), dim=-1)
    return build_maxcoord_full_state(p, q, torch.randn_like(p), torch.randn_like(p))


def motion_fixture(frames=100, bodies=14):
    q = torch.zeros(frames, bodies, 4)
    q[..., 0] = 1
    return SimpleNamespace(
        clip_id=torch.zeros(frames, dtype=torch.long),
        clip_start=torch.tensor([0]),
        clip_end=torch.tensor([frames]),
        root_p=torch.zeros(frames, 3),
        root_quat=q[:, 0],
        q=torch.zeros(frames, 29),
        body_p=torch.zeros(frames, bodies, 3),
        body_quat=q,
        body_v=torch.zeros(frames, bodies, 3),
        body_w=torch.zeros(frames, bodies, 3),
        body_names=tuple(["pelvis"] + [f"body_{i}" for i in range(1, bodies)]),
        joint_names=tuple(f"joint_{i}" for i in range(29)),
        last_frame=lambda f: torch.full_like(f, frames - 1),
        future_frames=lambda f, h: torch.minimum(
            f[..., None] + torch.arange(1, h + 1), torch.full_like(f[..., None], frames - 1)
        ),
    )


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_representation_and_both_wms(self):
        full = fixture()
        rep = MaxCoordRepresentation(14)
        rebuilt = rep.reconstruct_full_state(rep.extract_aux_state(full), rep.represent(full))
        for i, (a, b) in enumerate(
            zip(
                decompose_maxcoord_state(full, 14),
                decompose_maxcoord_state(rebuilt, 14),
                strict=True,
            )
        ):
            if i == 1:
                a, b = quat_to_mat(a), quat_to_mat(b)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
        for kind in ["explicit", "latent"]:
            cfg = TrainConfig(wm_kind=kind, wm_hidden=(32, 32), latent_dim=16)
            wm = build_world_model(cfg, rep, 29)
            norms = build_vecnorms(cfg, rep.rep_state_dim, 29)
            wm.input_transform = lambda key, x: norms[key](x)
            xi = torch.randn(3, 33)
            actions = torch.randn(3, 8, 29)
            for k, v in wm.observation_network_inputs(full).items():
                norms[k].update(v)
            initial = wm.initialize(full, xi)
            for k, v in wm.transition_network_inputs(initial, actions[:, 0]).items():
                norms[k].update(v)
            stats = {k: v.clone() for k, v in norms.state_dict().items()}
            result = wm.compute_loss(WMBatch(full, actions, full[:, None].expand(-1, 8, -1), xi))
            result.loss.backward()
            grads = [p.grad for p in wm.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(float(sum(g.abs().sum() for g in grads)), 0)
            for k, v in norms.state_dict().items():
                torch.testing.assert_close(v, stats[k], rtol=0, atol=0)
            targets = wm.prepare_targets(full[:, None])
            self.assertFalse(targets.f.requires_grad)
            if kind == "latent":
                self.assertFalse(targets.z.requires_grad)
                torch.testing.assert_close(initial.z.reshape(3, -1, 8).sum(-1), torch.ones(3, 2))
            for horizon in [8, 32]:
                state = wm.initialize(full, xi)
                a = torch.zeros(3, horizon, 29, requires_grad=True)
                trajectory = wm.rollout(state, a)
                (
                    trajectory.final.aux.square().mean() + trajectory.final.f.square().mean()
                ).backward()
                self.assertTrue(torch.isfinite(a.grad).all())
                self.assertGreater(float(a.grad.abs().sum()), 0)

    def test_goal_full_orientation(self):
        motion = motion_fixture()
        motion.root_p[1:, 0] = 1
        motion.root_quat[:] = torch.tensor([2**-0.5, 0.0, 2**-0.5, 0.0])
        result = goal(motion, torch.tensor([0]), motion.root_quat[:1])
        torch.testing.assert_close(
            result[0, :3], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6
        )

    def test_same_policy_loop_and_raw_replay(self):
        rep = MaxCoordRepresentation(14)
        task = TrackingTask(motion_fixture(), rep)
        full = fixture(2)
        xi = torch.zeros(2, 33)
        raw_xi = xi.clone()
        raw_xi[:, 0] = 0.75
        physical = PhysicalState(rep.extract_aux_state(full), rep.represent(full), xi)
        initial = TrackingState(
            physical,
            task.command.from_frame(torch.zeros(2, dtype=torch.long)),
            torch.zeros(2, 29),
            raw_xi,
            torch.zeros(2, dtype=torch.long),
            full,
        )

        class RealFixture(RolloutEnv):
            def reset(self):
                return initial

            def step(self, state, action):
                nxt = state._replace(
                    command=task.command.advance(state.command),
                    episode_step=state.episode_step + 1,
                    previous_action=action,
                    full=state.full + 1,
                )
                return Transition(
                    nxt,
                    nxt,
                    torch.ones(2),
                    torch.zeros(2, dtype=torch.bool),
                    torch.zeros(2, dtype=torch.bool),
                    {},
                    {"episode": torch.ones(2, dtype=torch.long), "step": state.episode_step},
                )

        real = RealFixture()
        real.task = task
        calls = []

        def policy(obs):
            calls.append(obs.features)
            return torch.zeros(2, 29)

        records = list(rollout(real, real.reset(), policy, 4))
        replay = ReplayBuffer(8, 2, 182, 29, "cpu")
        for record in records:
            replay.add(record)
        torch.testing.assert_close(replay.s[0], full, rtol=0, atol=0)
        torch.testing.assert_close(replay.s1[0], full + 1, rtol=0, atol=0)
        replay.prepare(3, 4, task.motion)
        self.assertEqual(replay.windows.numel(), 4)
        replay.ep[1, 0] = 8
        replay.prepare(3, 4, task.motion)
        self.assertEqual(replay.windows.numel(), 2)
        cfg = TrainConfig(wm_kind="explicit", wm_hidden=(32, 32))
        wm = build_world_model(cfg, rep, 29)
        imagined = WorldModelEnv(wm, task)
        state = imagined.reset(
            full, initial.command, initial.previous_action, raw_xi, initial.episode_step
        )
        steps = list(rollout(imagined, state, policy, 4))
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(s.transition.state is s.transition.continuation for s in steps))
        self.assertTrue(all(s.transition.state.full is None for s in steps))
        self.assertEqual(steps[-1].transition.state.command.motion_t.tolist(), [4, 4])

    def test_command_replay_restore_and_clip_boundary(self):
        motion = motion_fixture(frames=220)
        motion.clip_start = torch.tensor([0, 100])
        motion.clip_end = torch.tensor([100, 220])
        motion.clip_id[100:] = 1
        task = TrackingTask(motion, MaxCoordRepresentation(14))
        command = task.command.from_frame(torch.tensor([99, 105]))
        self.assertEqual(command.motion_id.tolist(), [0, 1])
        self.assertEqual(command.motion_t.tolist(), [99, 5])
        nxt = task.command.advance(command)
        self.assertEqual(task.command.frame(nxt).tolist(), [99, 106])
        # The second clip uses local time; replay must retain its identity.
        full = fixture(1)
        xi = torch.randn(1, 33)
        previous = torch.randn(1, 29)
        episode_step = torch.tensor([499])
        command = task.command.from_frame(torch.tensor([105]))
        for kind in ("explicit", "latent"):
            wm = build_world_model(
                TrainConfig(wm_kind=kind, wm_hidden=(32, 32), latent_dim=16),
                task.representation,
                29,
            )
            env = WorldModelEnv(wm, task)
            state = env.reset(full, command, previous, xi, episode_step)
            state = state._replace(full=full)
            action = torch.zeros_like(previous)
            transition = env.step(state, action)
            replay = ReplayBuffer(4, 1, 182, 29, "cpu")
            replay.add(
                SimpleNamespace(
                    state=state,
                    action=action,
                    transition=transition._replace(
                        state=transition.state._replace(full=full + 1),
                        info={"episode": torch.tensor([7]), "step": torch.tensor([1000])},
                    ),
                )
            )
            replay.prepare(1, 32, motion)
            batch = replay.sample_starts(1)
            from supermimic.env.mdp.command import CommandState

            restored = env.reset(
                batch["s"],
                CommandState(batch["motion_id"], batch["motion_t"]),
                batch["a_prev"],
                batch["xi"],
                batch["episode_step"],
            )
            torch.testing.assert_close(restored.physical.f, state.physical.f)
            torch.testing.assert_close(restored.previous_action, previous)
            torch.testing.assert_close(restored.xi, xi)
            self.assertEqual(restored.command.motion_id.item(), 1)
            self.assertEqual(restored.command.motion_t.item(), 5)
            result = env.step(restored, action)
            self.assertEqual(result.state.episode_step.item(), 500)
            self.assertFalse(
                result.truncated.item()
            )  # imagination keeps existing no-timeout behavior
            torch.testing.assert_close(result.reward, transition.reward)
            torch.testing.assert_close(result.state.xi, xi)

    def test_vecnorm_frozen_forward(self):
        norm = VecNorm(4)
        x = torch.randn(8, 4)
        norm.update(x)
        before = {k: v.clone() for k, v in norm.state_dict().items()}
        x = x.requires_grad_()
        norm(x).sum().backward()
        for k, v in norm.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()

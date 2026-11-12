import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from supermimic.dataset.motion import MotionDataset
from supermimic.env.mdp.reference import TrackingReference
from supermimic.env.mdp.config import TaskConfig
from supermimic.env.mdp.tracking import TrackingTask
from supermimic.env.mdp.state import PhysicalState
from supermimic.env.mdp.observation import concatenate, observation_width
from supermimic.env.rollout.interface import PolicyObservation
from supermimic.env.mdp.geometry.representation import MaxCoordRepresentation
from test_core import fixture, motion_fixture


class TaskConfigTests(unittest.TestCase):
    def test_dataset_is_not_changed_by_body_selection(self):
        n, bodies = 5, 3
        data = dict(
            joint_names=np.array([f"joint{i}" for i in range(29)]),
            body_names=np.array(["pelvis", "torso", "other"]),
            clip_names=np.array(["clip"]),
            fps=np.array(50),
            starts=np.array([0]),
            ends=np.array([n]),
        )
        for name, shape in {
            "root_p": (n, 3),
            "root_quat": (n, 4),
            "root_v": (n, 3),
            "root_w": (n, 3),
            "q": (n, 29),
            "qd": (n, 29),
            "body_p": (n, bodies, 3),
            "body_quat": (n, bodies, 4),
            "body_v": (n, bodies, 3),
            "body_w": (n, bodies, 3),
        }.items():
            data[name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.npz"
            np.savez(path, **data)
            dataset = MotionDataset(path)
            self.assertEqual(dataset.data.batch_size, torch.Size([n]))
            window = dataset.get_slice(
                torch.tensor([0]), torch.tensor([3]), torch.tensor([0, 1, 2])
            )
            self.assertEqual(window.batch_size, torch.Size([1, 3]))
            torch.testing.assert_close(window.body_p, dataset.body_p[torch.tensor([[3, 4, 4]])])
            torch.testing.assert_close(window.to("cpu").q, window.q)
            first = TrackingReference(dataset, ("pelvis", "torso"))
            second = TrackingReference(dataset, ("pelvis", "other", "torso"))
            self.assertEqual(dataset.body_names, ("pelvis", "torso", "other"))
            torch.testing.assert_close(first.body_p, dataset.body_p[:, [0, 1]])
            torch.testing.assert_close(second.body_p, dataset.body_p[:, [0, 2, 1]])
            self.assertFalse(hasattr(dataset, "bin_start"))
            with self.assertRaises(ValueError):
                TrackingReference(dataset, ("pelvis", "missing"))

    def test_terms_are_selected_in_task_config(self):
        cfg = TaskConfig(
            reward_weights={"root_pos": 2.0},
            reward_sigmas={"root_pos": 0.7},
            actor_observations=("goal",),
            critic_observations=("features", "xi_n"),
        )
        cfg.validate()
        rep = MaxCoordRepresentation(14)
        task = TrackingTask(motion_fixture(), rep, cfg)
        full = fixture(2)
        state = PhysicalState(rep.extract_aux_state(full), rep.represent(full), torch.zeros(2, 33))
        self.assertFalse(hasattr(state, "z"))
        reward, terms = task.reward(
            state, torch.zeros(2, dtype=torch.long), torch.zeros(2, 29), torch.zeros(2, 29)
        )
        self.assertEqual(tuple(terms), ("root_pos",))
        expected = 2 * torch.exp(-state.aux[:, :3].square().sum(-1) / 0.7**2) * 0.02
        torch.testing.assert_close(reward, expected)
        obs = PolicyObservation(state.f, torch.zeros(2, 380), state.xi_n, full)
        for names in (cfg.actor_observations, cfg.critic_observations):
            self.assertEqual(
                concatenate(obs, names).shape[-1], observation_width(names, 214, 10, 29)
            )
        cfg.actor_observations = ("latent",)
        with self.assertRaises(ValueError):
            cfg.validate()

"""Native G1 LAFAN tracking evaluation, using the same real env/policy loop."""

import argparse
import json
import torch
from supermimic.algo.supermimic.checkpoint import load_policy
from supermimic.env.sim.env import SimulationEnv
from supermimic.env.rollout.policy import rollout
from supermimic.dataset.motion import MotionDataset
from supermimic.env.mdp.reference import TrackingReference
from supermimic.env.mdp.config import TaskConfig
from supermimic.env.mdp.tracking import TrackingTask
from supermimic.algo.supermimic.wm import MaxCoordRepresentation


@torch.no_grad()
def evaluate(path, motion_path=None, device="cuda"):
    policy, saved = load_policy(path, device)
    task_cfg = TaskConfig(**saved["config"]["task"])
    motion = TrackingReference(
        MotionDataset(motion_path or task_cfg.motion, device), task_cfg.tracking_bodies
    )
    if motion.body_names != tuple(saved["body_names"]) or motion.joint_names != tuple(
        saved["joint_names"]
    ):
        raise ValueError("Evaluation robot/data contract differs")
    task = TrackingTask(
        motion,
        MaxCoordRepresentation(len(motion.body_names)).to(device),
        task_cfg,
    )
    env = SimulationEnv(task, motion.num_clips, device, eval_mode=True, auto_reset=False)
    state = env.reset(frames=motion.clip_start)
    alive = torch.ones(motion.num_clips, dtype=torch.bool, device=device)
    progress = torch.zeros(motion.num_clips, device=device)
    returns = torch.zeros_like(progress)
    horizon = int((motion.clip_end - motion.clip_start).max())
    for record in rollout(env, state, policy, horizon):
        result = record.transition
        valid = alive & ~result.terminated
        progress += valid.float()
        returns += result.reward * valid
        alive = alive & ~(result.terminated | result.truncated)
        if not bool(alive.any()):
            break
    length = (motion.clip_end - motion.clip_start - 1).float()
    rows = [
        {"motion": name, "progress": float(p), "tracking_return": float(r)}
        for name, p, r in zip(
            motion.clip_names,
            progress / length,
            returns / (length * task.max_step_reward),
            strict=True,
        )
    ]
    print(
        json.dumps(
            {
                "protocol": "native LAFAN hard height/tilt termination; no root-position termination",
                "motions": rows,
            },
            indent=2,
        )
    )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--motion")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(8)
    evaluate(a.checkpoint, a.motion, a.device)


if __name__ == "__main__":
    main()

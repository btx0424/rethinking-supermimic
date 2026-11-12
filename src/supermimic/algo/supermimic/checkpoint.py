"""The G1-only checkpoint contract. Historical formats live on the backup branch."""

from dataclasses import asdict
import torch
import json
from supermimic.env.mdp.observation import observation_width, concatenate
from supermimic.algo.supermimic.policy.actor import Actor
from .normalization import VecNorm

FORMAT = "g1_lafan_v1"
PROTOCOL = "task_terms_command_state_all_collision_friction_robot_root_goal_v2"


def save(algo, iteration):
    data = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "config": asdict(algo.cfg),
        "iteration": iteration,
        "body_names": algo.motion.body_names,
        "joint_names": algo.motion.joint_names,
        "wm": algo.wm.state_dict(),
        "actor": algo.actor.state_dict(),
        "critic": algo.critic.state_dict(),
        "vecnorms": algo.vecnorms.state_dict(),
        "reward_norm": algo.reward_norm.state_dict(),
        "opt_wm": algo.opt_wm.state_dict(),
        "opt_pi": algo.opt_pi.state_dict(),
        "opt_v": algo.opt_v.state_dict(),
    }
    torch.save(data, algo.log_dir / f"ckpt_{iteration:06d}.pt")
    temporary = algo.log_dir / "last.pt.tmp"
    torch.save(data, temporary)
    temporary.replace(algo.log_dir / "last.pt")


def read(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("format") != FORMAT or data.get("protocol") != PROTOCOL:
        raise ValueError(
            "Expected G1-only checkpoint; older formats use branch maxcoord-before-cleanup-20261111"
        )
    return data


def resume(algo, path):
    data = read(path, algo.cfg.device)
    old = data["config"]
    current = asdict(algo.cfg)
    keys = (
        "wm_kind",
        "wm_hidden",
        "latent_dim",
        "simnorm_dim",
        "actor_hidden",
        "critic_heads",
        "vecnorm_decay",
        "gamma",
        "normalize_reward",
        "optim",
    )
    if json.dumps(old["task"], sort_keys=True) != json.dumps(current["task"], sort_keys=True):
        raise ValueError("Resume task configuration differs")
    for key in keys:
        a, b = old[key], current[key]
        if isinstance(a, (list, tuple)):
            a, b = tuple(a), tuple(b)
        if a != b:
            raise ValueError(f"Resume configuration differs: {key}")
    if (
        tuple(data["body_names"]) != algo.motion.body_names
        or tuple(data["joint_names"]) != algo.motion.joint_names
    ):
        raise ValueError("Checkpoint/motion body or joint order differs")
    for key in ("wm", "actor", "critic", "vecnorms", "reward_norm", "opt_wm", "opt_pi", "opt_v"):
        getattr(algo, key).load_state_dict(data[key])
    return data["iteration"]


def load_policy(path, device="cuda"):
    data = read(path, device)
    cfg = data["config"]
    feature_dim = 15 * len(data["body_names"]) + 4
    action_dim = len(data["joint_names"])
    width = observation_width(
        cfg["task"]["actor_observations"], feature_dim, cfg["task"]["goal_frames"], action_dim
    )
    actor = Actor(width, action_dim, cfg["actor_hidden"]).to(device).eval()
    actor.load_state_dict(data["actor"])
    norm = VecNorm(width, cfg["vecnorm_decay"]).to(device)
    norm.load_state_dict(
        {k.removeprefix("actor."): v for k, v in data["vecnorms"].items() if k.startswith("actor.")}
    )

    def policy(observation):
        return actor(norm(concatenate(observation, cfg["task"]["actor_observations"])))

    return policy, data

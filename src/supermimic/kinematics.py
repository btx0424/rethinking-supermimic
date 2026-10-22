"""Differentiable forward kinematics matching MuJoCo's body recursion.

The chain is read once from a compiled G1 model. Imagination never calls MuJoCo.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .math_utils import quat_apply, quat_apply_inverse, quat_mul, quat_normalize, rotvec_to_quat

TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


class TorchKinematics:
    """Batched pose and spatial velocity of the tracked G1 bodies.

    State is the pelvis link pose and twist in the world frame, plus the 29
    hinge coordinates in ``joint_names`` order. A hinge angle is applied
    relative to ``qpos0``, which is how MuJoCo defines the joint transform.
    """

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...]):
        self.joint_names = joint_names
        self.body_names = TRACKED_BODIES
        self.nbody = int(model.nbody)
        name_to_body = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)
        }
        missing = [name for name in TRACKED_BODIES if name not in name_to_body]
        if missing:
            raise KeyError(f"Tracked bodies missing from the G1 model: {missing}")
        self.tracked_ids = [name_to_body[name] for name in TRACKED_BODIES]
        self.anchor_index = TRACKED_BODIES.index("torso_link")
        self.pelvis_index = TRACKED_BODIES.index("pelvis")
        self.anchor_name = "torso"

        qpos_adr = {}
        dof_adr = {}
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            qpos_adr[name] = int(model.jnt_qposadr[jid])
            dof_adr[name] = int(model.jnt_dofadr[jid])
        unknown = [name for name in joint_names if name not in qpos_adr]
        if unknown:
            raise KeyError(f"Joints missing from the G1 model: {unknown}")
        self.qpos_adr = [qpos_adr[name] for name in joint_names]
        self.dof_adr = [dof_adr[name] for name in joint_names]

        bodies = []
        for i in range(1, model.nbody):
            jadr = int(model.body_jntadr[i])
            jnum = int(model.body_jntnum[i])
            joints = []
            for k in range(jnum):
                jid = jadr + k
                jtype = int(model.jnt_type[jid])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    kind = "free"
                    q_index = -1
                elif jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
                    kind = "hinge"
                    q_index = joint_names.index(name)
                else:
                    raise NotImplementedError(f"Joint {name} has unsupported type {jtype}")
                qadr = int(model.jnt_qposadr[jid])
                joints.append(
                    {
                        "kind": kind,
                        "q_index": q_index,
                        "axis": model.jnt_axis[jid].copy(),
                        "anchor": model.jnt_pos[jid].copy(),
                        "qpos0": float(model.qpos0[qadr]) if kind == "hinge" else 0.0,
                    }
                )
            bodies.append(
                {
                    "parent": int(model.body_parentid[i]),
                    "pos": model.body_pos[i].copy(),
                    "quat": model.body_quat[i].copy(),
                    "joints": joints,
                }
            )
        self._bodies = bodies
        self._device_cache: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}

    def set_anchor(self, name: str) -> None:
        """Reward and anchor-orientation body. ``torso`` or ``pelvis``."""
        links = {"torso": "torso_link", "pelvis": "pelvis"}
        if name not in links:
            raise ValueError(f"Anchor must be one of {tuple(links)}, got {name!r}.")
        self.anchor_index = self.body_names.index(links[name])
        self.anchor_name = name

    def _buffers(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        key = (str(device), dtype)
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        spec = []
        for body in self._bodies:
            joints = []
            for joint in body["joints"]:
                joints.append(
                    {
                        "kind": joint["kind"],
                        "q_index": joint["q_index"],
                        "axis": torch.tensor(joint["axis"], device=device, dtype=dtype),
                        "anchor": torch.tensor(joint["anchor"], device=device, dtype=dtype),
                        "qpos0": joint["qpos0"],
                    }
                )
            spec.append(
                {
                    "parent": body["parent"],
                    "pos": torch.tensor(body["pos"], device=device, dtype=dtype),
                    "quat": torch.tensor(body["quat"], device=device, dtype=dtype),
                    "joints": joints,
                }
            )
        tracked = torch.tensor(self.tracked_ids, device=device, dtype=torch.long)
        cached = {"bodies": spec, "tracked": tracked}
        self._device_cache[key] = cached
        return cached

    def forward(
        self,
        p: torch.Tensor,
        quat: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked body position, quaternion, linear and angular velocity.

        All vectors are in the world frame. Shapes are (..., 14, 3) and (..., 14, 4).
        """
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        quat = quat_normalize(quat.reshape(-1, 4))
        v = v.reshape(-1, 3)
        w = w.reshape(-1, 3)
        q = q.reshape(-1, q.shape[-1])
        qd = qd.reshape(-1, qd.shape[-1])
        n = p.shape[0]
        buf = self._buffers(p.device, p.dtype)
        eye = torch.zeros(n, 4, device=p.device, dtype=p.dtype)
        eye[:, 0] = 1.0
        pos = [torch.zeros(n, 3, device=p.device, dtype=p.dtype)]
        rot = [eye]
        lin = [torch.zeros(n, 3, device=p.device, dtype=p.dtype)]
        ang = [torch.zeros(n, 3, device=p.device, dtype=p.dtype)]
        for body in buf["bodies"]:
            parent = body["parent"]
            joints = body["joints"]
            if len(joints) == 1 and joints[0]["kind"] == "free":
                pos.append(p)
                rot.append(quat)
                lin.append(v)
                ang.append(w)
                continue
            local_pos = body["pos"].expand(n, 3)
            local_quat = body["quat"].expand(n, 4)
            xpos = quat_apply(rot[parent], local_pos) + pos[parent]
            xquat = quat_normalize(quat_mul(rot[parent], local_quat))
            w_b = ang[parent]
            v_b = lin[parent] + torch.cross(ang[parent], xpos - pos[parent], dim=-1)
            for joint in joints:
                if joint["kind"] != "hinge":
                    raise NotImplementedError(joint["kind"])
                angle = q[:, joint["q_index"]] - joint["qpos0"]
                rate = qd[:, joint["q_index"]]
                axis = joint["axis"].expand(n, 3)
                anchor = joint["anchor"].expand(n, 3)
                xanchor = quat_apply(xquat, anchor) + xpos
                axis_w = quat_apply(xquat, axis)
                qloc = rotvec_to_quat(axis * angle.unsqueeze(-1))
                xquat_new = quat_normalize(quat_mul(xquat, qloc))
                xpos_new = xanchor - quat_apply(xquat_new, anchor)
                v_anchor = v_b + torch.cross(w_b, xanchor - xpos, dim=-1)
                w_b = w_b + axis_w * rate.unsqueeze(-1)
                v_b = v_anchor + torch.cross(w_b, xpos_new - xanchor, dim=-1)
                xpos, xquat = xpos_new, xquat_new
            pos.append(xpos)
            rot.append(xquat)
            lin.append(v_b)
            ang.append(w_b)
        tracked = buf["tracked"].tolist()
        body_p = torch.stack([pos[i] for i in tracked], dim=-2).reshape(*shape, len(tracked), 3)
        body_q = torch.stack([rot[i] for i in tracked], dim=-2).reshape(*shape, len(tracked), 4)
        body_v = torch.stack([lin[i] for i in tracked], dim=-2).reshape(*shape, len(tracked), 3)
        body_w = torch.stack([ang[i] for i in tracked], dim=-2).reshape(*shape, len(tracked), 3)
        return body_p, body_q, body_v, body_w


def compile_g1() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    """Compile the mjlab G1 and return its model plus hinge-joint names."""
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    model = Entity(get_g1_robot_cfg()).spec.compile()
    names = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return model, tuple(names)


def verify_kinematics(kin: TorchKinematics, model: mujoco.MjModel, samples: int = 8) -> None:
    """Raise if torch FK disagrees with MuJoCo on random configurations."""
    data = mujoco.MjData(model)
    gen = torch.Generator().manual_seed(0)
    q = torch.empty(samples, len(kin.joint_names)).uniform_(-0.4, 0.4, generator=gen)
    qd = torch.empty_like(q).uniform_(-1.0, 1.0, generator=gen)
    p = torch.zeros(samples, 3)
    p[:, 2] = 0.8
    yaw = torch.empty(samples).uniform_(-1.0, 1.0, generator=gen)
    quat = torch.stack((yaw.mul(0.5).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), yaw.mul(0.5).sin()), -1)
    v = torch.empty(samples, 3).uniform_(-0.5, 0.5, generator=gen)
    w = torch.empty(samples, 3).uniform_(-0.5, 0.5, generator=gen)
    pred_p, pred_q, pred_v, pred_w = kin.forward(p, quat, v, w, q, qd)
    pos_err, quat_err, vel_err = 0.0, 0.0, 0.0
    for i in range(samples):
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[0:3] = p[i].numpy()
        data.qpos[3:7] = quat[i].numpy()
        data.qpos[kin.qpos_adr] = q[i].numpy()
        rot = quat[i]
        # Free-joint angular velocity is stored in the body frame.
        data.qvel[0:3] = v[i].numpy()
        data.qvel[3:6] = quat_apply_inverse(rot.unsqueeze(0), w[i].unsqueeze(0)).squeeze(0).numpy()
        data.qvel[kin.dof_adr] = qd[i].numpy()
        mujoco.mj_forward(model, data)
        for b, body_id in enumerate(kin.tracked_ids):
            pos_err = max(pos_err, float(torch.linalg.norm(pred_p[i, b] - torch.tensor(data.xpos[body_id])).item()))
            mq = torch.tensor(data.xquat[body_id], dtype=torch.float32)
            dot = float((pred_q[i, b] * mq).sum().abs())
            quat_err = max(quat_err, 1.0 - dot)
            res = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 0)
            # mj_objectVelocity reports the body COM. Link-origin velocity is what the
            # simulator exposes as body_link_lin_vel_w.
            ang_mj = res[:3]
            offset = data.xipos[body_id] - data.xpos[body_id]
            lin_mj = res[3:] - np.cross(ang_mj, offset)
            vel_err = max(
                vel_err,
                float(torch.linalg.norm(pred_w[i, b] - torch.tensor(ang_mj, dtype=torch.float32)).item()),
                float(torch.linalg.norm(pred_v[i, b] - torch.tensor(lin_mj, dtype=torch.float32)).item()),
            )
    if pos_err > 1e-4 or quat_err > 1e-5 or vel_err > 1e-3:
        raise RuntimeError(
            f"Torch FK disagrees with MuJoCo: pos {pos_err:.3e}, quat {quat_err:.3e}, vel {vel_err:.3e}"
        )

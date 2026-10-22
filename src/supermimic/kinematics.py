"""Differentiable forward kinematics matching MuJoCo's body recursion.

The chain is read once from a compiled G1 model. Imagination never calls MuJoCo.
Bodies are advanced depth-by-depth with tensor gathers so siblings share one
kernel launch instead of a Python loop over every link.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .math_utils import quat_apply, quat_apply_inverse, quat_mul, quat_normalize

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


def _hinge_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Unit-axis hinge: ``axis`` (..., 3), ``angle`` (...)."""
    half = 0.5 * angle
    return torch.cat((half.cos().unsqueeze(-1), axis * half.sin().unsqueeze(-1)), dim=-1)


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

        # Body 0 is the world. Bodies 1..nbody-1 match MuJoCo ids.
        parents = np.zeros(self.nbody, dtype=np.int64)
        body_pos = np.zeros((self.nbody, 3), dtype=np.float64)
        body_quat = np.zeros((self.nbody, 4), dtype=np.float64)
        body_quat[:, 0] = 1.0
        axis = np.zeros((self.nbody, 3), dtype=np.float64)
        anchor = np.zeros((self.nbody, 3), dtype=np.float64)
        qpos0 = np.zeros(self.nbody, dtype=np.float64)
        q_index = np.full(self.nbody, -1, dtype=np.int64)
        kind = np.full(self.nbody, -1, dtype=np.int64)  # -1 world, 0 free, 1 hinge
        depth = np.zeros(self.nbody, dtype=np.int64)

        for i in range(1, model.nbody):
            parents[i] = int(model.body_parentid[i])
            depth[i] = depth[parents[i]] + 1
            body_pos[i] = model.body_pos[i]
            body_quat[i] = model.body_quat[i]
            jadr = int(model.body_jntadr[i])
            jnum = int(model.body_jntnum[i])
            if jnum != 1:
                raise NotImplementedError(
                    f"Body {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)} "
                    f"has {jnum} joints; depth-parallel FK expects one."
                )
            jid = jadr
            jtype = int(model.jnt_type[jid])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                kind[i] = 0
            elif jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
                kind[i] = 1
                q_index[i] = joint_names.index(name)
                axis[i] = model.jnt_axis[jid]
                anchor[i] = model.jnt_pos[jid]
                qpos0[i] = float(model.qpos0[int(model.jnt_qposadr[jid])])
            else:
                raise NotImplementedError(f"Joint {name} has unsupported type {jtype}")

        free_ids = np.nonzero(kind == 0)[0]
        if free_ids.size != 1 or int(free_ids[0]) != 1:
            raise NotImplementedError("Expected a single free joint on body 1 (pelvis).")
        self._free_id = int(free_ids[0])

        hinge_layers: list[np.ndarray] = []
        for d in range(2, int(depth.max()) + 1):
            ids = np.nonzero((depth == d) & (kind == 1))[0]
            if ids.size:
                hinge_layers.append(ids.astype(np.int64))

        # G1 (and most humanoids) place each hinge at the child body origin with
        # qpos0=0. That drops the anchor translation from the hinge step.
        hinge_ids = np.nonzero(kind == 1)[0]
        zero_anchor = bool(
            np.allclose(anchor[hinge_ids], 0.0)
            and np.allclose(qpos0[hinge_ids], 0.0)
        )
        self._cpu = {
            "parents": parents,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "axis": axis,
            "anchor": anchor,
            "qpos0": qpos0,
            "q_index": q_index,
            "hinge_layers": hinge_layers,
            "tracked": np.asarray(self.tracked_ids, dtype=np.int64),
            "zero_anchor": zero_anchor,
        }
        # Kept for callers/tests that inspect the old Python chain.
        self._bodies = [
            {
                "parent": int(parents[i]),
                "pos": body_pos[i].copy(),
                "quat": body_quat[i].copy(),
                "joints": (
                    [{"kind": "free", "q_index": -1, "axis": axis[i].copy(), "anchor": anchor[i].copy(), "qpos0": 0.0}]
                    if kind[i] == 0
                    else [
                        {
                            "kind": "hinge",
                            "q_index": int(q_index[i]),
                            "axis": axis[i].copy(),
                            "anchor": anchor[i].copy(),
                            "qpos0": float(qpos0[i]),
                        }
                    ]
                ),
            }
            for i in range(1, self.nbody)
        ]
        self._device_cache: dict[tuple[str, torch.dtype], dict] = {}

    def set_anchor(self, name: str) -> None:
        """Reward and anchor-orientation body. ``torso`` or ``pelvis``."""
        links = {"torso": "torso_link", "pelvis": "pelvis"}
        if name not in links:
            raise ValueError(f"Anchor must be one of {tuple(links)}, got {name!r}.")
        self.anchor_index = self.body_names.index(links[name])
        self.anchor_name = name

    def _buffers(self, device: torch.device, dtype: torch.dtype) -> dict:
        key = (str(device), dtype)
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        c = self._cpu

        def t(x, dt=dtype):
            return torch.as_tensor(x, device=device, dtype=dt)

        layers = []
        for ids in c["hinge_layers"]:
            layers.append(
                {
                    "body": torch.as_tensor(ids, device=device, dtype=torch.long),
                    "parent": torch.as_tensor(c["parents"][ids], device=device, dtype=torch.long),
                    "local_pos": t(c["body_pos"][ids]),
                    "local_quat": t(c["body_quat"][ids]),
                    "axis": t(c["axis"][ids]),
                    "anchor": t(c["anchor"][ids]),
                    "qpos0": t(c["qpos0"][ids]),
                    "q_index": torch.as_tensor(c["q_index"][ids], device=device, dtype=torch.long),
                }
            )
        cached = {
            "layers": layers,
            "tracked": torch.as_tensor(c["tracked"], device=device, dtype=torch.long),
            "free_id": self._free_id,
            "nbody": self.nbody,
            "zero_anchor": c["zero_anchor"],
        }
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
        nbody = buf["nbody"]
        free_id = buf["free_id"]

        # (batch, body, *) — gather siblings with a single index select per depth.
        pos = p.new_zeros(n, nbody, 3)
        rot = p.new_zeros(n, nbody, 4)
        rot[..., 0] = 1.0
        lin = p.new_zeros(n, nbody, 3)
        ang = p.new_zeros(n, nbody, 3)

        pos[:, free_id] = p
        rot[:, free_id] = quat
        lin[:, free_id] = v
        ang[:, free_id] = w

        zero_anchor = buf["zero_anchor"]
        for layer in buf["layers"]:
            parent = layer["parent"]
            body = layer["body"]
            q_idx = layer["q_index"]
            p_pos = pos[:, parent]
            p_rot = rot[:, parent]
            p_lin = lin[:, parent]
            p_ang = ang[:, parent]
            # Broadcast static body params to (N, L, *). torch.cross needs matching ndim.
            local_pos_b = layer["local_pos"].expand(n, -1, -1)
            local_quat_b = layer["local_quat"].expand(n, -1, -1)
            axis_b = layer["axis"].expand(n, -1, -1)

            xpos = quat_apply(p_rot, local_pos_b) + p_pos
            xquat_pre = quat_normalize(quat_mul(p_rot, local_quat_b))
            v_b = p_lin + torch.cross(p_ang, xpos - p_pos, dim=-1)
            angle = q[:, q_idx] - layer["qpos0"]
            rate = qd[:, q_idx]
            # Joint axis lives in the pre-hinge body frame (MuJoCo).
            axis_w = quat_apply(xquat_pre, axis_b)
            xquat = quat_normalize(quat_mul(xquat_pre, _hinge_quat(axis_b, angle)))
            w_b = p_ang + axis_w * rate.unsqueeze(-1)

            if zero_anchor:
                # Hinge at the body origin: origin and linear velocity unchanged.
                pos[:, body] = xpos
                rot[:, body] = xquat
                lin[:, body] = v_b
                ang[:, body] = w_b
                continue

            anchor_b = layer["anchor"].expand(n, -1, -1)
            xanchor = quat_apply(xquat_pre, anchor_b) + xpos
            xpos_new = xanchor - quat_apply(xquat, anchor_b)
            v_anchor = v_b + torch.cross(p_ang, xanchor - xpos, dim=-1)
            v_b = v_anchor + torch.cross(w_b, xpos_new - xanchor, dim=-1)
            pos[:, body] = xpos_new
            rot[:, body] = xquat
            lin[:, body] = v_b
            ang[:, body] = w_b

        tracked = buf["tracked"]
        n_tracked = tracked.shape[0]
        return (
            pos[:, tracked].reshape(*shape, n_tracked, 3),
            rot[:, tracked].reshape(*shape, n_tracked, 4),
            lin[:, tracked].reshape(*shape, n_tracked, 3),
            ang[:, tracked].reshape(*shape, n_tracked, 3),
        )


def make_kinematics(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    backend: str = "warp",
    device: str | torch.device = "cuda",
):
    """``warp`` (default) or ``torch``. Both expose ``TorchKinematics``' public attributes."""
    kin = TorchKinematics(model, joint_names)
    if backend == "torch":
        return kin
    if backend == "warp":
        from .warp_fk import WarpKinematics

        return WarpKinematics(kin, device)
    raise ValueError(f"Unknown FK backend {backend!r}. Expected 'warp' or 'torch'.")


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

"""Warp forward kinematics. One thread walks the G1 chain; gradients come from Warp's tape.

Quaternions on the torch side stay (w, x, y, z). Warp's ``wp.quat`` is (x, y, z, w);
the kernel converts at the boundary.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .kinematics import TorchKinematics
from .math_utils import quat_normalize

wp.config.quiet = True
wp.init()
# mjlab turns this off globally. This module needs the adjoint for torch grads.
wp.set_module_options({"enable_backward": True})


@wp.func
def _wxyz(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[3], q[0], q[1], q[2])


@wp.func
def _xyzw(q: wp.vec4) -> wp.quat:
    return wp.normalize(wp.quat(q[1], q[2], q[3], q[0]))


@wp.func
def _hinge_quat(axis: wp.vec3, angle: wp.float32) -> wp.quat:
    half = wp.float32(0.5) * angle
    s = wp.sin(half)
    return wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(half))


@wp.kernel
def _fk(
    parent: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3),
    local_quat: wp.array(dtype=wp.vec4),
    axis: wp.array(dtype=wp.vec3),
    anchor: wp.array(dtype=wp.vec3),
    qpos0: wp.array(dtype=wp.float32),
    q_index: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    zero_anchor: wp.int32,
    p_in: wp.array(dtype=wp.vec3),
    quat_in: wp.array(dtype=wp.vec4),
    v_in: wp.array(dtype=wp.vec3),
    w_in: wp.array(dtype=wp.vec3),
    q_in: wp.array2d(dtype=wp.float32),
    qd_in: wp.array2d(dtype=wp.float32),
    tracked: wp.array(dtype=wp.int32),
    pos: wp.array2d(dtype=wp.vec3),
    rot: wp.array2d(dtype=wp.vec4),
    lin: wp.array2d(dtype=wp.vec3),
    ang: wp.array2d(dtype=wp.vec3),
    out_p: wp.array2d(dtype=wp.vec3),
    out_q: wp.array2d(dtype=wp.vec4),
    out_v: wp.array2d(dtype=wp.vec3),
    out_w: wp.array2d(dtype=wp.vec3),
):
    i = wp.tid()
    nbody = parent.shape[0]
    for b in range(nbody):
        k = kind[b]
        if k == 0:
            pos[i, b] = p_in[i]
            rot[i, b] = quat_in[i]
            lin[i, b] = v_in[i]
            ang[i, b] = w_in[i]
        elif k < 0:
            pos[i, b] = wp.vec3(0.0, 0.0, 0.0)
            rot[i, b] = wp.vec4(1.0, 0.0, 0.0, 0.0)
            lin[i, b] = wp.vec3(0.0, 0.0, 0.0)
            ang[i, b] = wp.vec3(0.0, 0.0, 0.0)
        else:
            par = parent[b]
            p_rot = _xyzw(rot[i, par])
            p_pos = pos[i, par]
            p_ang = ang[i, par]
            local_q = _xyzw(local_quat[b])
            axis_b = axis[b]
            xpos = wp.quat_rotate(p_rot, local_pos[b]) + p_pos
            xquat_pre = wp.normalize(wp.mul(p_rot, local_q))
            v_b = lin[i, par] + wp.cross(p_ang, xpos - p_pos)
            j = q_index[b]
            angle = q_in[i, j] - qpos0[b]
            rate = qd_in[i, j]
            axis_w = wp.quat_rotate(xquat_pre, axis_b)
            xquat = wp.normalize(wp.mul(xquat_pre, _hinge_quat(axis_b, angle)))
            w_b = p_ang + axis_w * rate
            if zero_anchor == 1:
                pos[i, b] = xpos
                rot[i, b] = _wxyz(xquat)
                lin[i, b] = v_b
                ang[i, b] = w_b
            else:
                anch = anchor[b]
                xanchor = wp.quat_rotate(xquat_pre, anch) + xpos
                xpos_new = xanchor - wp.quat_rotate(xquat, anch)
                v_anchor = v_b + wp.cross(p_ang, xanchor - xpos)
                pos[i, b] = xpos_new
                rot[i, b] = _wxyz(xquat)
                lin[i, b] = v_anchor + wp.cross(w_b, xpos_new - xanchor)
                ang[i, b] = w_b
    n_tracked = tracked.shape[0]
    for t in range(n_tracked):
        b = tracked[t]
        out_p[i, t] = pos[i, b]
        out_q[i, t] = rot[i, b]
        out_v[i, t] = lin[i, b]
        out_w[i, t] = ang[i, b]


def _as_wp(tensor: torch.Tensor, dtype, requires_grad: bool):
    return wp.from_torch(tensor, dtype=dtype, requires_grad=requires_grad)


class _Scratch:
    """Body-local poses for one in-flight FK. Reused only after its backward."""

    def __init__(self, n: int, nbody: int, device):
        self.n = n
        self.busy = False
        self.pos = wp.zeros((n, nbody), dtype=wp.vec3, device=device, requires_grad=True)
        self.rot = wp.zeros((n, nbody), dtype=wp.vec4, device=device, requires_grad=True)
        self.lin = wp.zeros((n, nbody), dtype=wp.vec3, device=device, requires_grad=True)
        self.ang = wp.zeros((n, nbody), dtype=wp.vec3, device=device, requires_grad=True)

    def zero_grad(self) -> None:
        for arr in (self.pos, self.rot, self.lin, self.ang):
            if arr.grad is not None:
                arr.grad.zero_()


class WarpKinematics:
    """Same outputs as ``TorchKinematics.forward``, evaluated in one Warp kernel."""

    def __init__(self, kin: TorchKinematics, device: str | torch.device = "cuda"):
        self._kin = kin
        c = kin._cpu
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch_device.index is None:
            torch_device = torch.device("cuda", torch.cuda.current_device())
        device = wp.device_from_torch(torch_device)
        kind = np.full(kin.nbody, -1, dtype=np.int32)
        kind[kin._free_id] = 0
        hinge = np.nonzero(c["q_index"] >= 0)[0]
        kind[hinge] = 1
        self.nbody = kin.nbody
        self.n_j = len(kin.joint_names)
        self.tracked_ids = list(kin.tracked_ids)
        self.n_tracked = len(self.tracked_ids)
        self._zero_anchor = 1 if c["zero_anchor"] else 0
        self._parent = wp.array(c["parents"].astype(np.int32), dtype=wp.int32, device=device)
        self._local_pos = wp.array(c["body_pos"].astype(np.float32), dtype=wp.vec3, device=device)
        self._local_quat = wp.array(c["body_quat"].astype(np.float32), dtype=wp.vec4, device=device)
        self._axis = wp.array(c["axis"].astype(np.float32), dtype=wp.vec3, device=device)
        self._anchor = wp.array(c["anchor"].astype(np.float32), dtype=wp.vec3, device=device)
        self._qpos0 = wp.array(c["qpos0"].astype(np.float32), dtype=wp.float32, device=device)
        self._q_index = wp.array(c["q_index"].astype(np.int32), dtype=wp.int32, device=device)
        self._kind = wp.array(kind, dtype=wp.int32, device=device)
        self._tracked = wp.array(np.asarray(self.tracked_ids, dtype=np.int32), dtype=wp.int32, device=device)
        self._device = device
        self._pool: list[_Scratch] = []
        self._nograd: _Scratch | None = None
        self._fn = _apply_factory(self)

    def set_anchor(self, name: str) -> None:
        self._kin.set_anchor(name)

    def __getattr__(self, name: str):
        return getattr(self._kin, name)

    def _checkout(self, n: int, keep: bool) -> _Scratch:
        """``keep`` holds the buffer until backward. Several FK calls can sit in one graph."""
        if not keep:
            scratch = self._nograd
            if scratch is None or scratch.n < n:
                scratch = self._nograd = _Scratch(n, self.nbody, self._device)
            scratch.zero_grad()
            return scratch
        for scratch in self._pool:
            if not scratch.busy and scratch.n >= n:
                scratch.busy = True
                scratch.zero_grad()
                return scratch
        scratch = _Scratch(n, self.nbody, self._device)
        scratch.busy = True
        self._pool.append(scratch)
        return scratch

    def _release(self, scratch: _Scratch) -> None:
        if scratch is not self._nograd:
            scratch.busy = False

    @torch.compiler.disable
    def forward(self, p, quat, v, w, q, qd):
        # Eager only. Dynamo cannot trace Warp's ctypes kernel launch, and this
        # runs inside the compiled imagination step once body observations need FK.
        # Autograd is unchanged: the tape still backprops into the caller.
        # Match TorchKinematics: the free-joint quaternion is normalized in torch
        # so its gradient stays on the autograd graph.
        return self._fn.apply(p, quat_normalize(quat), v, w, q, qd)


def _apply_factory(kin: WarpKinematics):
    class _Apply(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p, quat, v, w, q, qd):
            shape = p.shape[:-1]
            p = p.reshape(-1, 3).contiguous()
            quat = quat.reshape(-1, 4).contiguous()
            v = v.reshape(-1, 3).contiguous()
            w = w.reshape(-1, 3).contiguous()
            q = q.reshape(-1, q.shape[-1]).contiguous()
            qd = qd.reshape(-1, qd.shape[-1]).contiguous()
            n = p.shape[0]
            # Function.forward runs under no_grad, so this is the grad-mode signal.
            keep = any(ctx.needs_input_grad)
            scratch = kin._checkout(n, keep)
            out_p = torch.empty(n, kin.n_tracked, 3, device=p.device, dtype=p.dtype)
            out_q = torch.empty(n, kin.n_tracked, 4, device=p.device, dtype=p.dtype)
            out_v = torch.empty(n, kin.n_tracked, 3, device=p.device, dtype=p.dtype)
            out_w = torch.empty(n, kin.n_tracked, 3, device=p.device, dtype=p.dtype)
            # Detach so Warp's grad buffer is not the leaf's .grad. Views share storage.
            wp_p = _as_wp(p.detach(), wp.vec3, True)
            wp_quat = _as_wp(quat.detach(), wp.vec4, True)
            wp_v = _as_wp(v.detach(), wp.vec3, True)
            wp_w = _as_wp(w.detach(), wp.vec3, True)
            wp_q = _as_wp(q.detach(), wp.float32, True)
            wp_qd = _as_wp(qd.detach(), wp.float32, True)
            wp_op = _as_wp(out_p, wp.vec3, True)
            wp_oq = _as_wp(out_q, wp.vec4, True)
            wp_ov = _as_wp(out_v, wp.vec3, True)
            wp_ow = _as_wp(out_w, wp.vec3, True)
            tape = wp.Tape()
            stream = wp.stream_from_torch(torch.cuda.current_stream(p.device))
            with wp.ScopedStream(stream):
                with tape:
                    wp.launch(
                        _fk,
                        dim=n,
                        inputs=[
                            kin._parent,
                            kin._local_pos,
                            kin._local_quat,
                            kin._axis,
                            kin._anchor,
                            kin._qpos0,
                            kin._q_index,
                            kin._kind,
                            kin._zero_anchor,
                            wp_p,
                            wp_quat,
                            wp_v,
                            wp_w,
                            wp_q,
                            wp_qd,
                            kin._tracked,
                            scratch.pos,
                            scratch.rot,
                            scratch.lin,
                            scratch.ang,
                        ],
                        outputs=[wp_op, wp_oq, wp_ov, wp_ow],
                        device=kin._device,
                    )
            ctx.tape = tape
            ctx.scratch = scratch if keep else None
            ctx.inputs = (wp_p, wp_quat, wp_v, wp_w, wp_q, wp_qd)
            ctx.outputs = (wp_op, wp_oq, wp_ov, wp_ow)
            ctx.shape = shape
            ctx.n_tracked = kin.n_tracked
            return (
                out_p.view(*shape, kin.n_tracked, 3),
                out_q.view(*shape, kin.n_tracked, 4),
                out_v.view(*shape, kin.n_tracked, 3),
                out_w.view(*shape, kin.n_tracked, 3),
            )

        @staticmethod
        def backward(ctx, gp, gq, gv, gw):
            outs = ctx.outputs
            grads = (
                gp.reshape(-1, ctx.n_tracked, 3).contiguous(),
                gq.reshape(-1, ctx.n_tracked, 4).contiguous(),
                gv.reshape(-1, ctx.n_tracked, 3).contiguous(),
                gw.reshape(-1, ctx.n_tracked, 3).contiguous(),
            )
            for arr, grad in zip(outs, grads):
                wp.to_torch(arr.grad).copy_(grad)
            stream = wp.stream_from_torch(torch.cuda.current_stream(grads[0].device))
            try:
                with wp.ScopedStream(stream):
                    ctx.tape.backward()
            finally:
                if ctx.scratch is not None:
                    kin._release(ctx.scratch)
            result = []
            for arr in ctx.inputs:
                g = wp.to_torch(arr.grad)
                result.append(None if g is None else g.detach().clone())
            return tuple(result)

    return _Apply

# Rethinking Supermimic

## Installation

We use mjlab for its lightweightness. The robot model will be the G1 shipped with mjlab.

CUDA 12.6 is the default (`cu126` torch, Warp `1.18.0+cu12`) and matches a 12.8 driver:

```bash
uv sync
```

CUDA 13 (driver 580 or newer) is the other fork. It uses the PyPI torch and Warp wheels:

```bash
uv sync --no-default-groups --extra cu13
```

## Training

The 40 LAFAN1 clips are read from `aa-projects/aa-bfm/data/lafan1/raw/g1` (30 Hz) and cached at 50 Hz in `data/lafan1_g1.npz` on first use. mjlab supplies the G1 and the physics step; the world model and the policy update do not call the simulator.

```bash
PYTHONPATH=src python -m supermimic.train
PYTHONPATH=src python -m supermimic.eval --ckpt logs/lafan1/<timestamp>/ckpt_002000.pt
```

Hydra writes each run under `logs/<swanlab.experiment>/<timestamp>/` (ckpts, `train_log.csv`, `config.yaml`).

```bash
PYTHONPATH=src python -m supermimic.play
PYTHONPATH=src python -m supermimic.play --ckpt logs/lafan1/<timestamp>/last.pt
```

`play` opens a Viser page (default http://localhost:8080) with a ground plane to scrub LAFAN1 clips. With `--ckpt`, the solid robot is the policy and a translucent blue ghost is the reference.

Hydra selects the policy observation and the anchor. The config observation is Table A.1 plus the 6D reference-root orientation, and the anchor is the pelvis. `obs.base_lin_vel=true` adds body-frame linear velocity. `actor=trunk` concatenates the goal with the history. `actor=film` keeps that trunk on proprioception and lets the goal scale and shift each layer.

`swanlab.mode` is `offline` by default, which writes the run under `swanlog/` without uploading. `online` uploads, and `disabled` turns logging off. `local` needs the SwanLab dashboard extra, which is not installed here.

## Ours, the paper, and supermimic_vibed

`supermimic_vibed` is an independent Isaac Lab reproduction. We do not follow it. The shared core is what the paper actually fixes: Algorithm A.1, Table A.5, the Table A.1 observation with 10 frames of history, a K=10 reference goal into the encoder only, the Table A.2 reward without the self-collision term and with joint-limit weight −2, pelvis termination at `|e_z| > 0.25 m` or tilt `> 0.8`, sim context ξ from Table A.3, the Table A.4 widths with `z ∈ R^32` through `tanh`, analytic reward and termination, a deterministic policy, sensor noise added outside the world model, return-based 1 s bin resets (50% uniform / 40% softmax(−25 G_b/G_max) / 10% past softmax), and the Appendix F.1 evaluation.

The paper leaves the following unspecified. The table is how this repo and `supermimic_vibed` each fill them.

| Choice | Paper | Ours | supermimic_vibed |
|---|---|---|---|
| Dynamics state | `s_{t+1} = f(s_t, a_t, ξ)`. No coordinates. | Pelvis pose and twist plus joints. The MLP predicts a heading-frame increment by default (`wm_frame=heading`); `wm_frame=pelvis` writes the features and the position increment in the full pelvis frame. Both are conditioned on a per-body ground-contact probability that the model also predicts. | Same increment parameterization. |
| MLP details | Widths only. Training precision is TF32. | SiLU, no LayerNorm, no residual blocks, zero-initialized world-model output. TF32. A running mean and variance is shared across modules on each input they have in common: state features (world model and critic), goal (encoder and critic), ξ (world model and critic), plus the policy observation and the world-model action. `--wm-norm`, `--wm-residual`, and `--stochastic-actor` are optional. | World model: SiLU, LayerNorm, its own running input and output norm, bf16 matmuls. Policy and critic: ELU, each with its own running input norm. |
| World-model loss | Eq. 1 is an unweighted square on the state. An analytic head drops its term. | State term only, in the true state's local frame, divided by fixed nominal scales (0.25 m, 0.5 rad, 1 m/s, 3.14 rad/s, 0.5 rad, 5 rad/s). The same scales are the logged 8-step error. | State term only, divided by a running output scale with per-group floors. |
| Goal encoding | "K=10 future reference frames." No features, no frame, no decision about whether the current frame counts. | Frames `f+1` through `f+10`, in the current reference yaw frame: root displacement, 6D root rotation, root linear and angular velocity, joint position, joint velocity scaled by 0.1. | Frames `f+1` through `f+10`, same features, joint velocity scaled by 0.1. |
| Policy observation | Table A.1 only. | The config is Table A.1 plus the 6D reference-root orientation and the heading-frame root position error (`obs.anchor_ori`, `obs.anchor_pos`, pelvis anchor). Body-frame linear velocity is off. | Table A.1 plus a 6D reference-root orientation in the robot root. |
| Tracked bodies and anchor | Table A.2 says "following BeyondMimic" and writes the body errors without an anchor. Termination is the pelvis. | The 14 BeyondMimic bodies. Body position and orientation are yaw-and-xy anchored on `torso_link` by default (`anchor=torso`); `anchor=pelvis` switches the anchor. Root reward and termination use the pelvis. | The same 14 bodies. The anchor is the pelvis. |
| Critic input | `V(s, g, ξ)`. | Heading-invariant state, the goal, normalized ξ, and privileged tracking errors: joint error, anchored body position, heading-frame root position, and 6D reference-root orientation. | Invariant state, ξ, and privileged tracking errors (joint error, anchored body position, root error, relative orientation). |
| Joint offset | A component of ξ. It is not said whether it biases the encoder or the setpoint. | Encoder bias on the measured joint position. Friction, offset, and torso CoM are all redrawn every episode. | Added to the PD target. Friction and torso CoM are drawn once at startup; the offset is redrawn every episode. |
| Contacts | The self-collision *penalty* is removed. Simulator contacts are not discussed. | G1 self-collisions stay on. | Self-collisions disabled. |
| Episode boundary | Termination is height and tilt. Clip end and a time limit are not defined. | 10 s or clip end ends the episode and is bootstrapped, not terminal. Imagination uses a sigmoid around the hard thresholds (0.02 m, 0.04 tilt). | Same 10 s / clip-end bootstrap and a sigmoid termination. |
| Reset state | Bins only. BeyondMimic also perturbs the reference state. | BeyondMimic pose, velocity, and joint noise. Joints are clipped to the soft limits (0.9 of the range), which is also what the joint-limit penalty uses. | Same noise. Joints are clipped to the hard limits. Noisy actions are clamped to ±20. |
| When the policy starts learning | Both model and policy update every iteration. | Both update from the first iteration that contains an 8-step window. | 500 world-model-only iterations first. |
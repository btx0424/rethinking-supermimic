# Rethinking Supermimic

SuperMimic-style humanoid motion tracking: a learned world model (MLP), a policy trained by
backpropagating through short imagined windows, and a critic that bootstraps the window return.
mjlab supplies the G1 and the physics step; the world model and the policy update never call the
simulator.

## Installation

We use mjlab for its lightweightness. The robot model will be the G1 shipped with mjlab.

Torch / Warp are **not** in the base deps — pick one CUDA fork:

**CUDA 13** (driver 580+, this machine): PyPI torch ≥ 2.14 and Warp `1.18.0`:

```bash
uv sync --no-default-groups --extra cu13
```

**CUDA 12.6** (default group; older drivers): `cu126` torch and the Warp `+cu12` wheel.
Put the wheel at `wheels/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl`
(from the [Warp v1.18.0 release](https://github.com/NVIDIA/warp/releases/tag/v1.18.0);
GitHub release downloads often time out, so keep a local copy), then:

```bash
uv sync
```
## Datasets

Two motion sources are supported. Both resample to 50 Hz and cache into the repo-local
`<repo>/data/` directory on first use; both that directory and the run logs are git-ignored.

### LAFAN1 (whole-body tracking)

40 clips of a G1 retarget, 30 Hz CSV. Point `LAFAN1_ROOT` at a directory holding `raw/g1/*.csv`
and `meta.json`:

```bash
export LAFAN1_ROOT=/path/to/aa-bfm/data/lafan1
```

If the variable is unset, the loader looks in `<repo>/data/lafan1`, and if that is empty it
downloads through `aa_bfm.Dataloader` (HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset`). Either
way the resampled library is written to `<repo>/data/lafan1_g1.npz`.

### Weave (object interaction)

Hand–object clips for the inspire-hand G1, 50 Hz, with a free-joint object. The dataset is not
auto-downloaded, so `WEAVE_ROOT` is required:

```bash
export WEAVE_ROOT=/path/to/Weave      # contains the rollout folders and objects/<name>/<name>.obj
```

One cache per object set is written to `<repo>/data/weave_g1_hand_<objects>.npz`. The robot MJCF
ships in `robot_assets/unitree_g1/`, so no external asset path is needed.

### Visualization

Scrub the reference motions in a browser (Viser, default http://localhost:8080):

```bash
PYTHONPATH=src python -m supermimic.utils.viz_motion --dataset lafan1
PYTHONPATH=src python -m supermimic.utils.viz_motion --dataset weave --objects smalltable smallbox
PYTHONPATH=src python -m supermimic.utils.viz_motion --dataset weave_xhand --objects smalltable
```

Useful flags: `--port`, `--hz` (viewer refresh; clips always step at 50 Hz), `--split {train,test}`
and `--max-clips` (object datasets only). The same viewer is reachable programmatically as
`Lafan1Motion.viser_play()` / `WeaveMotion.viser_play()` / `WeaveXHandMotion.viser_play()`.

### Weave XHand (newer release)

The XHand release is a separate dataset with a different robot and primitive-fitted object
colliders. Its robot is committed as a ready-to-load MuJoCo model,
`robot_assets/unitree_g1/g1_29dof_rev_1_0_xhand.xml`, with XHand-specific meshes (hands and
wrist adapter) in `robot_assets/unitree_g1/meshes/g1_xhand/` — the G1 body meshes it needs were
already in the parent `meshes/` directory byte-identically, so nothing is duplicated. The vendored
URDF sits next to the XML as the source of truth; `python -m supermimic.robots.g1_xhand`
regenerates the XML from it. Only the motions and object assets are external:

```bash
export WEAVE_XHAND_ROOT=/path/to/G1-Xhand-OMOMO   # contains assets/ and motions/
```

It is 8 objects × ~250–370 clips at 30 Hz, resampled to 50 Hz and cached per object and split as
`data/weave_xhand_<objects>_<split>.npz`. The release has no official train/test split, so every
fifth clip (20%) is held out for evaluation (`split=test`). Objects use the release's primitives
(boxes / cylinders / capsules) for collision instead of the visual mesh, with density 200 kg/m³
(25 for `largebox`) — 0.16–6.6 kg per object.

## Train, play, evaluate

### Train

```bash
PYTHONPATH=src python -m supermimic.train
PYTHONPATH=src python -m supermimic.train algo=nablamimic
PYTHONPATH=src python -m supermimic.train task=weave
PYTHONPATH=src python -m supermimic.train task=weave_xhand task.object_name=smallbox
# Robot-only sanity check: same clips, no object in the scene or the rewards.
PYTHONPATH=src python -m supermimic.train task=weave_xhand task.has_object=false
```

Hydra writes each run under `logs/<swanlab.experiment>/<timestamp>/` (`ckpt_*.pt`, `train_log.csv`,
`config.yaml`). `algo=` selects the trainer (see [Algorithms](#algorithms)).

`task=` selects the task config group (`lafan1`, `weave`, `weave_xhand`), and everything about it is
nested under the same node:

| key | meaning |
|---|---|
| `task.name` | which task spec to build |
| `task.object_name` | object task only: which object's clips and collider |
| `task.split` | `weave_xhand` only: `train` or the held-out `test` |
| `task.simple_hands` | `weave_xhand` only: fitted primitive hand colliders |
| `task.has_object` | `false` tracks the robot only; object terms are dropped and the run logs to `<task>_noobj` |

Any other config key can be overridden on the command line, e.g. `wm_ema=null`, `imagine_wm=ema`,
`iterations=5000`, `save_optim=true`.

### Play

Interactive rollout of the reference and, with a checkpoint, the learned policy:

```bash
PYTHONPATH=src python -m supermimic.play                          # reference only
PYTHONPATH=src python -m supermimic.play --ckpt logs/lafan1/<timestamp>/last.pt
PYTHONPATH=src python -m supermimic.play --task weave             # reference, no ckpt
```

`play` opens a Viser page with a ground plane. With `--ckpt`, the solid robot is the policy and a
translucent blue ghost is the reference; the task is inferred from the checkpoint. Flags: `--port`,
`--device`, `--hz`.

### Evaluate

```bash
PYTHONPATH=src python -m supermimic.eval --ckpt logs/lafan1/<timestamp>/ckpt_002000.pt
```

Plays each clip from its first frame and reports the normalized return (Appendix F.1 protocol).
By default it evaluates the EMA weights; `--live` uses the online policy instead. `--device` and
`--quiet` are also available.

## Algorithms

`algo=supermimic` and `algo=nablamimic` share the whole world-model stack — replay, WM fitting,
imagined rollouts, the reward/termination model, the critic, and EMA checkpoints. They differ
only in the **policy improvement operator**: how the imagined window turns into a policy update.

| | `supermimic` | `nablamimic` |
|---|---|---|
| Idea | SuperMimic's BPTT: differentiate the window return w.r.t. policy parameters and take one Adam step. | ∇Mimic / PODS: the gradient defines *target actions*, and the policy is regressed onto them. |
| Gradient | `∂J/∂θ` through the WM and the policy. | `∂J/∂a` w.r.t. actions only; the policy runs under `no_grad` and its parameters get no direct gradient. |
| Update | One optimizer step on the window gradient. | Action targets `y = a + η·g/RMS(g)` with a fixed RMS step `δ` rad, then `M` Adam regression steps on `‖π(o) − y‖²` (joint radians). |
| Tail control | None by default (`grad_clip_norm` only). | Per-sample clip of `‖∂J/∂a‖` to `act_cot_clip` × batch median. |
| Reuse of a window | Once. | `action_target_steps` regression steps per window (default 8), from a ring buffer of `action_target_windows`. |
| Key knobs | `pi_lr`, `grad_clip_norm`, `pi_warmup` | `action_target_steps`, `action_target_delta`, `act_cot_clip`, `action_target_lr` (also sets `pi_lr`), `action_target_batch` |

Practical notes:

- `nablamimic` sets `pi_lr = action_target_lr` (default `6e-4`) and `grad_clip_value = 0.1` in
  `conf/algo/nablamimic.yaml`; everything else starts from the `supermimic` defaults.
- The `imagine_wm` switch (`live` | `ema`) applies to both: it picks which world model supplies the
  policy-training gradients. WM fitting always updates the live model.
- Both are world-model methods: gradients come from the learned model, not the simulator.
  `nablamimic` borrows only the improvement operator of ∇Mimic, which pairs it with exact
  simulator gradients ([NablaEngine](https://github.com/OamIcs/NablaEngine)).

## Configuration notes

Hydra selects the policy observation and the anchor. The config observation is Table A.1 plus the
6D reference-root orientation, and the anchor is the pelvis. `obs.base_lin_vel=true` adds
body-frame linear velocity. `actor=trunk` concatenates the goal with the history. `actor=film`
keeps that trunk on proprioception and lets the goal scale and shift each layer.

`swanlab.mode` is `offline` by default, which writes the run under `swanlog/` without uploading.
`online` uploads, and `disabled` turns logging off. `local` needs the SwanLab dashboard extra,
which is not installed here.

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
| Episode boundary | Termination is height and tilt. Clip end and a time limit are not defined. | Termination is height and tilt. The time limit (10 s) and clip end *truncate* instead: they are bootstrapped, not terminal. Imagination uses a sigmoid around the hard thresholds (0.02 m, 0.04 tilt). | Same 10 s / clip-end bootstrap and a sigmoid termination. |
| Reset state | Bins only. BeyondMimic also perturbs the reference state. | BeyondMimic pose, velocity, and joint noise. Joints are clipped to the soft limits (0.9 of the range), which is also what the joint-limit penalty uses. | Same noise. Joints are clipped to the hard limits. Noisy actions are clamped to ±20. |
| When the policy starts learning | Both model and policy update every iteration. | Both update from the first iteration that contains an 8-step window. | 500 world-model-only iterations first. |

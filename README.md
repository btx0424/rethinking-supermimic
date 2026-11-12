# G1 LAFAN world-model tracking

Train an explicit or latent maximal-coordinate world model and an explicit actor
on the prepared G1 LAFAN dataset. Both models use the same simulator, physical
state, observations, rewards, replay and policy update.

## Layout

```text
src/supermimic/
  dataset/motion.py           # complete MotionData TensorClass + clip metadata
  env/
    sim/env.py                # G1 MuJoCo simulation, reset and domain randomization
    mdp/
      config.py               # tracking bodies, observation groups, reward/termination settings
      command.py              # explicit motion identity/time and shared advancement
      reference.py            # task body selection and reset bins
      observation.py
      reward.py
      termination.py
      state.py                # physical term inputs, no latent variables
      geometry/               # common MaxCoord geometry/integration
      tracking.py             # task composition
    rollout/                  # common policy/transition interface
  algo/supermimic/
    wm/                       # interface, explicit/latent models, imagined rollout
    policy/                   # explicit actor and critic
    trainer.py                # collection, VecNorm and training orchestration
    updates.py                # WM loss, differentiable actor rollout, critic updates
    replay.py                 # raw physical transitions and command/episode state
    capture.py                # CUDA Graph updates with refreshed inputs
    checkpoint.py
  robots/g1.py                # mjlab G1 asset adapter
  nn/, utils/                 # shared numerical, optimizer and logging utilities
scripts/train.py, eval.py, check_training.py, check_acceleration.py
```

Dataset owns all available body/joint data. `MotionData` is a TensorClass with
frame batch dimension; `get_slice(motion_id, motion_t, steps)` returns typed
batched windows, clamped to each clip. Dataset knows no tracking-body roles,
reward weights or reset distribution. The task selects bodies without mutating
the dataset. No object/dex-hand tasks, latent actors, IK reward or specialized
foot/hand matching remain.

## Runtime and data

```bash
uv sync
```

Pinned runtime: Python 3.11–3.13, Torch 2.11 CUDA 12.8, mjlab 1.6,
MuJoCo/mujoco-warp 3.11, Warp 1.16 and TensorDict 0.13. CUDA is required for
simulation and CUDA Graph training.

Supply `data/lafan1_g1.npz` or override `task.motion`. The prepared cache contains
40 G1 clips at 50 Hz, 29 joint coordinates and named body poses/twists. Required
fields are documented in `dataset/motion.py`; orientations are wxyz quaternions.
This branch consumes the prepared cache and does not download or retarget source
motions. The cache and training outputs are ignored by Git.

## Training

From this repository:

```bash
uv run scripts/train.py wm_kind=explicit wandb.mode=online
uv run scripts/train.py wm_kind=latent wandb.mode=online
```

Use `--config path.yaml` for a YAML configuration, followed by `key=value`
overrides. See `TrainConfig` and `TaskConfig` for the full schema. For example,
`task.reward_weights.root_pos=0.0` disables that reward's contribution;
`task.actor_observations=[features,goal]` selects actor input terms. The default
critic additionally sees `xi_n`. Observation composition determines network and
VecNorm input sizes. W&B online is required for formal training; short local
checks use disabled logging.

Defaults: 4096 sim environments, collect 4 steps/iteration, a 1000-step replay
per environment, 10 WM batches of 2048 eight-step sequences, then 4096 imagined
starts with horizon 32 for one actor update and 20 critic minibatch updates
(5 epochs over the 131,072 imagined rows, 4 minibatches each).
WM windows never cross episode resets; actor starts have enough reference frames
for the imagined horizon. Replay capacity is 4,096,000 transitions (250 four-step
collect calls once full). Collection continues from the preceding collect;
only ended episodes reset.

The goal term gives, for each of the next 10 reference frames, the reference
root position and orientation in the current robot root frame
(`ref_root_pos_future_b`, rot6d) plus reference joint positions. Actor and
critic both receive it; there is no separate privileged group.

The actor always consumes explicit physical features. The latent WM trains both
integrated physical-state loss and latent consistency against a stopped-gradient
next-state encoder target. Its consistency weight is configurable. All network
inputs pass through algo-owned VecNorm, updated only from real collect samples;
replay stores raw quantities. Latent SimNorm operates on latent outputs.

Both rollout paths share command advancement, observation/reward terms and
physical representation. Reward scores the successor against the next reference
frame. State includes command `(motion_id, motion_t)`, previous action, raw sim
context and episode step. Imagined starts restore these from replay rather than
resampling reference poses. No term memory is allocated. Real termination resets
selected environments; imagination uses soft survival weights without resets or
simulator timeout. Terminal successors remain separate from reset continuations.

CUDA Graph captures WM and actor/critic updates after warmup. Simulation collection
stays eager. The experimental torch.compile path was removed. Logging separates
`SimReward`, `WMReward`, `WMLearning`, `ActorCriticLearning`, and `Performance`.
Every reward component is logged before dt multiplication; training uses dt=0.02.
Every iteration records collect/train timing, including graph execution.

## Checks and evaluation

```bash
uv run python -m unittest discover -s tests -v
uv run scripts/check_training.py --wm explicit --output /tmp/g1-explicit-check
uv run scripts/check_training.py --wm latent --output /tmp/g1-latent-check
uv run scripts/check_acceleration.py --wm explicit --full-size --output /tmp/g1-explicit-graph
uv run scripts/check_acceleration.py --wm latent --full-size --output /tmp/g1-latent-graph
uv run scripts/eval.py --checkpoint /path/to/last.pt
```

Output directories must be fresh. Eval is a native LAFAN diagnostic, not the
MotionDecode deployment benchmark. Resume with `resume=/path/to/last.pt` and a
fresh `run_dir`, keeping architecture/task settings identical. Resume restores
models, optimizers and normalization; replay/sim collection starts fresh, so it
is not bitwise continuation of the old rollout.

## Cleanup compatibility

The pre-cleanup implementation is preserved at commit `49a4d12`, branch
`maxcoord-before-cleanup-20261111`. Old checkpoint formats and legacy playback
live there. This branch uses `g1_lafan_v1` plus its explicit task/state protocol;
it rejects older protocols instead of guessing their normalization or inputs.

G1 assets come from mjlab. Friction randomization now applies to all robot
collision geoms selected by collision masks, not named feet. This is a deliberate
protocol change; the cleanup is not an exact training reproduction of the old
branch. Existing remote training snapshots and datasets were not changed.

See `THIRD_PARTY.md` for the retained MaxCoord attribution.

# MaxCoord world models

`algo=supermimic_maxcoord` is independent of the existing trainers and replay
format. It currently supports the LAFAN1 G1 task (14 tracked bodies) only.

| WM | Actor input (critic also gets xi_n) | Configuration |
| --- | --- | --- |
| Explicit | `f_t + goal` | `wm_kind=explicit actor_kind=explicit` |
| Latent | `z_t + goal` | `wm_kind=latent actor_kind=latent` |
| Latent | `f_t + goal` | `wm_kind=latent actor_kind=explicit` |

The complete state is each body's world position, wxyz quaternion, linear
velocity, and angular velocity. `f_t` has 214 channels: pelvis-relative body
positions and 6D orientations, body twists expressed in pelvis axes, root height,
and projected gravity. World root pose is carried separately in seven auxiliary
channels. Local velocities are rotated world velocities, not body-minus-root
velocities. The 84 output channels are body linear/angular velocity increments
for one control step, expressed in the current pelvis frame.

Explicit dynamics predicts `d_t = dynamics(f_t, a_t, xi_n)`. The latent model uses
`z_t = SimNorm(encoder(f_t))`, `z_{t+1} = SimNorm(dynamics(z_t, a_t, xi_n))` and
`d_t = decoder(z_t, a_t, xi_n)`. Only initialization encodes the observed state;
imagined latent states are advanced autoregressively. Both models integrate
`d_t` with Hybrid's MaxCoord equations. Pose integration uses `v + 0.625 * dv`
at four simulator substeps; twists advance by the full delta, without another
factor of dt. All body features are then expressed in the new pelvis frame.

World-model fitting uses eight recorded actions and eight successor states.
Body states and twists are reconstructed by FK from the unchanged replay
buffer's root/joint states. Supervision is on integrated states, never on delta
outputs. Root position/orientation and non-root body position/orientation/linear
velocity/angular velocity use nominal errors `0.01 / 0.02 / 0.05 / 0.1 / 1 / 3.14`.
The loss is the weighted mean over channels, as in Hybrid, rather than an equal
mean of six group losses. Root orientation uses `R / sqrt(2)`; body orientation
uses the first two matrix columns in row-interleaved order.

Latent models additionally use
`consistency_weight * MSE(z_pred, stop_gradient(SimNorm(encoder(f_true_next))))`
over all eight successors. The configurable initial weight is `1.0`; its raw
and weighted contributions are logged separately. Encoder gradients still flow
through the rollout initialization and state loss.

Goals contain the next ten frames of reference displacement, relative 6D
orientation, and reference joint positions (380 channels). Displacement is
future-reference minus current-reference position, expressed in current
reference pelvis axes; it contains no robot/reference position error.
Orientation is future-reference relative to the current robot pelvis. Future
frames clamp at the clip end. Critic receives the normalized actor inputs plus fixed-normalized simulator context
by default; actor inputs remain unchanged.

Actor training backpropagates through 32 imagined transitions with analytic
rewards and soft termination. WM and critic weights are frozen during actor
backward; state/action gradients remain live. The critic learns per-head
lambda returns. `joint_limit` is removed consistently from this algorithm's
collection, imagination, evaluation, and playback. Predicted body kinematics
go directly into rewards; there is no imagined FK or inferred joint angle.
Other algorithms' rewards and replay buffers are unchanged.

## Run

Use the existing CUDA-specific environment installation instructions first.
For CUDA 12, place the pinned Warp wheel under `wheels/` before `uv sync`.
File-path entry points avoid the nested `supermimic/supermimic` package shadowing
that occurs when executing `src/supermimic/train.py` directly.

```bash
uv run scripts/prepare_lafan1.py --root data/lafan1 --source /path/to/G1/CSVs
uv run scripts/check_maxcoord.py --reference-root /path/to/HybridRobotics/supermimic
uv run scripts/check_maxcoord_sim.py

# Small eager smoke; uses the normal eight-action and 32-step horizons.
uv run scripts/train_maxcoord.py wm_kind=latent actor_kind=latent \
  num_envs=16 iterations=4 buffer_steps=64 wm_batches=1 wm_batch=16 pi_batch=4 \
  wm_hidden='[32,32]' actor_hidden='[32,32]' latent_dim=16 \
  critic_epochs=1 critic_minibatches=1 pi_warmup=0 \
  compile=false compile_mdp=false swanlab.mode=disabled log_every=1

uv run scripts/eval_maxcoord.py --ckpt /path/to/last.pt --quiet
```

For a persistent prepared environment, `.venv/bin/python scripts/train_maxcoord.py`
also works without invoking dependency resolution. Normal runs use the existing
SwanLab logger, TensorBoard, and `train_log.csv`. A requested non-disabled logger
requires the SwanLab SDK; this trainer does not silently continue without it.
Check authentication/connectivity before formal experiments. `log_backend=wandb`
selects W&B instead, with independent `wandb.project` and `wandb.mode` fields. `online_run.json` records the W&B ID/URL.

New checkpoints have an explicit `maxcoord_v3` format tag, model/actor configuration,
normalizers, critic, reward scale, and (by default) optimizer state. `resume=...`
restores v2/v3 checkpoints with matching architecture/collection protocol
and critic context setting and the
iteration, but starts a new simulator and fresh replay;
it is not a bitwise continuation of an interrupted trajectory. Eval and the
existing visual playback loader recognize the tag before legacy key heuristics.

## Validation and limits

The CPU checker compares conversion and 32-step integration against upstream in
an isolated subprocess, checks identity-rotation gradients, loss weighting,
target stop-gradient, decoder timing, and finite nonzero 8/32-step actor
gradients for all three combinations. The simulator checker reports FK agreement
and six-group integration residuals using true velocity increments.

Short smoke tests verify execution, not learning quality. No full training
comparison, learned-policy visual assessment, or conclusion about why earlier
latent experiments failed follows from these checks. Long-term compiled-mode
learning equivalence still requires separate measurement.

## Optional learner acceleration

`cuda_graphs=true` captures each full group of ten WM updates and the complete
actor/critic update, including backward and optimizer steps. Simulation, replay
sampling and random critic permutations stay outside capture; fresh inputs are
copied into static storage every iteration. Three real warmup updates precede
capture. Policy capture waits until actor learning-rate warmup has finished.
Normalization retains its per-minibatch update order. Device-side asynchronous
finite assertions replace host-synchronizing loss/gradient checks.

Both new flags default to false. To try graph-only, append
`cuda_graphs=true compile_updates=false` to the normal training command.
`compile_updates=true` additionally compiles FK/normalization, WM loss, actor
transitions, bootstrap and critic kernels. Use `fk=torch` and CUDA for this
path; tensor shapes and parameter/optimizer storage must remain fixed after
capture. Do not mutate the captured optimizer's Python hyperparameters.

On RP3 RTX4090, the full-size latent-WM/explicit-actor graph-only path passed
two consecutive refreshed-input updates against the original eager optimizer.
Twenty measured iterations after eight warmup iterations averaged 4.712 s for
eager, 1.048 s for graph-only and 1.002 s for compile + graph, with the same
GPU/CPU affinity. This was a shared-host short benchmark, not a learning-curve
comparison. Collection became the largest stage. Initial compilation and graph
capture are excluded from those steady-state numbers.

**Compile + graph remains experimental:** one full-size actor update differed
by 2.269% in relative update-vector norm, above the checker's 2% threshold,
despite small gradient/loss differences. Do not treat it as verified equivalent
or infer a 2.269% loss regression. The healthy formal run was left unchanged.
`scripts/check_maxcoord_acceleration.py` records numerical comparisons; task
records contain the timing curves and all failed as well as successful trials.

Upstream attribution and license: [THIRD_PARTY.md](THIRD_PARTY.md).


## Review protocol (v2)

New runs condition explicit dynamics on `[norm(f), norm(a), normalize_xi(xi)]`
and latent dynamics/decoder on `[z, norm(a), normalize_xi(xi)]`. The encoder
remains feature-only, and actor inputs are unchanged; the critic now additionally receives xi_n. Replay windows use
xi from their first row; imagination carries that constant context in RolloutState.
Collection and inference initialize it from the simulator. SimNorm is applied
after both encoder and dynamics (group size 8; must divide latent_dim). The
consistency target remains stop-gradient. Entropy, max group probability and
predicted/target cosine are logged alongside consistency MSE.

`collect_steps=4` controls collection independently of learning batch sizes.
The shared legacy `utd` field is unused and rejected unless it retains 0.4.
Initial collection remains max(2*wm_window,32). With 1024 envs this adds 4096
transitions/iteration and the 1000-row replay spans 250 iterations. This differs
from the original running 13-step protocol and Hybrid's average 2.5 steps.
Goal displacement uses the full current reference root rotation, including
pitch and roll, deliberately rather than yaw alone.

The previous protocol used maxcoord_v2; new checkpoints use maxcoord_v3. Old v1 eval/play explicitly reconstruct xi-off,
SimNorm-off models; v1 resume is rejected instead of silently changing physics
inputs, latent geometry or sampling protocol. Start a fresh v2 run.

Logging uses mimic-lite-style slash namespaces: SimReward and WMReward each
contain every weighted reward component and total; WMReward also distinguishes
normalized_total. Components and totals are weighted means BEFORE multiplying by dt, not
episode sums. WM normalized_total is also displayed before dt; learning rewards
and returns retain the original dt scaling. WMLearning holds state/MSE/latent
losses and diagnostics; ActorCriticLearning holds losses and gradient norms.
Performance holds iteration/time. CSV keeps its flat monitoring schema.
For W&B use `log_backend=wandb wandb.mode=online wandb.project=...`; SwanLab
retains its own settings. No W&B setting is taken from swanlab.mode/project.

The v2 formal run uses an isolated RP7 copy of the verified RP3
active-adaptation runtime (Python3.12, Torch2.11+cu128, Warp1.16), frozen
separately from the Python>=3.13 project lock.
That lock describes a different supported installation, not the runtime used
for the RP3 measurements. Runtime package versions are saved with the new run.


## Unified WM interface and performance logs

`WorldModel` owns `initialize(full, xi_n)`, `forward(state, action)`,
`rollout(state, actions)`, `compute_loss(WMBatch)` and
`update_normalization(WMBatch)` (or the existing full/actions form). `train(mode)`
still switches PyTorch mode; optimizers and CUDA Graph capture belong to the
learner. Goal, frame, reward, termination and previous action stay in the
tracking environment/learner, outside the physical model.

`WMState` (compatible alias of `RolloutState`) carries aux, f, optional z and xi_n;
features/latent properties expose the descriptive names. `WMTrajectory` retains
initial state and B×H successor aux/features/optional latents; `.final` gives its
last state. NamedTuple containers support compile/CUDA pytrees. The latent decoder
still reads z_t, not z_{t+1}; target encoding is under no_grad. Both learner paths
share the same model-owned loss, with prepared-target methods supporting compile.
LossOutput metrics are live tensors; detach them when retaining diagnostics.

`critic_xi=true` appends 33 channels (friction, 29 encoder biases, 3 COM offsets)
to each head's input. Explicit actor critic width becomes 627, latent actor critic
width 669 for G1 and the ten-frame goal. Context stays constant along imagination.
This does not add official-style robot-relative reference position to the critic.
Legacy v2 resume is available with `critic_xi=false`; adding context requires a
fresh run rather than silent parameter padding. v1/v2/v3 policy-only loading
remains compatible because the actor architecture did not change.

Each iteration logs `Performance/collect_time`, `Performance/train_time` and
`Performance/compute_time` in seconds to the selected logger and TensorBoard,
and writes performance.csv. Model metrics retain log_every. Collect covers real
sampling and replay insertion; train includes replay indices/sampling, WM and
actor/critic updates. CUDA event spans include GPU execution and CPU launch gaps,
with one synchronization after training; compute_time is wall time through that
boundary. Initial replay warmup, logging and checkpoint I/O are excluded. This
instrumentation adds a synchronization per iteration; it is not a profiler trace.

# Changelog

## 2026-11-12

### Explicit optimizer comparison controls

- Add critic optimizer selection and actor elementwise gradient clipping after
  global norm clipping, including CUDA Graph execution. Defaults retain pure
  Adam critic and no value clipping. Resume rejects incompatible settings.
- Add explicit `muon_adam` selection for official zero-decay Adam splitting;
  existing `muon` retains its AdamW behavior. Actor betas remain 0.9/0.95.
- Let training and eager/graph checks accept TrainConfig overrides.
- Validation: 12 local CPU unit tests pass; Ruff checks changed files and
  repository E4/E7/E9/F rules. Remote full-size checks recorded in task artifacts.


### G1-only modular cleanup

- Keep explicit/latent world models and explicit actor only. Remove legacy
  algorithms, multi-task/object/dex-hand assets, IK and specialized foot/hand
  matching, old training/playback scripts and experimental torch.compile paths.
- Organize real simulation under env/sim, shared MDP terms and physical geometry
  under env/mdp, and all WM/policy learning under algo/supermimic. Both rollout
  paths share observations/rewards/command progression. Latent continuation is
  internal to the WM adapter; term-facing PhysicalState has no latent fields.
- Full MotionData uses TensorClass (TensorDict 0.13), with typed batched get_slice
  windows clamped to clip bounds. Dataset is independent of task selection;
  TrackingReference selects/reorders configured bodies without mutating it.
- TaskConfig controls observation groups, reward weights/sigmas and termination.
  Default input/reward semantics, collect-only VecNorm, raw replay, per-component
  pre-dt logging and per-iteration collect/train timing are retained.
- Domain randomization selects all robot collision geoms rather than named feet.
  Only the mjlab G1 asset and prepared LAFAN NPZ input path remain.
- Validation: seven unit tests; explicit/latent six-iteration simulator/graph
  smokes; both formal-size eager/graph comparisons (maximum relative update
  error 0.05553% / 0.03911%); CLI checkpoint resume and native 40-clip eval.
  These are correctness checks, not a training-quality comparison.
- Pin the validated runtime and document the new CLI, state/replay contract and
  checkpoint compatibility. Preserve old source at 49a4d12 on branch
  maxcoord-before-cleanup-20261111. Existing remote snapshots are untouched.

## 2026-11-11

### Unified WM interface and critic context

- Share initialize/forward/rollout/compute_loss/update_normalization through
  WorldModel and typed WMState/WMBatch/WMTrajectory/LossOutput containers.
  Latent-specific supervision now lives in the model; eager and accelerated
  learners use the same implementation. nn.Module.train(mode) is unchanged.
- Critic inputs append fixed-normalized simulator context by default
  (`critic_xi=true`); actor goal architecture, two value heads, reference-position
  inputs and reward normalization are unchanged. Preserve xi in rollout state
  independently of whether the WM itself conditions on it.
- New checkpoints use maxcoord_v3. v1/v2/v3 eval/play remain supported; legacy
  v2 resume requires critic_xi=false instead of silently changing critic shape.
- Every iteration records Performance/collect_time, train_time and compute_time
  in seconds, plus performance.csv, regardless of log_every. CUDA events and
  one end-of-iteration synchronization include actual graph execution. Timing
  excludes logging/checkpoint writes; train includes replay preparation.
- CPU frozen-v2 comparisons preserve exact WM outputs/losses/gradients; nine
  legacy/new policy cases agree exactly. Both WM types pass full-size graph
  checks with critic context; three six-iteration graph smokes verify per-step
  timing and reward logging. Existing remote training snapshots are unchanged.

### Review follow-up

- Acceleration checker accepts explicit/latent WM and actor selections, so the
  full-size explicit training path can be checked independently before launch.
  RP7 full-size explicit graph check passes; maximum relative update error
  0.1038%. A separate explicit LAFAN40 graph run is online (4svs7xbx).

- New MaxCoord v2 world models condition on fixed-normalized physics context
  (xi); the encoder remains feature-only. Encoder/dynamics latents use SimNorm
  (default group size 8), with entropy/max-probability/cosine diagnostics.
- Collection now uses exactly `collect_steps=4`; legacy `utd` accepts only its
  unused default. Keep initial 32-step fill. Full reference orientation remains
  intentional and is covered by a pitch regression check.
- Separate SimReward, WMReward, WMLearning and ActorCriticLearning logging
  sections. Both real and imagined reward terms are reported individually,
  alongside totals; raw terms are weighted means before multiplying by dt. CSV retains
  flat names for monitoring. W&B has independent project/mode config and a
  declared dependency.
- v1 checkpoints retain eval/play support with xi off and SimNorm disabled;
  training resume requires v2 and matching architecture/collection protocol.
  Eager updates also use asynchronous finite assertions.

- Validation: xi/SimNorm/full-orientation and 8/32-step gradient checks pass;
  RP7 six-iteration graph smokes pass for all three combinations, with exactly
  four collection steps and seven pre-dt reward components summing to total.
  Full-size two-update graph comparison passes (maximum relative parameter
  update error 0.0856%); v1 eval/play loading passes. Fresh online v2 training
  launched on RP7 GPU1 with an isolated copy of the verified Torch2.11 runtime.

### Added

- Opt-in MaxCoord `cuda_graphs` and `compile_updates` learner paths, with
  refreshed static inputs, capturable optimizer state, asynchronous finite
  checks and a two-update correctness checker. Sampling stays outside capture;
  both flags default to false.

- MaxCoord can explicitly select W&B online logging with `log_backend=wandb`;
  the existing SwanLab default is preserved. Run ID/URL are saved beside logs.

- Independent `supermimic_maxcoord` trainer with Hybrid-compatible per-body
  representation/integration, explicit and latent world models, and separately
  selectable explicit/latent actors. Eight-action fitting supervises integrated
  states; latent dynamics also matches stop-gradient encoder targets.
- Shared ten-frame goal, 32-step actor BPTT, per-head lambda-return critic,
  checkpoint/resume, and explicit checkpoint dispatch in eval/play. This trainer
  removes joint-limit rewards consistently from real and imagined trajectories.
- File-path training/evaluation/data-preparation entry points, numerical and
  real-transition checks, architecture documentation, and upstream attribution.

### Fixed

- Declare Hydra, einops, SwanLab and TensorBoard runtime dependencies. Regenerate
  the lock file to include these and match the existing local CUDA 12 Warp source.
- Package Hydra config so the imported file-path training entry point works.

### Validation

- Full-size refreshed-input CUDA Graph checks pass against the original eager
  optimizer. Same-GPU/CPU-affinity 20-step benchmarks average 4.712 s eager,
  1.048 s graph-only and 1.002 s compile + graph. Combined compilation remains
  experimental: actor update-vector error 2.269% exceeds the preset 2% tolerance.
  Short checks do not establish long-term learning equivalence; the existing
  healthy formal run and its frozen source were not changed.
- CPU round-trip maximum error `1.43e-6`; 32-step integration against Hybrid
  `5efbf114` differs by at most `1.31e-6`. Three combinations pass finite nonzero
  8/32-step gradient checks, decoder timing and target stop-gradient checks.
- RP5: all three combinations and the unchanged baseline complete four eager
  smoke iterations with Muon and save checkpoints. Real-transition check passes;
  replay FK feature error is at most `4.77e-6`.
- Local latent resume, 40-clip eval loading, play loader with two physical steps,
  and five existing replay regressions pass. Small explicit compiled-transition
  training also passes; whole-loss compilation was replaced by per-step compilation
  after a missing-gradient failure on the local Torch 2.11 runtime.
- Smoke results establish execution correctness only. A separate formal
  latent-WM/explicit-actor LAFAN-40 run is active; learned-policy comparisons
  have not been completed.

## 2026-11-08

### Fixed

- Replay windows now include the last transition of an episode, including its
  terminal successor, without crossing episode boundaries. This applies to all
  algorithms using the shared SuperMimic replay buffer.
- SuperMimic Latent defaults to the live world model. Collection and simulator
  critic targets now use the same selected encoder as imagination, including
  when `imagine_wm=ema` is explicitly requested.
- Latent checkpoint playback selects the encoder used during training instead
  of always using EMA weights.

### Added

- A persistent changelog and repository instruction to maintain it.
- CPU regression checks for replay windows and terminal successors.

### Validation

- All five CPU replay regression tests passed, covering terminal successors,
  single-step windows, exact-length buffers, ring-buffer wrapping, and step gaps.
- Checked live/EMA checkpoint encoder selection, including legacy fallbacks.
- Changed Python files compile successfully; `git diff --check` passed.
- Full training was not run locally: the environment lacks `omegaconf`.

### Explicit rollout command and episode state

- Store motion id and clip-local integer motion time in CommandState; sim and
  imagination share MotionCommand.advance and clip-end clamping.
- Replay preserves paired physical state, command, previous action, raw simulator
  context and episode step. Imagined reset restores these fields without
  resampling a motion or resetting the robot to its reference pose.
- Preserve pre-reset successors separately from reset continuation states;
  partial resets clear only selected episode counters. Simulator timeout stays
  500 steps, while imagination retains soft-survival/no-timeout behavior.
- No term-memory container or history-dependent reward was added.
- Validation: five CPU unit tests and explicit/latent six-iteration CUDA Graph
  training checks pass, including mixed-clip restore, partial reset, raw replay,
  collect-only normalization, reward component logging and checkpoint equality.

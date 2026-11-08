# Changelog

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

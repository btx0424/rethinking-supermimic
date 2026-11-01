"""SwanLab experiment logging with a no-op fallback when the SDK is unavailable."""

from __future__ import annotations

from typing import Any

_active = False


def init(
    *,
    project: str,
    name: str | None,
    config: dict[str, Any],
    mode: str,
) -> None:
    global _active
    _active = False
    if mode == "disabled":
        return
    try:
        import swanlab
    except ImportError as exc:
        print(f"swanlab unavailable ({exc}); training continues without SwanLab.", flush=True)
        return
    swanlab.init(project=project, name=name, config=config, mode=mode)
    _active = True


def log(metrics: dict[str, Any], *, step: int) -> None:
    if not _active:
        return
    import swanlab

    swanlab.log(metrics, step=step)


def finish() -> None:
    global _active
    if not _active:
        return
    import swanlab

    swanlab.finish()
    _active = False

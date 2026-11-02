"""Viser playback for LAFAN1 or Weave reference motions."""

from __future__ import annotations

import argparse

from supermimic.motions import get_motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("lafan1", "weave"),
        default="lafan1",
        help="Which Motion loader to visualize.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hz", type=float, default=30.0, help="Viewer refresh rate.")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["smalltable", "smallbox"],
        help="Weave object subset (ignored for lafan1).",
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=200,
        help="Cap clips loaded for Weave (lafan1 uses the full cache).",
    )
    args = parser.parse_args()

    motion = get_motion(args.dataset)
    session_kwargs: dict = {}
    if args.dataset == "weave":
        session_kwargs = {
            "objects": tuple(args.objects),
            "split": args.split,
            "max_clips": args.max_clips,
        }
    motion.viser_play(port=args.port, hz=args.hz, **session_kwargs)


if __name__ == "__main__":
    main()

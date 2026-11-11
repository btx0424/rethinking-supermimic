"""Prepare LAFAN1 metadata/cache from the Unitree G1 retarget CSV directory."""
import argparse
import json
from pathlib import Path

# CSV order documented by Unitree's LAFAN1_Retargeting_Dataset README.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Optional existing CSV directory to symlink locally")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    raw = args.root.resolve() / "raw" / "g1"
    if args.source and not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.symlink_to(args.source.resolve(), target_is_directory=True)
    clips = sorted(path.stem for path in raw.glob("*.csv"))
    if len(clips) != 40:
        raise ValueError(f"Expected all 40 G1 clips in {raw}, found {len(clips)}")
    meta = {"joint_names": JOINT_NAMES, "clip_names": clips}
    path = args.root / "meta.json"
    if path.exists() and json.loads(path.read_text()) != meta:
        raise ValueError(f"Existing {path} differs; refusing to overwrite metadata")
    path.write_text(json.dumps(meta, indent=2) + "\n")
    from supermimic.motions.lafan1 import build_cache
    build_cache(args.cache, raw_dir=raw)


if __name__ == "__main__":
    main()

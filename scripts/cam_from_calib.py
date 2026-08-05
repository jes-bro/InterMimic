#!/usr/bin/env python3
"""Print RECORD_VIDEO_CAM_POS / _CAM_TARGET reproducing a real camera's view.

Replaying a reconstruction from the same viewpoint the video was shot from makes
errors legible: a limb that tracked well and one that drifted look identical
from a three-quarter view, and obviously different from the angle the tracker
actually saw.

    python scripts/cam_from_calib.py trajectory/gopro_calibs.csv cam04

PRECONDITION -- the motion must already be in the calibration's world frame.
Isaac Gym interprets these coordinates in the sim's frame, which is whatever
frame the clip carries, and CARI4D reconstructs in the CAMERA's frame. Applied
to an untransformed clip this points the camera at empty space, because in that
frame the camera is at the origin rather than where the calibration puts it.

Use rotate_pt.py --from-calib to align first, and NOT with --around-root: that
mode deliberately keeps each frame at its original location, so it fixes
orientation while leaving translations in the camera frame.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot


def read_camera(csv_path: Path, cam_uid: str) -> dict:
    """Return one camera's world placement from an Ego-Exo4D gopro_calibs.csv.

    Args:
        csv_path: Path to gopro_calibs.csv.
        cam_uid: Camera identifier, e.g. "cam04".

    Returns:
        {"t_wc": (3,) camera centre in world, "R_wc": camera-to-world rotation}.

    Raises:
        SystemExit: if the file, the camera, or the expected columns are absent.
    """
    if not csv_path.is_file():
        raise SystemExit(f"no calibration at {csv_path}")
    with open(csv_path) as f:
        rows = {r["cam_uid"]: r for r in csv.DictReader(f)}
    if cam_uid not in rows:
        raise SystemExit(f"{cam_uid} not in {csv_path}; found {sorted(rows)}")
    row = rows[cam_uid]
    try:
        t_wc = np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"])
        quat = [float(row[f"q{a}_world_cam"]) for a in "xyzw"]
    except KeyError as exc:
        raise SystemExit(f"{csv_path} lacks *_world_cam columns ({exc}); this "
                         f"expects Ego-Exo4D's gopro_calibs.csv layout")
    return {"t_wc": t_wc, "R_wc": sRot.from_quat(quat)}


def main() -> int:
    """Print the two environment variables, and how well the view will match."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("calib", type=Path, help="gopro_calibs.csv")
    parser.add_argument("cam_uid", help="e.g. cam04")
    parser.add_argument("--distance", type=float, default=4.0,
                        help="metres along the view axis to place the look-at "
                             "target. Only sets where the camera points, not "
                             "what is in focus (default: 4).")
    args = parser.parse_args()

    cam = read_camera(args.calib, args.cam_uid)
    pos = cam["t_wc"]
    # A camera looks down +Z in its own frame (OpenCV convention), so the view
    # axis in world is R_wc applied to that.
    forward = cam["R_wc"].apply([0.0, 0.0, 1.0])
    target = pos + forward * args.distance

    # Isaac Gym's set_camera_location takes no up-vector; it assumes world +Z is
    # up. Any tilt of the real camera therefore shows as roll the sim cannot
    # reproduce, so report it rather than letting it be a silent mismatch.
    up = cam["R_wc"].apply([0.0, -1.0, 0.0])
    tilt = np.degrees(np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))

    print(f"# {args.cam_uid}: centre {np.round(pos, 3)}, "
          f"looking toward {np.round(forward, 3)}")
    print(f"# camera up maps to world {np.round(up, 3)} ({tilt:.1f} deg off +Z)")
    if tilt > 15.0:
        print(f"# WARNING: {tilt:.1f} deg is too much for Isaac Gym's fixed z-up "
              f"camera to reproduce; the render will be rolled by roughly that.")
    print()
    print(f"export RECORD_VIDEO_CAM_POS={pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}")
    print(f"export RECORD_VIDEO_CAM_TARGET="
          f"{target[0]:.4f},{target[1]:.4f},{target[2]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

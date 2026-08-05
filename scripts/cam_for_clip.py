#!/usr/bin/env python3
"""Print RECORD_VIDEO_CAM_POS / _CAM_TARGET that frame a clip's whole motion.

The replay camera is static, and its default (3,3,2.5) looking at (0,0,1) only
frames a subject who stays near the origin. A clip that has been rotated about
the world origin, or re-seated vertically, or simply covers ground -- a player
running in to shoot -- leaves that view entirely, and the render shows an empty
court.

    python scripts/cam_for_clip.py InterAct/behave_cari4d/sub100_bball_000.pt

Reads every joint of every frame, so the framing covers the motion rather than
just the root, and solves for the distance at which that fits the camera's field
of view. The vertical field is the binding one at 1280x720, which is why the
aspect ratio is part of the calculation rather than a fudge factor.

    eval "$(python scripts/cam_for_clip.py <clip>)"    # to apply directly
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


BODY_POS = slice(162, 318)   # 52 joints x 3, per intermimic.py:_load_motion
OBJ_POS = slice(318, 321)


def clip_bounds(path: Path, include_object: bool = True) -> tuple:
    """Return (centre, radius) enclosing every joint across every frame.

    Args:
        path: The 591-channel motion tensor.
        include_object: Whether the object's trajectory also has to be in frame.
            A thrown ball travels well past the player, so excluding it frames
            the person more tightly at the cost of losing the shot.

    Returns:
        (centre (3,), radius in metres).

    Raises:
        SystemExit: if the file is missing or not 591 channels wide.
    """
    if not path.is_file():
        raise SystemExit(f"no motion tensor at {path}")
    data = torch.load(str(path), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{path.name}: {data.shape[-1]} channels, want 591")

    pts = data[:, BODY_POS].reshape(data.shape[0], -1, 3).numpy().reshape(-1, 3)
    if include_object:
        pts = np.concatenate([pts, data[:, OBJ_POS].numpy()], axis=0)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)
    print(f"# {path.name}: {data.shape[0]} frames")
    print(f"#   x {lo[0]:+.2f}..{hi[0]:+.2f}  y {lo[1]:+.2f}..{hi[1]:+.2f}  "
          f"z {lo[2]:+.2f}..{hi[2]:+.2f}")
    print(f"#   centre {np.round(centre, 2)}  enclosing radius {radius:.2f} m")
    return centre, radius


def heading_degrees(path: Path) -> float:
    """Return the subject's mean direction of travel, as a world azimuth.

    Taken from root_pos rather than root_rot so it needs no knowledge of which
    local axis the rig calls forward -- a person running somewhere is going
    where their trajectory goes. Frame-to-frame steps are summed and the total
    displacement used, so a dribble's side-to-side jitter does not dominate.

    Raises:
        SystemExit: if the subject barely moves, since the heading is then
            meaningless and silently returning zero would aim the camera
            somewhere arbitrary.
    """
    data = torch.load(str(path), map_location="cpu")
    root = data[:, 0:3].numpy()
    delta = root[-1, :2] - root[0, :2]
    if np.linalg.norm(delta) < 0.3:
        raise SystemExit(
            f"{path.name}: net travel is only {np.linalg.norm(delta):.2f} m, so "
            f"there is no meaningful heading. Use --azimuth instead.")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def camera_for(centre: np.ndarray, radius: float, azimuth: float,
               elevation: float, fov_deg: float, width: int, height: int,
               margin: float) -> tuple:
    """Return (position, target) viewing a sphere from a given direction.

    The distance solves for the tighter of the two fields of view. At 1280x720
    the vertical field is roughly 59 degrees against the horizontal 90, so
    sizing on the horizontal alone would crop the top and bottom of the motion.
    """
    half_h = np.radians(fov_deg) / 2.0
    half_v = np.arctan(np.tan(half_h) * height / width)
    distance = radius * margin / np.tan(min(half_h, half_v))

    az, el = np.radians(azimuth), np.radians(elevation)
    offset = np.array([np.cos(el) * np.cos(az),
                       np.cos(el) * np.sin(az),
                       np.sin(el)]) * distance
    return centre + offset, centre


def main() -> int:
    """Print the two export lines, and the geometry they were derived from."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pt_path", type=Path)
    parser.add_argument("--from-heading", type=float, default=None,
                        metavar="DEG",
                        help="Place the camera at this angle from the subject's "
                             "direction of travel instead of a world azimuth. "
                             "0 puts it AHEAD of him, so he moves toward it and "
                             "is seen head-on; 180 follows from behind; 90 is "
                             "side-on. (The camera is offset ALONG the heading, "
                             "so 0 is in front, not behind.) Use this to "
                             "approximate the angle the source footage had: the "
                             "exact viewpoint would need the clip in the "
                             "calibration's world frame, and rotation plus the "
                             "floor shift have moved it out of that frame, but "
                             "the view RELATIVE to the runner is preserved.")
    parser.add_argument("--azimuth", type=float, default=45.0,
                        help="viewing direction around vertical, degrees "
                             "(default: 45, a three-quarter view)")
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="height of the camera above the subject, degrees "
                             "(default: 20). Near 0 is a courtside view; large "
                             "values look down and flatten the motion.")
    parser.add_argument("--margin", type=float, default=1.15,
                        help="fraction of extra room around the motion "
                             "(default: 1.15)")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="horizontal field of view in degrees; matches "
                             "Isaac Gym's CameraProperties default (90)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-object", action="store_true",
                        help="frame only the person. A thrown ball travels far "
                             "past the player and pulls the camera back with it.")
    args = parser.parse_args()

    src = args.pt_path.expanduser().resolve()
    centre, radius = clip_bounds(src, include_object=not args.no_object)

    azimuth = args.azimuth
    if args.from_heading is not None:
        heading = heading_degrees(src)
        azimuth = heading + args.from_heading
        print(f"#   heading {heading:.1f} deg, camera {args.from_heading:.0f} deg "
              f"off it -> azimuth {azimuth:.1f} deg")
    if radius < 1e-6:
        raise SystemExit("the clip has zero extent; nothing to frame")

    pos, target = camera_for(centre, radius, azimuth, args.elevation,
                             args.fov, args.width, args.height, args.margin)
    print(f"#   camera {np.round(pos, 2)} -> {np.round(target, 2)}  "
          f"(distance {np.linalg.norm(pos - target):.2f} m)")
    print()
    print(f"export RECORD_VIDEO_CAM_POS={pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}")
    print(f"export RECORD_VIDEO_CAM_TARGET="
          f"{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

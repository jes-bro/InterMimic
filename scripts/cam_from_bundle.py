#!/usr/bin/env python3
"""Place the replay camera exactly where the filming camera was.

CARI4D reconstructs in the camera's own frame: the camera sits at the origin
looking down +Z. That frame does not survive into the simulator -- retargeting,
the upright flip and the floor shift each move the clip -- so the camera's
coordinates cannot simply be copied across.

They do not need to be. The bundle and the motion tensor describe the same
person walking the same path, one in camera coordinates and one in simulator
coordinates, so the rigid transform between the two frames can be recovered from
the trajectories themselves and then applied to the camera. No knowledge of what
the intervening stages did to the frame is required, which also means this keeps
working if those stages change.

    python scripts/cam_from_bundle.py --bundle <cari4d .pth> \\
        --pt InterAct/behave_cari4d/sub100_bball_000.pt

Prints the residual of the fit. A good alignment lands within a few centimetres;
a large residual means the two trajectories are not the same motion, and the
camera it prints would be meaningless.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_bundle_translation(bundle_path: Path, key: str = "pr") -> np.ndarray:
    """Return the (T, 3) SMPL root translation from a CARI4D bundle.

    Reuses cari4d_to_interact.py's permissive unpickler, since the bundle
    references classes that are not importable here.

    Raises:
        SystemExit: if the bundle, the sub-dict, or smpl_t is missing.
    """
    helper = Path(__file__).with_name("cari4d_to_interact.py")
    spec = importlib.util.spec_from_file_location("_c2i", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not bundle_path.is_file():
        raise SystemExit(f"no bundle at {bundle_path}")
    bundle = mod._load_bundle(bundle_path)
    if key not in bundle:
        raise SystemExit(f"bundle has no '{key}'; got {list(bundle)}")
    if "smpl_t" not in bundle[key]:
        raise SystemExit(f"bundle['{key}'] has no smpl_t; got {list(bundle[key])}")
    return bundle[key]["smpl_t"].detach().cpu().numpy().astype(np.float64)


def load_pt_root(pt_path: Path) -> np.ndarray:
    """Return the (T, 3) root translation from a 591-channel motion tensor.

    Raises:
        SystemExit: if the file is missing or not 591 channels wide.
    """
    if not pt_path.is_file():
        raise SystemExit(f"no motion tensor at {pt_path}")
    data = torch.load(str(pt_path), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{pt_path.name}: {data.shape[-1]} channels, want 591")
    return data[:, 0:3].numpy().astype(np.float64)


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Return (R, t, rms) mapping src onto dst, rotation and translation only.

    Kabsch. Scale is deliberately not fitted: both frames are metric, so a scale
    far from 1 would mean something is wrong upstream, and absorbing it here
    would hide that instead of surfacing it in the residual.

    Returns:
        (3x3 rotation, (3,) translation, RMS residual in metres).
    """
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - R @ src_c
    rms = float(np.sqrt((((src @ R.T + t) - dst) ** 2).sum(axis=1).mean()))
    return R, t, rms


def main() -> int:
    """Print the camera pose, in simulator coordinates, of the filming camera."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="the CARI4D .pth the clip was converted from")
    parser.add_argument("--pt", type=Path, required=True,
                        help="the installed motion tensor, after any rotation")
    parser.add_argument("--bundle-key", default="pr", choices=["pr", "gt", "in"])
    parser.add_argument("--pull-back", type=float, default=0.0,
                        help="metres to retreat along the view axis. The real "
                             "camera framed a whole court; the reconstruction is "
                             "one player, so the true position can sit far away "
                             "with the subject small (default: 0, exact).")
    parser.add_argument("--center-subject", action="store_true",
                        help="Aim at the subject rather than straight down the "
                             "real camera's optical axis. The camera stays where "
                             "it was, so the viewpoint is unchanged -- only what "
                             "is centred in frame differs. The real camera framed "
                             "a whole court, so its axis points wherever the "
                             "operator wanted, and the player can sit well off "
                             "to one side.")
    parser.add_argument("--target-distance", type=float, default=None,
                        help="override the look-at distance in metres. Default "
                             "is the subject's mean distance from the camera, "
                             "which keeps him centred.")
    args = parser.parse_args()

    src = load_bundle_translation(args.bundle.expanduser().resolve(),
                                  args.bundle_key)
    dst = load_pt_root(args.pt.expanduser().resolve())
    if len(src) != len(dst):
        raise SystemExit(f"frame counts differ: bundle {len(src)}, tensor "
                         f"{len(dst)}. These are not the same clip.")

    R, t, rms = rigid_fit(src, dst)
    print(f"# aligned {len(src)} frames, RMS residual {rms * 100:.1f} cm")
    if rms > 0.15:
        print(f"# WARNING: {rms * 100:.0f} cm is too large for the same motion in "
              f"two frames. The camera below is not trustworthy.")

    # In the bundle's frame the camera is the origin, looking down +Z.
    cam_pos = t                      # R @ [0,0,0] + t
    forward = R @ np.array([0.0, 0.0, 1.0])
    distance = (args.target_distance if args.target_distance is not None
                else float(np.linalg.norm(dst - cam_pos, axis=1).mean()))

    if args.center_subject:
        # Centroid of every joint over every frame, not the root: the root is at
        # the pelvis, so aiming there puts the head near the top of frame.
        pt = torch.load(str(args.pt.expanduser().resolve()), map_location="cpu")
        target = pt[:, 162:318].reshape(pt.shape[0], -1, 3).numpy().reshape(-1, 3).mean(axis=0)
        off_axis = np.degrees(np.arccos(np.clip(
            (target - cam_pos) / np.linalg.norm(target - cam_pos) @ forward,
            -1.0, 1.0)))
        print(f"# centring on the subject: {off_axis:.1f} deg off the real "
              f"camera's axis")
    else:
        target = cam_pos + forward * distance

    if args.pull_back:
        cam_pos = cam_pos - forward * args.pull_back
        print(f"# pulled back {args.pull_back:.1f} m along the view axis")

    up = R @ np.array([0.0, -1.0, 0.0])
    tilt = np.degrees(np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))
    print(f"# camera at {np.round(cam_pos, 2)}, looking {np.round(forward, 3)}, "
          f"subject {distance:.2f} m away")
    print(f"# camera up maps to {np.round(up, 3)} ({tilt:.1f} deg off world +Z); "
          f"Isaac Gym assumes z-up and will roll by that much")
    print()
    print(f"export RECORD_VIDEO_CAM_POS="
          f"{cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f}")
    print(f"export RECORD_VIDEO_CAM_TARGET="
          f"{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

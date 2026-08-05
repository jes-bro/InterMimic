#!/usr/bin/env python3
"""Report which way a 591-channel motion tensor is actually oriented.

Fixing an upside-down replay by trying rotations and re-rendering is slow and
tells you little when it fails -- a wrong guess and a rotation that never ran
look identical. This reads the answer straight out of the joint positions, so
the question becomes arithmetic instead of guesswork.

    python scripts/check_pt_orientation.py InterAct/behave_cari4d/sub100_bball_000.pt

Pass a second path (or rely on the .bak written by rotate_pt.py) to compare
before and after, which separates "the rotation was wrong" from "the rotation
never touched this file".

The measurement is the pelvis-to-head vector, averaged over frames. It needs no
assumption about frame conventions, upright_start, or which camera shot the
clip: whatever the tensor holds, a standing person's head is above their pelvis,
so in a correct z-up clip that vector is near +Z.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# Joint order is SMPLH_BONE_ORDER_NAMES, as printed by interact2mimic.py.
PELVIS, L_ANKLE, R_ANKLE, HEAD = 0, 7, 8, 15
BODY_POS = slice(162, 318)   # 52 joints x 3, per intermimic.py:_load_motion
ROOT_POS = slice(0, 3)
AXES = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
        "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}


def load_body_pos(path: Path) -> np.ndarray:
    """Return the (T, 52, 3) joint positions from a 591-channel tensor.

    Raises:
        SystemExit: if the file is missing or is not 591 channels wide, since
            every slice below would otherwise read the wrong data silently.
    """
    if not path.is_file():
        raise SystemExit(f"no motion tensor at {path}")
    data = torch.load(str(path), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{path.name}: {data.shape[-1]} channels, want 591")
    return data[:, BODY_POS].reshape(data.shape[0], -1, 3).numpy(), data


def nearest_axis(v: np.ndarray) -> tuple:
    """Return the (name, degrees) of the world axis a unit vector is closest to."""
    best, best_dot = None, -2.0
    for name, axis in AXES.items():
        d = float(v @ np.array(axis, dtype=float))
        if d > best_dot:
            best, best_dot = name, d
    return best, float(np.degrees(np.arccos(np.clip(best_dot, -1.0, 1.0))))


def describe_root_rot(data) -> None:
    """Print where root_rot sends each local axis, and whether it agrees with body_pos.

    This is the channel that matters for a replay: play_dataset_step
    (intermimic.py:2064) drives the humanoid from root_pos, root_rot and
    dof_pos, and never reads body_pos. A conclusion drawn from body_pos alone
    can therefore describe data the render does not use.

    The consistency check is the spread of the body's up direction expressed in
    the root's own frame. If positions and rotations describe the same skeleton
    that direction is a fixed local axis and the spread is small; a large spread
    means the two halves of the tensor are in different frames, and only the
    rotations govern what is drawn.
    """
    from scipy.spatial.transform import Rotation as sRot

    T = data.shape[0]
    root = sRot.from_quat(data[:, 3:7].numpy())
    body = data[:, BODY_POS].clone().reshape(T, -1, 3).numpy()
    up_world = body[:, HEAD] - body[:, PELVIS]
    up_world /= (np.linalg.norm(up_world, axis=1, keepdims=True) + 1e-12)

    print("  root_rot sends local")
    for name, axis in (("+X", [1, 0, 0]), ("+Y", [0, 1, 0]), ("+Z", [0, 0, 1])):
        w = root.apply(np.tile(axis, (T, 1))).mean(axis=0)
        w = w / (np.linalg.norm(w) + 1e-12)
        near, deg = nearest_axis(w)
        print(f"    {name} -> world {np.round(w, 3)}  (nearest {near}, {deg:.1f} deg)")

    local_up = root.inv().apply(up_world)
    mean_local = local_up.mean(axis=0)
    mean_local /= (np.linalg.norm(mean_local) + 1e-12)
    spread = np.degrees(np.arccos(np.clip(local_up @ mean_local, -1.0, 1.0)))
    near, deg = nearest_axis(mean_local)
    print(f"  body up in root frame {np.round(mean_local, 3)} "
          f"(nearest {near}, {deg:.1f} deg), spread {spread.mean():.1f} deg "
          f"mean / {spread.max():.1f} max")
    if spread.mean() > 20.0:
        print("  -> positions and rotations DISAGREE. Trust root_rot: it is what "
              "the replay renders.")
    else:
        print("  -> positions and rotations agree.")


def describe(path: Path) -> np.ndarray:
    """Print the up-direction and ground-clearance of one tensor.

    Returns:
        The mean pelvis-to-head unit vector, for the caller to compare.
    """
    body, data = load_body_pos(path)
    up = body[:, HEAD] - body[:, PELVIS]
    up = up.mean(axis=0)
    up = up / (np.linalg.norm(up) + 1e-12)
    axis, deg = nearest_axis(up)

    feet_z = body[:, [L_ANKLE, R_ANKLE], 2]
    head_z = body[:, HEAD, 2]
    root = data[:, ROOT_POS].numpy()

    print(f"{path.name}: {body.shape[0]} frames")
    print(f"  pelvis->head  {np.round(up, 3)}  -- nearest {axis} ({deg:.1f} deg off)")
    print(f"  head z        {head_z.min():.2f} .. {head_z.max():.2f} m")
    print(f"  ankle z       {feet_z.min():.2f} .. {feet_z.max():.2f} m")
    print(f"  root pos z    {root[:, 2].min():.2f} .. {root[:, 2].max():.2f} m")
    if axis == "+Z" and deg < 30:
        print(f"  -> upright.")
    elif axis == "-Z":
        print(f"  -> UPSIDE DOWN (head below pelvis).")
    else:
        print(f"  -> lying down / rolled: up points along {axis}, not a vertical axis.")
    if feet_z.mean() > head_z.mean():
        print(f"  -> feet sit above the head on average, consistent with a flip.")
    describe_root_rot(data)
    return up


def main() -> int:
    """Describe one tensor, or two, reporting the angle between them."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pt_path", type=Path)
    parser.add_argument("compare", type=Path, nargs="?", default=None,
                        help="second tensor to compare against. Defaults to "
                             "<pt_path>.bak when that exists, which is what "
                             "rotate_pt.py leaves behind.")
    args = parser.parse_args()

    src = args.pt_path.expanduser().resolve()
    up_a = describe(src)

    other = args.compare
    if other is None:
        other = None
        for cand in (src.parent / ".rotate_pt_backups" / (src.name + ".bak"),
                     src.with_suffix(src.suffix + ".bak")):
            if cand.is_file():
                other = cand
                break
    if other is None:
        return 0

    print()
    up_b = describe(Path(other).expanduser().resolve())
    angle = float(np.degrees(np.arccos(np.clip(up_a @ up_b, -1.0, 1.0))))
    print()
    print(f"angle between the two up-vectors: {angle:.1f} deg")
    if angle < 1.0:
        # The distinction that matters: a rotation that ran and was wrong needs
        # a different rotation, one that never ran needs a different command.
        print("  -> effectively identical. The rotation did not reach this file "
              "(wrong path, or overwritten afterwards by a later install).")
    else:
        print("  -> the file did change, so the rotation ran; if it is still "
              "wrong, the rotation applied was not the right one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

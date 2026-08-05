#!/usr/bin/env python3
"""Report a motion tensor's up-direction frame by frame, not averaged.

check_pt_orientation.py averages the rotated up-axis over frames and normalizes
the result. That hides the case where a clip is upright for part of its length
and inverted for the rest: the mean of two opposing directions is near zero, and
normalizing turns whatever noise remains into a confident-looking unit vector.
A clip that flips halfway therefore reports as cleanly upright.

This prints the magnitude before normalizing -- which is the tell, since it only
approaches 1 when the frames agree -- along with how many frames point downward
and where any transition happens.

    python scripts/check_pt_perframe.py InterAct/behave_cari4d/sub100_bball_000.pt

Reads root_rot (channels 3:7), because that is what play_dataset_step
(intermimic.py:2064) hands the humanoid. The rig's up is local +Z: the generated
MJCF puts Pelvis at the origin with a plain freejoint and L_Hip below it at
z=-0.079, so no frame offset stands between the two.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


def frame_up_vectors(path: Path) -> np.ndarray:
    """Return the (T, 3) world direction of the rig's local +Z, per frame.

    Raises:
        SystemExit: if the file is missing or not 591 channels wide, rather than
            silently slicing whatever happens to sit at columns 3:7.
    """
    if not path.is_file():
        raise SystemExit(f"no motion tensor at {path}")
    data = torch.load(str(path), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{path.name}: {data.shape[-1]} channels, want 591")
    quat = data[:, 3:7].numpy().astype(np.float64)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise SystemExit(f"{path.name}: root_rot has zero-length quaternions; "
                         f"the tensor is malformed, not merely mis-rotated")
    return sRot.from_quat(quat / norms).apply(np.tile([0.0, 0.0, 1.0], (len(quat), 1)))


def runs(mask: np.ndarray) -> list:
    """Return [(start, end, value)] for consecutive equal stretches of a bool array.

    Turns a per-frame flag into the handful of intervals worth reading, so a
    clip that inverts partway shows its transition instead of 101 lines.
    """
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((start, i - 1, bool(mask[start])))
            start = i
    return out


def report(path: Path) -> None:
    """Print the per-frame verdict for one tensor."""
    up = frame_up_vectors(path)
    mean = up.mean(axis=0)
    mag = float(np.linalg.norm(mean))
    down = up[:, 2] < 0

    print(f"{path.name}: {len(up)} frames")
    print(f"  mean up {np.round(mean, 3)}  magnitude {mag:.3f}")
    print(f"  up-z    min {up[:, 2].min():+.3f}  max {up[:, 2].max():+.3f}")
    print(f"  frames pointing down: {int(down.sum())} of {len(up)}")

    segments = runs(down)
    if len(segments) == 1:
        verdict = "INVERTED throughout" if segments[0][2] else "upright throughout"
        print(f"  -> {verdict}. A single global rotation is the right fix.")
    else:
        print(f"  -> MIXED: {len(segments)} stretches.")
        for lo, hi, is_down in segments:
            print(f"       frames {lo:>4}-{hi:<4} {'DOWN' if is_down else 'up'}"
                  f"  ({hi - lo + 1} frames)")
        print("  -> a single global rotation cannot fix this; the flip is in the "
              "source motion, not in the frame it was exported to.")

    if mag < 0.9 and len(segments) == 1:
        print(f"  NOTE: magnitude {mag:.3f} is low without a sign change, so the "
              f"clip tumbles rather than flipping -- averaged reports of this "
              f"tensor are not trustworthy.")


def main() -> int:
    """Report each tensor given, defaulting to the .bak alongside the first."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pt_paths", type=Path, nargs="+")
    parser.add_argument("--no-bak", action="store_true",
                        help="skip the .bak comparison that rotate_pt.py leaves")
    args = parser.parse_args()

    paths = [p.expanduser().resolve() for p in args.pt_paths]
    if not args.no_bak and len(paths) == 1:
        bak = paths[0].with_suffix(paths[0].suffix + ".bak")
        if bak.is_file():
            paths.append(bak)

    for i, path in enumerate(paths):
        if i:
            print()
        report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

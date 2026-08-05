#!/usr/bin/env python3
"""Measure how close a dribbled ball gets to the floor, bounce by bounce.

"The ball does not quite reach the floor" has several possible causes with very
different fixes, and the per-bounce numbers separate them:

  Every bounce short by about the same amount
      A bias -- the object sits too high throughout. Either the floor was fitted
      from the person and is slightly off, or the tracked depth carries a
      constant offset. Fix by shifting, not by re-tracking.

  Bounces short by varying amounts, deeper ones nearer the floor
      Temporal sampling. A dribble's contact lasts a few milliseconds and 30 Hz
      frames mostly miss it, so the lowest SAMPLED point sits above the lowest
      real point. Higher resolution does not help; more frames or a ballistic
      fit between bounces does.

  Bounces scattered with no pattern, some below the floor
      Tracking noise. This is where better triangulation or the ego view helps.

    python scripts/check_ball_contact.py --pt <clip .pt> --mesh <ball .obj>

The mesh gives the radius, which is what the centre height is measured against.
Without it the script still reports heights, but cannot say what contact means.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


OBJ_POS = slice(318, 321)
BODY_POS = slice(162, 318)
FEET = [7, 8, 10, 11]          # L/R Ankle, L/R Toe


def ball_radius(mesh_path: Path) -> float:
    """Return half the mesh's mean extent, in metres.

    Mean of the three extents rather than the largest: a reconstruction is never
    a perfect sphere, and the largest axis would overstate the radius and make
    every bounce look short by the difference.

    Raises:
        SystemExit: if the mesh cannot be read.
    """
    try:
        import trimesh
    except ImportError:
        raise SystemExit("trimesh is required; it ships in the intermimic env")
    if not mesh_path.is_file():
        raise SystemExit(f"no mesh at {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    extents = np.asarray(mesh.extents, dtype=float)
    spread = extents.max() / extents.min() if extents.min() > 0 else np.inf
    if spread > 1.15:
        print(f"note: extents {np.round(extents, 3)} differ by {spread:.2f}x, so "
              f"this is not a good sphere and 'radius' is an average")
    return float(extents.mean() / 2.0)


def local_minima(z: np.ndarray, min_gap: int = 4) -> list:
    """Return indices of local minima at least min_gap frames apart.

    A dribble's bounces are the minima of the ball's height. The spacing rule
    keeps frame-to-frame jitter near a trough from being counted as several
    separate bounces.
    """
    idx = [i for i in range(1, len(z) - 1) if z[i] <= z[i - 1] and z[i] < z[i + 1]]
    kept = []
    for i in idx:
        if not kept or i - kept[-1] >= min_gap:
            kept.append(i)
        elif z[i] < z[kept[-1]]:
            kept[-1] = i
    return kept


def main() -> int:
    """Report each bounce's clearance and what pattern the clearances make."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None,
                        help="the ball .obj, for its radius")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="source frame rate, used to work out how much of "
                             "the clearance sampling alone explains (default: 30)")
    args = parser.parse_args()

    data = torch.load(str(args.pt.expanduser().resolve()), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{args.pt.name}: {data.shape[-1]} channels, want 591")
    obj_z = data[:, OBJ_POS].numpy()[:, 2]
    body = data[:, BODY_POS].reshape(data.shape[0], -1, 3).numpy()
    foot_z = body[:, FEET, 2].min(axis=1)

    print(f"{args.pt.name}: {len(obj_z)} frames")
    print(f"  ball centre z   {obj_z.min():+.3f} .. {obj_z.max():+.3f} m")
    print(f"  lowest foot z   {foot_z.min():+.3f} .. {foot_z.max():+.3f} m")

    radius = None
    if args.mesh is not None:
        radius = ball_radius(args.mesh.expanduser().resolve())
        print(f"  ball radius     {radius:.3f} m "
              f"(diameter {radius * 2:.3f} m)")
        print(f"  centre height at true contact would be {radius:.3f} m")

    mins = local_minima(obj_z)
    if not mins:
        print("  no local minima -- the ball never bounces in this clip")
        return 0

    print()
    print(f"  {'frame':>6} {'centre_z':>9} {'clearance':>10}")
    print("  " + "-" * 28)
    clearances = []
    for i in mins:
        if radius is None:
            print(f"  {i:>6} {obj_z[i]:>+9.3f} {'n/a':>10}")
            continue
        gap = obj_z[i] - radius
        clearances.append(gap)
        print(f"  {i:>6} {obj_z[i]:>+9.3f} {gap:>+10.3f}")

    if not clearances:
        return 0

    c = np.array(clearances)
    print()
    print(f"  {len(c)} bounces, clearance mean {c.mean():+.3f} m, "
          f"sd {c.std():.3f} m, range {c.min():+.3f}..{c.max():+.3f}")

    # How much of the clearance sampling alone explains. Between bounces the
    # ball is in free flight, so the interval fixes the impact speed: a ball
    # that takes T seconds between bounces leaves the floor at g*T/2 and returns
    # at the same speed. The nearest frame can miss contact by up to half a
    # frame, so the lowest SAMPLED height sits up to v/(2*fps) above the floor.
    # This is the number that matters -- sd alone cannot tell aliasing from bias,
    # because a dribble whose period divides the frame interval aliases by the
    # same amount every bounce and looks perfectly consistent.
    expected = None
    if len(mins) > 1:
        period = float(np.mean(np.diff(mins))) / args.fps
        impact_v = 9.81 * period / 2.0
        expected = impact_v / (2.0 * args.fps)
        print(f"  bounce period {period:.2f} s -> impact speed {impact_v:.1f} m/s")
        print(f"  sampling at {args.fps:.0f} Hz can leave the lowest frame up to "
              f"{expected:.3f} m high on its own")

    if abs(c.mean()) < 0.02:
        print("  -> the ball reaches the floor. Nothing to fix.")
    elif expected is not None and c.mean() <= expected * 1.5:
        print(f"  -> SAMPLING accounts for this. The ball does reach the floor "
              f"between frames. Fix by fitting the ballistic arc between "
              f"bounces, not by re-tracking or by higher resolution -- neither "
              f"adds frames.")
    elif expected is not None and c.mean() > expected * 1.5:
        print(f"  -> {c.mean():+.3f} m exceeds the {expected:.3f} m sampling can "
              f"explain, so about {c.mean() - expected:+.3f} m is real bias or "
              f"tracking error. sd {c.std():.3f} m says which: small means a "
              f"constant offset to correct, large means noisy tracking where "
              f"better triangulation helps.")
    else:
        print("  -> only one bounce, so sampling error cannot be estimated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

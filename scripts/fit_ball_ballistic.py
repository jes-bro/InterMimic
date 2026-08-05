#!/usr/bin/env python3
"""Fit the ball's free-flight arcs to find where it really reached, between frames.

The lowest tracked frame is not the lowest point of a bounce -- contact lasts
milliseconds and 30 Hz frames mostly miss it. But a ball between bounces is in
free fall, so a parabola through the tracked points describes the continuous
trajectory, and ITS minimum is the true one no matter where the samples landed.

That makes this decisive about a ball that looks too high:

  fitted minimum near the radius
      The ball does reach the floor; only the sampling missed it. Nothing is
      mistracked and no shift is warranted.

  fitted minimum still high
      The ball genuinely never got there, so the tracking is wrong and denser
      sampling would not have helped.

The fitted acceleration is the check on the fit itself. Free flight gives
-9.81 m/s^2; anything far from that means the segment is not free flight -- the
ball is held, or the tracking is drifting -- and its minimum means nothing.

    python scripts/fit_ball_ballistic.py --pt <clip> --mesh <ball .obj>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


OBJ_POS = slice(318, 321)
G = 9.81


def sampled_troughs(z, min_gap=4):
    """Return indices of local minima at least min_gap frames apart."""
    idx = [i for i in range(1, len(z) - 1) if z[i] <= z[i - 1] and z[i] < z[i + 1]]
    kept = []
    for i in idx:
        if not kept or i - kept[-1] >= min_gap:
            kept.append(i)
        elif z[i] < z[kept[-1]]:
            kept[-1] = i
    return kept


def fit_freefall(t, z):
    """Fit z = z0 + v0*t - g*t^2/2 with gravity FIXED, returning (z0, v0, rms).

    Two free parameters rather than three. With only a handful of samples per
    arc a free quadratic will happily fit an acceleration nothing like gravity
    and hide a tracking failure behind a small residual; holding g at its real
    value turns that failure into a large residual instead.
    """
    y = z + 0.5 * G * t ** 2
    A = np.stack([np.ones_like(t), t], axis=1)
    (z0, v0), *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(((A @ np.array([z0, v0]) - y) ** 2).mean()))
    return float(z0), float(v0), rms


def bounce_from_arcs(t_fall, z_fall, t_rise, z_rise):
    """Return (time, height, rms) where the falling and rising arcs meet.

    The ball's contact is a velocity discontinuity, so one parabola cannot span
    it -- fitting across the bounce yields an acceleration nowhere near gravity.
    Fitting the two arcs separately and intersecting them puts the bounce where
    the ball's own motion says it was, which is between frames and generally
    below every sample.

    Returns (nan, nan, rms) when the arcs do not cross, which means they are not
    describing one bounce.
    """
    z0f, v0f, rms_f = fit_freefall(t_fall, z_fall)
    z0r, v0r, rms_r = fit_freefall(t_rise, z_rise)
    # Both carry the same -g t^2/2 term, so it cancels and the crossing is linear.
    if abs(v0f - v0r) < 1e-9:
        return float("nan"), float("nan"), max(rms_f, rms_r)
    t_c = (z0r - z0f) / (v0f - v0r)
    z_c = z0f + v0f * t_c - 0.5 * G * t_c ** 2
    return float(t_c), float(z_c), max(rms_f, rms_r)


def main() -> int:
    """Report where each bounce really happened, against the lowest sample."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None,
                        help="ball .obj, for the radius contact implies")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-contact-height", type=float, default=0.6,
                        help="ignore troughs above this (default: 0.6), so dips "
                             "mid-flight during a shot are not read as bounces")
    parser.add_argument("--arc-frames", type=int, default=4,
                        help="frames either side of a trough to fit (default: 4)")
    args = parser.parse_args()

    data = torch.load(str(args.pt.expanduser().resolve()), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{args.pt.name}: {data.shape[-1]} channels, want 591")
    z = data[:, OBJ_POS].numpy()[:, 2]
    t = np.arange(len(z)) / args.fps

    radius = None
    if args.mesh is not None:
        try:
            import trimesh
        except ImportError:
            raise SystemExit("trimesh is required; it ships in the intermimic env")
        mesh = trimesh.load(str(args.mesh.expanduser().resolve()), force="mesh",
                            process=False)
        radius = float(np.asarray(mesh.extents, dtype=float).mean() / 2.0)
        print(f"ball radius {radius:.3f} m -- contact puts the centre there")

    troughs = [i for i in sampled_troughs(z) if z[i] <= args.max_contact_height]
    print(f"{args.pt.name}: {len(z)} frames, {len(troughs)} troughs near the floor")
    if not troughs:
        print("  nothing below the contact height to analyse")
        return 0

    print()
    print(f"  {'trough':>7} {'sampled':>9} {'fitted':>8} {'below':>8} "
          f"{'fit_rms':>8}")
    print("  " + "-" * 46)

    fitted = []
    n = args.arc_frames
    for i in troughs:
        lo, hi = max(0, i - n), min(len(z) - 1, i + n)
        # The trough frame itself is excluded from both arcs: it is the sample
        # nearest an impact the camera did not resolve, so it belongs to neither
        # free-flight phase and drags both fits toward itself.
        t_fall, z_fall = t[lo:i], z[lo:i]
        t_rise, z_rise = t[i + 1:hi + 1], z[i + 1:hi + 1]
        if len(t_fall) < 2 or len(t_rise) < 2:
            print(f"  {i:>7} {z[i]:>9.3f} {'--':>8} {'--':>8} {'--':>8}"
                  f"   (too few frames either side)")
            continue
        t_c, z_c, rms = bounce_from_arcs(t_fall, z_fall, t_rise, z_rise)
        if not np.isfinite(z_c):
            print(f"  {i:>7} {z[i]:>9.3f} {'--':>8} {'--':>8} {rms:>8.3f}"
                  f"   (arcs do not cross)")
            continue
        print(f"  {i:>7} {z[i]:>9.3f} {z_c:>8.3f} {z[i] - z_c:>+8.3f} {rms:>8.3f}")
        fitted.append(z_c)

    if not fitted:
        return 0

    f = np.array(fitted)
    print()
    print(f"  fitted contact height: mean {f.mean():.3f} m over {len(f)} bounces")
    if radius is None:
        return 0
    if abs(f.mean() - radius) < 0.06:
        print(f"  -> matches the radius {radius:.3f}. The ball DOES reach the "
              f"floor; the frames simply miss the instant. Nothing is "
              f"mistracked, and the render looks high because the lowest "
              f"SAMPLE is high, not the lowest point.")
    else:
        print(f"  -> {f.mean():.3f} against a radius of {radius:.3f}, so the "
              f"ball's own motion says it turned around {f.mean() - radius:+.3f} m "
              f"above contact. That is not a sampling artifact: the arcs either "
              f"side already imply where it bottomed out, and it is not the "
              f"floor. The tracking is high.")
    print(f"  fit_rms well above a centimetre means the arcs are not free flight, "
          f"and neither conclusion holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

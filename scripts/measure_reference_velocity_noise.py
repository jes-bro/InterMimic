#!/usr/bin/env python3
"""How noisy is a reference clip's VELOCITY, relative to its motion?

Decides whether the reward's velocity terms (rewardWeights pv/rv/orv, zero in
every config in this repo) are safe to switch on for a given dataset.

The clips store positions, not velocities, so any velocity the reward grades is
a finite difference -- and differentiating amplifies noise. Mocap (OMOMO) is
smooth enough that this is free. A monocular 4D reconstruction is per-frame
optimised, so its positions can carry jitter that is invisible in the position
reward and dominant in the derivative. Turning on pv against a jittery reference
teaches the policy to chase reconstruction artefacts.

METHOD. Smooth the positions with a short moving average, difference both the
raw and smoothed tracks, and report how much of the velocity signal the smoother
removed. A high fraction means the velocity is mostly high-frequency content
that no controller should be asked to reproduce. Reported per clip, plus the
peak speeds so a genuinely fast motion is not mistaken for a noisy one.

    python3 scripts/measure_reference_velocity_noise.py InterAct/OMOMO_new/sub2_largetable_000.pt
    python3 scripts/measure_reference_velocity_noise.py InterAct/behave_cari4d_optj3d_cf2/*.pt
    python3 scripts/measure_reference_velocity_noise.py --glob 'InterAct/OMOMO_new/sub2_*.pt' --limit 12
"""
import argparse
import glob as globmod
import sys
from pathlib import Path

import numpy as np
import torch

I_BODY = slice(162, 318)     # 52 joints x 3, per intermimic.py:_load_motion
I_OBJP = slice(318, 321)
N_CHANNELS = 591


def moving_average(x, k):
    """Centred moving average along axis 0, edges held (no phase shift)."""
    if k <= 1:
        return x.copy()
    pad = k // 2
    padded = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    kern = np.ones(k) / k
    out = np.empty_like(x)
    flat = padded.reshape(padded.shape[0], -1)
    res = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="valid"), 0, flat)
    out[:] = res.reshape(x.shape)
    return out


def analyse(path, fps, smooth_k):
    t = torch.load(str(path), map_location="cpu", weights_only=False)
    if t.ndim != 2 or t.shape[-1] != N_CHANNELS:
        raise SystemExit(f"{path}: expected (T, {N_CHANNELS}), got {tuple(t.shape)}")
    t = t.detach()          # OMOMO clips can carry requires_grad
    body = t[:, I_BODY].numpy().astype(np.float64).reshape(len(t), -1, 3)
    obj = t[:, I_OBJP].numpy().astype(np.float64)

    def stats(p):
        v_raw = np.diff(p, axis=0) * fps
        v_smooth = np.diff(moving_average(p, smooth_k), axis=0) * fps
        resid = v_raw - v_smooth
        # Fraction of the velocity signal that the smoother removes. Scale-free,
        # so a fast clip and a slow one are directly comparable.
        denom = np.abs(v_smooth).mean() + 1e-12
        noise = np.abs(resid).mean() / denom
        speed = np.linalg.norm(v_raw, axis=-1)
        return noise, float(np.median(speed)), float(speed.max())

    b_noise, b_med, b_max = stats(body)
    o_noise, o_med, o_max = stats(obj[:, None, :])
    return {
        "frames": len(t),
        "body_noise": b_noise, "body_med_speed": b_med, "body_max_speed": b_max,
        "obj_noise": o_noise, "obj_med_speed": o_med, "obj_max_speed": o_max,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*", type=Path)
    p.add_argument("--glob", help="shell glob, quoted, as an alternative to listing clips")
    p.add_argument("--limit", type=int, help="only the first N clips")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--smooth", type=int, default=3,
                   help="moving-average window, in frames, used to define 'noise'")
    a = p.parse_args()

    paths = list(a.clips)
    if a.glob:
        paths += [Path(x) for x in sorted(globmod.glob(a.glob))]
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit("no clips given (pass paths, or --glob 'pattern')")

    print(f"smoothing window {a.smooth} frames @ {a.fps:g} fps\n")
    print(f"{'clip':<38} {'T':>4}  {'body noise':>10} {'med m/s':>8} {'max m/s':>8}"
          f"  {'obj noise':>9} {'obj max':>8}")
    print("-" * 96)
    rows = []
    for path in paths:
        r = analyse(path, a.fps, a.smooth)
        rows.append(r)
        print(f"{path.name[:38]:<38} {r['frames']:>4}  {r['body_noise']:>10.3f} "
              f"{r['body_med_speed']:>8.2f} {r['body_max_speed']:>8.2f}  "
              f"{r['obj_noise']:>9.3f} {r['obj_max_speed']:>8.2f}")

    if len(rows) > 1:
        bn = np.median([r["body_noise"] for r in rows])
        on = np.median([r["obj_noise"] for r in rows])
        mx = np.median([r["body_max_speed"] for r in rows])
        print("-" * 96)
        print(f"{'MEDIAN':<38} {'':>4}  {bn:>10.3f} {'':>8} {mx:>8.2f}  {on:>9.3f}")

    print("\nReading it: 'noise' is the share of the finite-differenced velocity that a")
    print(f"{a.smooth}-frame smoother removes. Low = the derivative is real motion and the")
    print("reward's pv/rv terms would grade something meaningful. High = the derivative is")
    print("mostly jitter, and a nonzero pv would train the policy to chase reconstruction")
    print("artefacts. Compare a candidate dataset against mocap (OMOMO) rather than reading")
    print("any single number as good or bad on its own.")


if __name__ == "__main__":
    sys.exit(main())

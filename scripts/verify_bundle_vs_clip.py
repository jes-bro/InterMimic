#!/usr/bin/env python3
"""Did the CARI4D->InterMimic conversion preserve the motion? Rigid-transform
invariants between the CARI4D bundle (.pth) and the installed clip (.pt).

A correct conversion = one rigid transform (upright rotation + x-flip + floor
translation). Rigid transforms CANNOT change relative geometry, so:

  1. per-frame ball SPEED |ball[t+1]-ball[t]| must match EXACTLY
  2. per-frame ball<->human distance must match to a few cm (the clip's root
     joint vs the bundle's SMPL translation differ by a small body-frame offset)
  3. trajectory-shape scalars (path length, lateral drift during any window)
     must match

If these hold, the installed clip IS the bundle's motion and any "looks wrong"
impression is viewing angle or the bundle itself. If they diverge, the
converter corrupted the motion, and the divergence frames localize where.

  python3 scripts/verify_bundle_vs_clip.py \
      --bundle /simurgh2/projects/ret-hoi/CARI4D/output/opt/.../Date03_Sub01_bball_dribble.pth \
      --clip InterAct/behave_cari4d_optj3d/sub100_bball_000.pt \
      --window 62 98        # optional: report lateral drift over these frames
"""
import argparse

import numpy as np
import torch

I_ROOT = slice(0, 3)
I_OBJP = slice(318, 321)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="CARI4D export .pth (has 'pr')")
    ap.add_argument("--clip", required=True, help="installed 591-channel .pt")
    ap.add_argument("--window", type=int, nargs=2, default=None,
                    help="frame range for the lateral-drift report (e.g. the shot)")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    b = torch.load(args.bundle, map_location="cpu")
    pr = b["pr"] if "pr" in b else b
    print(f"bundle keys: {sorted(b.keys()) if hasattr(b, 'keys') else type(b)}")
    print(f"pr keys:     {sorted(pr.keys())}")
    ball_b = np.asarray(pr["pose_abs"])[:, :3, 3].astype(np.float64)
    hum_b = np.asarray(pr["smpl_t"]).astype(np.float64)

    c = torch.load(args.clip, map_location="cpu")
    ball_c = c[:, I_OBJP].double().numpy()
    hum_c = c[:, I_ROOT].double().numpy()

    Tb, Tc = len(ball_b), len(ball_c)
    print(f"\nframes: bundle {Tb} | clip {Tc}" + ("" if Tb == Tc else "  <-- MISMATCH: "
          "conversion resampled or trimmed; all per-frame checks below are void"))
    T = min(Tb, Tc)

    # 1. speed profile -- exact under any rigid transform
    sp_b = np.linalg.norm(np.diff(ball_b[:T], axis=0), axis=1)
    sp_c = np.linalg.norm(np.diff(ball_c[:T], axis=0), axis=1)
    d = np.abs(sp_b - sp_c)
    print(f"\n== 1. ball speed profile (must match ~exactly) ==")
    print(f"  max |diff| {d.max():.4f} m/frame at frame {int(d.argmax())}; mean {d.mean():.4f}")
    print(f"  read: <0.01 = motion preserved; larger = converter changed the trajectory")

    # 2. ball<->human distance -- rigid-invariant up to the root-vs-smpl_t offset
    db = np.linalg.norm(ball_b[:T] - hum_b[:T], axis=1)
    dc = np.linalg.norm(ball_c[:T] - hum_c[:T], axis=1)
    dd = np.abs(db - dc)
    print(f"\n== 2. ball<->human distance (match to ~0.1m; offset explains small constant) ==")
    print(f"  {'frame':>5s} {'bundle':>8s} {'clip':>8s} {'|diff|':>8s}")
    for i in range(0, T, args.every):
        print(f"  {i:5d} {db[i]:8.3f} {dc[i]:8.3f} {dd[i]:8.3f}")
    print(f"  max |diff| {dd.max():.3f} m at frame {int(dd.argmax())}")

    # 3. shape scalars + lateral drift
    print(f"\n== 3. trajectory shape ==")
    print(f"  ball path length: bundle {sp_b.sum():.2f} m | clip {sp_c.sum():.2f} m")
    if args.window:
        a, z = args.window
        z = min(z, T - 1)
        for tag, ball in (("bundle", ball_b), ("clip", ball_c)):
            seg = ball[a:z + 1]
            vert = seg[:, 2] if tag == "clip" else seg[:, np.abs(seg - seg[0]).max(0).argmax()]
            # lateral = displacement perpendicular to the segment's max-rise axis:
            # report per-axis net displacement instead of guessing axes
            net = seg[-1] - seg[0]
            peak = np.abs(seg - seg[0]).max(0)
            print(f"  {tag}: frames {a}-{z} net displacement {np.round(net, 2)} | "
                  f"per-axis peak excursion {np.round(peak, 2)}")
        print(f"  read: the two rows' excursion MAGNITUDES should match as a set "
              f"(axes are permuted/flipped by the rigid transform). A lateral "
              f"excursion present in the bundle but missing in the clip = converter "
              f"flattened it; present in both = the motion is there and the replay "
              f"camera angle is hiding it.")


if __name__ == "__main__":
    main()

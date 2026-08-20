#!/usr/bin/env python3
"""The two numbers the speed-profile check can't see: is the conversion's
transform MIRRORED (det = -1) or TILTED (gravity mapped away from -z)?

The bundle ball and the installed clip ball are exact per-frame
correspondences, so the transform between them is solved outright (SVD /
orthogonal Procrustes, reflection ALLOWED so a mirror can be detected rather
than silently corrected):

  1. det(R): +1 proper rotation, -1 the world is mirrored
  2. residual RMS: ~0 confirms the correspondence really is one transform
  3. gravity: mean free-flight acceleration (a physical probe, no conventions)
     in each frame; report the angle between R @ g_bundle and g_clip, and
     between g_clip and world -z. Both ~0 = no tilt.

Optionally referee against a model-free triangulation npz (--triangulated):
Procrustes the installed clip ball onto the measured world points; report
residuals + det there too. Whoever disagrees with the measurement owns the bug.

  PYTHONPATH=/simurgh2/projects/ret-hoi/CARI4D python3 scripts/ball_transform_audit.py \
      --bundle .../Date03_Sub01_bball_dribble.pth \
      --clip InterAct/behave_cari4d_optj3d/sub100_bball_000.pt \
      --flight 62 97 \
      [--triangulated /simurgh2/projects/ret-hoi/CARI4D/bball_xyz_all.npz]
"""
import argparse

import numpy as np
import torch


def solve_transform(P, Q):
    """Best orthogonal (rotation OR reflection) R, t with Q ~= P @ R.T + t."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = (U @ Vt).T                      # unconstrained: det may be -1 (mirror)
    t = Q.mean(0) - P.mean(0) @ R.T
    res = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return R, t, res


def flight_gravity(P, a, z):
    """Mean acceleration over free-flight frames [a, z] -- the physical 'down'."""
    acc = np.diff(P, n=2, axis=0)       # per frame^2; direction is what matters
    seg = acc[a:z - 1]
    g = seg.mean(0)
    return g / np.linalg.norm(g)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--flight", type=int, nargs=2, default=[62, 97],
                    help="free-flight frame window for the gravity probe")
    ap.add_argument("--triangulated", help="model-free measured ball npz (referee)")
    args = ap.parse_args()

    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    pr = b["pr"] if "pr" in b else b
    P = np.asarray(pr["pose_abs"])[:, :3, 3].astype(float)
    c = torch.load(args.clip, map_location="cpu")
    Q = c[:, 318:321].double().numpy()
    T = min(len(P), len(Q))
    P, Q = P[:T], Q[:T]

    R, t, res = solve_transform(P, Q)
    print("== bundle -> clip transform (solved from the 101 correspondences) ==")
    print(f"  residual RMS {np.sqrt((res**2).mean()):.4f} m  max {res.max():.4f} m "
          f"(~0 = the pair really is ONE transform)")
    print(f"  det(R) = {np.linalg.det(R):+.3f}   (+1 proper rotation, -1 MIRRORED)")

    a, z = args.flight
    gb, gc = flight_gravity(P, a, z), flight_gravity(Q, a, z)
    print(f"\n== gravity probe (free-flight accel, frames {a}-{z}) ==")
    print(f"  bundle gravity dir {np.round(gb, 3)}")
    print(f"  clip   gravity dir {np.round(gc, 3)}")
    print(f"  angle(R @ g_bundle, g_clip) = {ang(R @ gb, gc):.2f} deg  (~0 = transform consistent)")
    print(f"  angle(g_clip, world -z)     = {ang(gc, np.array([0, 0, -1.0])):.2f} deg  "
          f"(~0 = clip world is upright, no tilt)")

    if args.triangulated:
        z_ = np.load(args.triangulated, allow_pickle=True)
        print(f"\n== referee: triangulated npz keys: {sorted(z_.files)} ==")
        key = next((k for k in ("xyz", "world_xyz", "ball_xyz", "points") if k in z_.files), None)
        if key is None:
            print("  no recognized xyz key -- paste the keys above and we'll wire it")
            return
        M = np.asarray(z_[key]).astype(float)
        fr = np.asarray(z_["frames"]).astype(int) if "frames" in z_.files else np.arange(len(M))
        keep = fr < T
        M, fr = M[keep], fr[keep]
        Rr, tr, rr = solve_transform(M, Q[fr])
        print(f"  clip vs MEASURED ball ({len(fr)} frames): residual RMS "
              f"{np.sqrt((rr**2).mean()):.3f} m  median {np.median(rr):.3f}  max {rr.max():.3f}")
        print(f"  det(R) = {np.linalg.det(Rr):+.3f}")
        print("  read: residuals ~= the 0.03-0.2m CARI4D reports = the installed ball IS the "
              "measured ball; det -1 = our conversion mirrored the world.")


if __name__ == "__main__":
    main()

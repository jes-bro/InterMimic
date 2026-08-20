#!/usr/bin/env python3
"""Per-DOF-slot audit across installed clips, with GROUND-TRUTH joint names
read from the subject's MJCF (body order = sim's dof order, body k -> dof k-1;
no hand-maintained mapping tables that could mislabel joints).

For every one of the 51 dof slots, report:
  - the MJCF body name of that slot
  - temporal motion (mean geodesic frame-to-frame step, deg) per clip --
    a template-filled channel is constant or canned
  - cross-clip identity: geodesic error between clips A/B (and B/C) per slot --
    0.00 deg across INDEPENDENT reconstructions = the channel never came from
    the exports at all

  python3 scripts/audit_dof_slots.py \
      --clips InterAct/behave_cari4d/sub100_bball_000.pt \
              InterAct/behave_cari4d_rectinj3/sub100_bball_000.pt \
              InterAct/behave_cari4d_optj3d/sub100_bball_000.pt \
      --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402  (validated MJCF parser)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dof(path):
    c = torch.load(path, map_location="cpu")
    T = c.shape[0]
    return c[:, 9:9 + 153].double().numpy().reshape(T, 51, 3)


def geo_deg(a, b):
    """per-frame geodesic angle (deg) between two (T,3) axis-angle tracks"""
    return np.degrees((sRot.from_rotvec(a).inv() * sRot.from_rotvec(b)).magnitude())


def motion_deg(a):
    """mean frame-to-frame rotation step (deg) -- 0 = frozen channel"""
    return float(np.degrees((sRot.from_rotvec(a[:-1]).inv() * sRot.from_rotvec(a[1:])).magnitude()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="2-3 installed .pt clips to cross-compare")
    ap.add_argument("--mjcf", default=os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml"))
    args = ap.parse_args()

    names = [n for n, _, _ in _parse_mjcf_tree(args.mjcf)]   # body 0 = Pelvis (no dof)
    dofs = [load_dof(p) for p in args.clips]
    T = min(len(d) for d in dofs)
    tags = [os.path.basename(os.path.dirname(p)) for p in args.clips]
    print("clips: " + " | ".join(f"{chr(65+i)}={t}" for i, t in enumerate(tags)) + f"  ({T} frames)")

    hdr = f"  {'slot':>4s} {'joint':14s} " + " ".join(f"move{chr(65+i)}" for i in range(len(dofs)))
    pairs = [(i, i + 1) for i in range(len(dofs) - 1)]
    hdr += "  " + " ".join(f"{chr(65+a)}vs{chr(65+b)}" for a, b in pairs)
    print(hdr + "   (deg; moveX = temporal step, XvsY = cross-clip mean error)")
    for k in range(51):
        name = names[k + 1] if k + 1 < len(names) else f"body{k+1}"
        moves = [motion_deg(d[:T, k]) for d in dofs]
        cross = [geo_deg(dofs[a][:T, k], dofs[b][:T, k]).mean() for a, b in pairs]
        flag = ""
        if any(c < 0.01 for c in cross):
            flag = "  <-- IDENTICAL across independent recons" + \
                   ("" if max(moves) > 0.01 else " AND frozen")
        print(f"  {k:4d} {name:14s} " +
              " ".join(f"{m:5.1f}" for m in moves) + "  " +
              " ".join(f"{c:5.1f}" for c in cross) + flag)


if __name__ == "__main__":
    main()

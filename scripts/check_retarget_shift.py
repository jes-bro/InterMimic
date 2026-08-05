#!/usr/bin/env python3
"""Does contact retargeting re-ground the reference for shorter bodies?

The reference motion is the SOURCE subject's, imposed unscaled on the target
body -- subject height is used only to normalise the reward's position error
(intermimic.py:1856), never to rescale the trajectory. So a target shorter than
the source has its feet off the ground in the reference: it is being asked to
track a pose it physically cannot reach standing.

Leg lengths measured from the MJCFs (thigh+shin+foot):
    sub2 (source) 0.993   sub13 0.996   sub16 0.913   sub10 0.854
so sub16's reference floats ~8cm and sub10's ~14cm.

This prints reference root height for the original clip and for each body's
contact-retargeted version. A retargeted root sitting lower by about the
leg-length deficit is retargeting planting the feet -- which would explain why
the retarget arms are the only ones that reach sub16.

  python3 scripts/check_retarget_shift.py                     # defaults below
  python3 scripts/check_retarget_shift.py --clip sub2_largetable_000.pt \
      --bodies sub16 sub10 sub13 --retarget-dir InterAct/OMOMO_retarget_contact_src2
"""
import argparse
import os

import torch

# root_pos is data_component_order[0], so channels 0:3 of every reference row.
ROOT_Z = 2


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="sub2_largetable_017.pt")
    ap.add_argument("--motion-dir", default="InterAct/OMOMO_new")
    ap.add_argument("--retarget-dir", default="InterAct/OMOMO_retarget_contact_src2")
    ap.add_argument("--bodies", nargs="+", default=["sub2", "sub13", "sub16", "sub10"])
    a = ap.parse_args()

    clip = a.clip if a.clip.endswith(".pt") else a.clip + ".pt"
    orig_path = os.path.join(a.motion_dir, clip)
    if not os.path.exists(orig_path):
        raise SystemExit(f"FATAL: no such clip: {orig_path}")
    o = load(orig_path)
    oz = o[:, ROOT_Z]
    print(f"clip {clip}  ({o.shape[0]} frames)")
    print(f"{'reference':22} {'min z':>8} {'mean z':>8} {'max z':>8} {'shift vs original':>19}")
    print("-" * 72)
    print(f"{'original (source)':22} {oz.min():8.3f} {oz.mean():8.3f} {oz.max():8.3f} {'--':>19}")

    missing = []
    for b in a.bodies:
        p = os.path.join(a.retarget_dir, b, clip)
        if not os.path.exists(p):
            missing.append(b)
            continue
        rz = load(p)[:, ROOT_Z]
        n = min(len(rz), len(oz))
        shift = (rz[:n] - oz[:n]).mean()
        print(f"{'retarget ' + b:22} {rz.min():8.3f} {rz.mean():8.3f} {rz.max():8.3f} {shift:+18.3f}m")

    if missing:
        print(f"\n  no retargeted file for: {' '.join(missing)}")
        print(f"  (looked under {a.retarget_dir}/<body>/{clip})")
    print("\nRead: a shift of about MINUS the leg-length deficit = retargeting is")
    print("lowering the root to plant the feet. Near zero = retargeting is not")
    print("touching root height, and floating is not what it fixes.")


if __name__ == "__main__":
    main()

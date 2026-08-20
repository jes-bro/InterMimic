#!/usr/bin/env python3
"""Left/right swap detector: which WRIST does the work, per the installed clip?

Ground truth (CARI4D per-wrist table + Jess's read of the source footage):
LEFT hand pushes the dribble (frames ~0-14), RIGHT hand takes the catch
(~24-32) and the release (~60). If the installed clip disagrees, the
SMPL-H -> InterMimic conversion swapped left/right somewhere -- a bug every
ball-based audit is blind to.

Wrist indices come from the MJCF by NAME (the sim's own enumeration), so no
hand-maintained mapping table can mislabel them here.

  python3 scripts/check_handedness.py --clip InterAct/behave_cari4d_optj3d/sub100_bball_000.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I_BODY = slice(162, 318)
I_OBJP = slice(318, 321)

# footage-verified truth for the bball take
WINDOWS = [("push",    2, 11, "L"),
           ("catch",  24, 32, "R"),
           ("carry",  50, 58, "R"),
           ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mjcf", default=os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml"))
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    names = [n for n, _, _ in _parse_mjcf_tree(args.mjcf)]
    iL, iR = names.index("L_Wrist"), names.index("R_Wrist")
    c = torch.load(args.clip, map_location="cpu")
    T = c.shape[0]
    bp = c[:, I_BODY].view(T, 52, 3)
    obj = c[:, I_OBJP]
    dL = (bp[:, iL] - obj).norm(dim=-1)
    dR = (bp[:, iR] - obj).norm(dim=-1)

    print(f"clip: {args.clip} ({T} frames)  L_Wrist idx {iL}, R_Wrist idx {iR} (by MJCF name)")
    print(f"\n  {'frame':>5s} {'L-ball':>7s} {'R-ball':>7s}  nearer")
    for i in range(0, T, args.every):
        n = "L" if dL[i] < dR[i] else "R"
        print(f"  {i:5d} {dL[i]:7.3f} {dR[i]:7.3f}  {n}")

    print("\n== verdict vs footage-verified truth ==")
    swapped = correct = 0
    for tag, a, z, want in WINDOWS:
        mL, mR = float(dL[a:z + 1].mean()), float(dR[a:z + 1].mean())
        got = "L" if mL < mR else "R"
        ok = got == want
        correct += ok
        swapped += (not ok)
        print(f"  {tag:6s} frames {a:2d}-{z:2d}: mean L {mL:.3f}  R {mR:.3f}  "
              f"-> {got} does the work (truth: {want})  {'OK' if ok else 'SWAPPED?'}")
    if swapped == len(WINDOWS):
        print("\n  VERDICT: every window inverted -> LEFT/RIGHT SWAP in the conversion.")
    elif swapped == 0:
        print("\n  VERDICT: handedness matches the footage -- no swap in the stored kinematics.")
    else:
        print("\n  VERDICT: mixed -- not a global swap; look at the disagreeing window(s).")


if __name__ == "__main__":
    main()

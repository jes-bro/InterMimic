#!/usr/bin/env python3
"""Diagnostic A: what does the UNRETARGETED reference actually ask each body
to do? Drives every subject's skeleton with sub2's dof_pos + root (exactly the
reference plain teachers must track -- intermimic drives targets with the
source's dof), FKs it with the validated MJCFChain, and measures physical
plausibility per body:

  foot-ground geometry (the "floating" hypothesis):
    pen_cm    mean penetration depth of the lowest foot point, cm (z < 0)
    %pen>2cm  fraction of frames with the lowest foot >2 cm under the floor
    %hover>5  fraction of frames with the LOWEST foot >5 cm above the floor
              (both feet airborne -- can't be balanced on the ground)
  hand-object reachability (known ANTI-correlated with difficulty; kept as
    the control column):
    hand_cm   mean min(hand-to-object) distance on object-contact frames, cm
  foot_dz_cm mean lowest-foot height minus sub2's on the SAME frames, cm --
    the direct floating/penetration mismatch from leg proportions (positive =
    this body's feet dangle above where sub2's were; negative = they'd have to
    penetrate). NOTE %hover uses body ORIGINS (which sit above the mesh even
    for sub2), so read it relative to sub2's row, never as an absolute.

sub2's own row is the calibration: its FK reproduces the stored reference, so
its numbers show what a perfectly-matched body looks like under these stats.
If sub16 stands out on the foot columns where easy bodies (sub10) don't, the
"floating"/ground-contact mechanism is the difficulty driver -- and it also
predicts which OTHER bodies should be hard (testable against gen-2 fold1).

  python3 scripts/scan_reference_feasibility.py                    # all reals, all clips
  python3 scripts/scan_reference_feasibility.py --bodies sub2 sub10 sub16 --limit 5
  python3 scripts/scan_reference_feasibility.py --csv out.csv      # machine-readable
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_contact import (MJCFChain, quat_xyzw_to_mat,               # noqa: E402
                              I_ROOTP, I_ROOTQ, I_DOF, I_OBJP)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_DIR = os.path.join(REPO, "InterAct/OMOMO_new")
I_CONTACT_OBJ = slice(330, 331)     # 1.0 on frames where the object is in contact
PEN_CM, HOVER_CM = 2.0, 5.0         # thresholds for the %-of-frames columns


def foot_hand_indices(chain):
    """Body indices for feet (toes+ankles), hands (wrists), pelvis -- resolved
    from the chain's own body names so a naming change fails loudly here."""
    feet = [i for i, n in enumerate(chain.names)
            if "Toe" in n or "Ankle" in n]
    hands = [i for i, n in enumerate(chain.names) if "Wrist" in n or "Hand" in n]
    pelvis = [i for i, n in enumerate(chain.names) if n == "Pelvis"]
    if len(feet) < 2 or not hands or len(pelvis) != 1:
        raise SystemExit(f"FATAL: body-name lookup failed "
                         f"(feet={len(feet)} hands={len(hands)} pelvis={len(pelvis)}); "
                         f"names[:6]={chain.names[:6]}")
    return feet, hands, pelvis[0]


def clip_stats(chain, clip):
    """One (body, clip) measurement. Returns per-frame arrays so the caller can
    pool across clips before averaging (clips have different lengths)."""
    dof = clip[:, I_DOF].double()
    root_pos = clip[:, I_ROOTP].double()
    root_rot = quat_xyzw_to_mat(clip[:, I_ROOTQ].double())
    pos = chain.fk(dof, root_pos=root_pos, root_rot=root_rot)   # (T,52,3) world
    feet, hands, pelvis = foot_hand_indices(chain)

    lowest_foot = pos[:, feet, 2].min(dim=1).values             # (T,)
    pen = torch.clamp(-lowest_foot, min=0)                      # depth below floor
    obj = clip[:, I_OBJP].double()
    contact = clip[:, I_CONTACT_OBJ].squeeze(-1) > 0.5
    hand_d = (pos[:, hands, :] - obj[:, None, :]).norm(dim=-1).min(dim=1).values
    return {
        "lowest_foot": lowest_foot.numpy(),
        "pen": pen.numpy(),
        "hand_d_contact": hand_d[contact].numpy(),              # only contact frames
    }


def body_row(pooled):
    """Aggregate pooled per-frame arrays into the table columns (cm / %)."""
    lf, pen = np.concatenate(pooled["lowest_foot"]), np.concatenate(pooled["pen"])
    hd = np.concatenate(pooled["hand_d_contact"])
    return {
        "pen_cm": 100 * pen.mean(),
        "pct_pen": 100 * (lf < -PEN_CM / 100).mean(),
        "pct_hover": 100 * (lf > HOVER_CM / 100).mean(),
        "hand_cm": 100 * hd.mean() if len(hd) else float("nan"),
        "lf_all": lf,                       # paired per-frame series for foot_dz
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="*",
                    default=[f"sub{i}" for i in range(1, 18)],
                    help="default: all 17 real subjects (sub4 included -- its "
                         "MJCF parses even though it crashes the simulator)")
    ap.add_argument("--motion-dir", default=MOTION_DIR)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    clips = sorted(glob.glob(os.path.join(args.motion_dir, "sub2_*.pt")))
    if not clips:
        raise SystemExit(f"FATAL: no sub2_*.pt clips in {args.motion_dir}")
    clips = clips[: args.limit] if args.limit else clips
    print(f"[scan] {len(clips)} sub2 clips x {len(args.bodies)} bodies "
          f"(reference = sub2 dof+root on each body's skeleton)")

    rows = {}
    for body in args.bodies:
        chain = MJCFChain(body)
        pooled = {k: [] for k in ("lowest_foot", "pen", "hand_d_contact")}
        for c in clips:
            clip = torch.load(c, map_location="cpu", weights_only=True).detach()
            for k, v in clip_stats(chain, clip).items():
                pooled[k].append(v)
        rows[body] = body_row(pooled)
    if "sub2" not in rows:
        raise SystemExit("FATAL: sub2 must be scanned (it is the paired baseline)")

    lf2 = rows["sub2"]["lf_all"]
    hdr = f"{'body':7s} {'pen_cm':>7s} {'%pen>2cm':>9s} {'%hover>5cm':>11s} {'hand_cm':>8s} {'foot_dz_cm':>11s}"
    print(hdr); print("-" * len(hdr))
    lines = []
    for body in args.bodies:
        r = rows[body]
        dz = 100 * float((r["lf_all"] - lf2).mean())     # paired: same clips/frames
        line = (f"{body:7s} {r['pen_cm']:7.2f} {r['pct_pen']:9.1f} "
                f"{r['pct_hover']:11.1f} {r['hand_cm']:8.2f} {dz:11.2f}")
        print(line)
        lines.append((body, r, dz))
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["body", "pen_cm", "pct_pen_gt2cm", "pct_hover_gt5cm",
                        "hand_cm", "foot_dz_cm"])
            for body, r, dz in lines:
                w.writerow([body, f"{r['pen_cm']:.3f}", f"{r['pct_pen']:.2f}",
                            f"{r['pct_hover']:.2f}", f"{r['hand_cm']:.3f}", f"{dz:.3f}"])
        print(f"[scan] wrote {args.csv}")
    return rows


if __name__ == "__main__":
    main()

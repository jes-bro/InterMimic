#!/usr/bin/env python3
"""relabel_contact_cpr.py -- neutralise the "must not touch" label on the lower
legs of a KNEELING subject. Nothing else changes.

WHY. interact2mimic.py labels every body more than 10 cm from the object as
-1 ("must not touch"); only the four FEET are exempt, and only while they are
on the ground. A CPR subject kneels beside the manikin, so the knees, ankles
and toes sit on the FLOOR, more than 10 cm from the manikin, and get -1. The
contact reward's rcg_all term (intermimic.py compute_cg_reward) penalises ANY
contact force on a -1 body, and the sim cannot tell floor contact from
manikin contact -- so kneeling itself is taxed on every frame. Setting those
six bodies to 0 ("neutral": neither graded for contact nor for its absence)
removes the tax and grades nothing the reference cannot justify.

WHAT IT DOES NOT DO. The hand labels are the converter's own vertex-distance
labels and are already right for a static object (a hand within 2 cm of the
manikin's surface points). They are NOT touched -- the hand relabel scripts
(relabel_contact_human.py / relabel_contact_soccer.py) assume a SPHERE with a
radius from the mesh bounding box, which is ~0.9 m for a manikin, and must
not be run on CPR. +1 on a lower-leg body (a knee actually on the manikin) is
kept: only -1 -> 0. contact_obj (channel 330) is untouched.

    python3 scripts/relabel_contact_cpr.py \\
        --src-dir InterAct/behave_cari4d_cpr --dst-dir InterAct/behave_cari4d_cpr_kn \\
        --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub509.xml
    (--census: report what WOULD change, write nothing)
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree                                   # noqa: E402
from relabel_contact_human import I_CONTACT_HUMAN, I_CONTACT_OBJ          # noqa: E402

LOWER_LEG_BODY_NAMES = ["L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe"]
HAND_BODY_IDS = list(range(17, 33)) + list(range(36, 52))   # what rcg_hand grades (intermimic.py)


def lower_leg_body_ids(mjcf):
    """Indices of the six lower-leg bodies in the clip's 52-body order, from the
    rig by NAME; a rig with a different order or missing bodies is a hard error."""
    names = [n for n, _, _ in _parse_mjcf_tree(mjcf)]
    if len(names) != 52:
        raise SystemExit(f"FATAL: {mjcf} has {len(names)} bodies, expected 52")
    missing = [n for n in LOWER_LEG_BODY_NAMES if n not in names]
    if missing:
        raise SystemExit(f"FATAL: {mjcf} lacks bodies {missing}; names: {names[:12]}...")
    ids = [names.index(n) for n in LOWER_LEG_BODY_NAMES]
    assert not set(ids) & set(HAND_BODY_IDS), "lower-leg ids overlap the hand block?!"
    return ids


def neutralise(t, body_ids):
    """-1 -> 0 on body_ids only. Returns (new tensor, frames changed per body)."""
    out = t.clone()
    ch = out[:, I_CONTACT_HUMAN]
    sub = ch[:, body_ids]
    changed = (sub < -0.5)
    sub[changed] = 0.0
    ch[:, body_ids] = sub
    out[:, I_CONTACT_HUMAN] = ch
    return out, changed.sum(dim=0).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", help="required unless --census")
    ap.add_argument("--mjcf", required=True, help="a 52-body rig -- lower-leg indices by name")
    ap.add_argument("--census", action="store_true", help="report only, write nothing")
    args = ap.parse_args()
    if not args.census and not args.dst_dir:
        ap.error("--dst-dir is required unless --census")

    src = Path(args.src_dir)
    clips = sorted(src.glob("*.pt"))
    if not clips:
        raise SystemExit(f"FATAL: no *.pt in {src}")
    ids = lower_leg_body_ids(args.mjcf)
    if not args.census:
        dst = Path(args.dst_dir)
        dst.mkdir(parents=True, exist_ok=True)

    print(f"{'clip':40s} {'T':>5s}  " + " ".join(f"{n:>7s}" for n in LOWER_LEG_BODY_NAMES) + "   (frames -1 -> 0)")
    tot = [0] * len(ids)
    for f in clips:
        t = torch.load(f, map_location="cpu", weights_only=False).detach()
        if t.dim() != 2 or t.shape[1] < I_CONTACT_HUMAN.stop:
            raise SystemExit(f"FATAL: {f.name}: expected (T, >= {I_CONTACT_HUMAN.stop}) got {tuple(t.shape)}")
        out, n = neutralise(t, ids)
        tot = [a + b for a, b in zip(tot, n)]
        print(f"{f.name:40s} {t.shape[0]:5d}  " + " ".join(f"{k:7d}" for k in n))
        if args.census:
            continue
        # guards: nothing outside the six lower-leg channels moved, contact_obj intact
        others = [i for i in range(52) if i not in ids]
        assert torch.equal(out[:, I_CONTACT_HUMAN][:, others], t[:, I_CONTACT_HUMAN][:, others]), "non-leg flags changed!"
        assert torch.equal(out[:, I_CONTACT_OBJ], t[:, I_CONTACT_OBJ]), "contact_obj changed!"
        mask = torch.ones(t.shape[1], dtype=torch.bool); mask[I_CONTACT_HUMAN] = False
        assert torch.equal(out[:, mask], t[:, mask]), "a channel outside contact_human changed!"
        torch.save(out, dst / f.name)
    print(f"{'TOTAL':40s} {'':5s}  " + " ".join(f"{k:7d}" for k in tot))
    if not args.census:
        print(f"wrote {len(clips)} clips to {dst}")


if __name__ == "__main__":
    main()

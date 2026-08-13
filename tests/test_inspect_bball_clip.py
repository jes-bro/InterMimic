#!/usr/bin/env python3
"""Tests for scripts/inspect_bball_clip.py's logic on a fabricated clip --
the data is cluster-only, so pin the math here: span grouping, wrist-index
resolution by name, and the hand-distance/lowest-z extraction.

Run:  python tests/test_inspect_bball_clip.py   (exit 0 = all green)
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import inspect_bball_clip as ib  # noqa: E402


def test_contact_spans():
    flags = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
    assert ib.contact_spans(flags) == [(0, 1, True), (2, 4, False),
                                       (5, 7, True), (8, 8, False)]
    assert ib.contact_spans(torch.zeros(3, dtype=torch.bool)) == [(0, 2, False)]
    print("ok: contact span grouping")


def test_wrist_indices_from_real_mjcf():
    # any per-subject MJCF present locally works; skip loudly if none rcloned
    import glob
    mjcfs = glob.glob(os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml"))
    if not mjcfs:
        print("SKIP: no per-subject MJCFs present locally")
        return
    idx, names = ib.wrist_indices(mjcfs[0])
    assert names[idx[0]] == "L_Wrist" and names[idx[1]] == "R_Wrist"
    assert len(names) == 52
    print(f"ok: wrist indices resolved by name ({idx})")


def test_extraction_math():
    # fabricate a 2-frame clip: known ball + body positions
    T = 2
    c = torch.zeros(T, 591)
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0            # all bodies at z=1
    bp[0, 5, 2] = 0.23           # frame 0: lowest body at the floor-offset height
    bp[1, 5, 2] = 0.0
    bp[0, 16] = torch.tensor([1.0, 0.0, 1.0])   # a "wrist" at x=1
    c[:, ib.I_BODY] = bp.view(T, -1)
    c[0, ib.I_OBJP] = torch.tensor([1.0, 0.0, 1.3])  # ball 0.3 above that wrist
    c[0, ib.I_CONTACT_OBJ] = 1.0
    bpv = c[:, ib.I_BODY].view(T, 52, 3)
    lowest = bpv[:, :, 2].min(dim=1).values
    assert abs(lowest[0] - 0.23) < 1e-6 and abs(lowest[1] - 0.0) < 1e-6
    d = (bpv[0, [16], :] - c[0, ib.I_OBJP][None, :]).norm(dim=-1).min()
    assert abs(d - 0.3) < 1e-6
    assert bool(c[0, ib.I_CONTACT_OBJ] > 0.5) and not bool(c[1, ib.I_CONTACT_OBJ] > 0.5)
    print("ok: lowest-z / hand-distance / flag extraction")


if __name__ == "__main__":
    test_contact_spans()
    test_wrist_indices_from_real_mjcf()
    test_extraction_math()
    print("ALL GREEN")

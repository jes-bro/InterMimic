#!/usr/bin/env python3
"""Tests for scripts/scan_reference_feasibility.py.

1. Calibration: sub2's stats from the FK must match the same stats computed
   directly from the clip's STORED body_pos (the FK is validated to 0.1mm, so
   any disagreement means the scanner indexes the wrong bodies/columns).
2. Sensitivity control: a different-proportioned body (sub16) must show a
   nonzero pelvis offset vs sub2 -- the scan can actually detect differences.
3. body_row math on hand-built arrays (thresholds and unit conversions).

Run:  python tests/test_scan_reference_feasibility.py   (exit 0 = all green)
"""
import glob
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import scan_reference_feasibility as scan  # noqa: E402
from retarget_contact import MJCFChain, I_BODY  # noqa: E402

CLIP = sorted(glob.glob(os.path.join(scan.MOTION_DIR, "sub2_*.pt")))[0]


def test_calibration_against_stored_body_pos():
    clip = torch.load(CLIP, map_location="cpu", weights_only=True).detach()
    chain = MJCFChain("sub2")
    s = scan.clip_stats(chain, clip)
    stored = clip[:, I_BODY].double().view(-1, 52, 3)
    feet, _, pelvis = scan.foot_hand_indices(chain)
    lf_stored = stored[:, feet, 2].min(dim=1).values.numpy()
    assert np.abs(s["lowest_foot"] - lf_stored).max() < 1e-3, \
        np.abs(s["lowest_foot"] - lf_stored).max()          # FK vs data: <1 mm
    print("ok: sub2 scan matches stored body_pos (<1mm)")


def test_sensitivity_other_body_differs():
    clip = torch.load(CLIP, map_location="cpu", weights_only=True).detach()
    s2 = scan.clip_stats(MJCFChain("sub2"), clip)
    s16 = scan.clip_stats(MJCFChain("sub16"), clip)
    # same root -> proportions show up downstream, in the FEET. sub16's legs
    # differ from sub2's, so the lowest-foot trajectory must differ (>3mm mean).
    dfoot = np.abs(s16["lowest_foot"] - s2["lowest_foot"]).mean()
    assert dfoot > 0.003, dfoot
    print(f"ok: sub16 foot geometry differs from sub2 (mean {100*dfoot:.2f} cm)")


def test_body_row_math():
    pooled = {
        "lowest_foot": [np.array([-0.05, 0.0, 0.10, 0.01])],  # 1 pen>2cm, 1 hover>5cm
        "pen": [np.array([0.05, 0.0, 0.0, 0.0])],
        "hand_d_contact": [np.array([0.10, 0.30])],
    }
    r = scan.body_row(pooled)
    assert abs(r["pen_cm"] - 100 * 0.05 / 4) < 1e-9
    assert abs(r["pct_pen"] - 25.0) < 1e-9 and abs(r["pct_hover"] - 25.0) < 1e-9
    assert abs(r["hand_cm"] - 20.0) < 1e-9
    assert np.allclose(r["lf_all"], pooled["lowest_foot"][0])
    print("ok: body_row thresholds and unit conversions")


if __name__ == "__main__":
    test_calibration_against_stored_body_pos()
    test_sensitivity_other_body_differs()
    test_body_row_math()
    print("ALL GREEN")

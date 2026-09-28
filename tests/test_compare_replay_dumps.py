#!/usr/bin/env python3
"""Tests for scripts/compare_replay_dumps.py.

The skeletons are the REAL per-subject MJCFs (sub2 = control, sub4 = suspect):
their parent-relative rest offsets are read from the files and driven by one
random joint-angle stream via forward kinematics, which is exactly what the
simulator's play_dataset does. So:
  - sub4's real bone offsets under sub2's angles must PASS (only lengths differ),
  - the same with one knee's rotation corrupted must FAIL and name that knee,
  - a NaN frame must FAIL loudly.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_compare_replay_dumps.py -q
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import compare_replay_dumps as crd  # noqa: E402

ASSETS = REPO / "isaacgym/src/intermimic/data/assets/smplx"
CONTROL, SUSPECT = ASSETS / "smplx_omomo_sub2.xml", ASSETS / "smplx_omomo_sub4.xml"


def test_real_mjcfs_present():
    assert CONTROL.is_file() and SUSPECT.is_file(), "real sub2/sub4 MJCFs required"


def rest_offsets(mjcf):
    """Parent-relative rest-pose offset of every body, in MJCF document order."""
    root = ET.parse(mjcf).getroot()
    return np.array([[float(x) for x in b.get("pos", "0 0 0").split()] for b in root.iter("body")])


def fk(offsets, parents, local_rots, root_pos):
    """Forward kinematics: local_rots (T,B) scipy Rotations per body (body 0 =
    root orientation). Returns global pos (T,B,3), quats (T,B,4) as (x,y,z,w)."""
    T, B = local_rots.shape
    pos = np.zeros((T, B, 3)); quat = np.zeros((T, B, 4))
    grot = np.empty((T, B), dtype=object)
    for t in range(T):
        for b in range(B):
            if parents[b] < 0:
                grot[t, b] = local_rots[t, b]
                pos[t, b] = root_pos[t]
            else:
                p = parents[b]
                grot[t, b] = grot[t, p] * local_rots[t, b]
                pos[t, b] = pos[t, p] + grot[t, p].apply(offsets[b])
            quat[t, b] = grot[t, b].as_quat()             # scipy = (x,y,z,w), same as Isaac Gym
    return pos, quat


def angle_stream(names, T=12, seed=0):
    """One random joint-angle stream shared by both bodies (T,B) Rotations."""
    rng = np.random.default_rng(seed)
    B = len(names)
    rots = np.empty((T, B), dtype=object)
    for t in range(T):
        for b in range(B):
            rots[t, b] = R.from_rotvec(rng.uniform(-0.6, 0.6, 3))
    root = np.cumsum(rng.uniform(-0.02, 0.02, (T, 3)), 0) + [0, 0, 0.95]
    return rots, root


def test_same_motion_different_bones_passes():
    names, parents = crd.mjcf_tree(CONTROL)
    assert len(names) == 52
    rots, root = angle_stream(names)
    cp, cr = fk(rest_offsets(CONTROL), parents, rots, root)
    sp, sr = fk(rest_offsets(SUSPECT), parents, rots, root)
    ok, lines = crd.compare(sp, sr, cp, cr, names, parents)
    assert ok, "\n".join(lines)
    assert any("skeleton difference" in l for l in lines)


def test_corrupted_knee_fails_and_is_named():
    names, parents = crd.mjcf_tree(CONTROL)
    rots, root = angle_stream(names)
    cp, cr = fk(rest_offsets(CONTROL), parents, rots, root)
    bad = rots.copy()
    k = names.index("L_Knee")
    for t in range(len(bad)):
        bad[t, k] = bad[t, k] * R.from_rotvec([0.0, 0.5, 0.0])   # wrong knee angle
    sp, sr = fk(rest_offsets(SUSPECT), parents, bad, root)
    ok, lines = crd.compare(sp, sr, cp, cr, names, parents)
    assert not ok
    fails = "\n".join(l for l in lines if "FAIL" in l)
    assert "L_Knee" in fails or "L_Ankle" in fails, fails   # knee rotation or its child's direction


def test_nan_frame_fails_loudly():
    names, parents = crd.mjcf_tree(CONTROL)
    rots, root = angle_stream(names, T=6)
    cp, cr = fk(rest_offsets(CONTROL), parents, rots, root)
    sp, sr = fk(rest_offsets(SUSPECT), parents, rots, root)
    sp[3, names.index("R_Knee")] = np.nan
    ok, lines = crd.compare(sp, sr, cp, cr, names, parents)
    assert not ok and any("NaN/inf" in l and "R_Knee" in l for l in lines), lines

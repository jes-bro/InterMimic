#!/usr/bin/env python3
"""rotate_pt.py --drop-to-floor: seats the scene on the lowest of ALL bodies by
default (kneeling clips land on the knees), or on the corrected four feet
[3, 4, 7, 8] with --floor-bodies feet.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_rotate_pt_floor.py -v
"""
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts/rotate_pt.py")
KNEE_L, ANKLE_L, TOE_L, KNEE_R, ANKLE_R, TOE_R, SPINE, CHEST = 2, 3, 4, 6, 7, 8, 10, 11


def kneeling_clip(T=20):
    """A kneeler: knees at z=0.30 (the lowest bodies), toes at 0.32, ankles 0.35,
    spine/chest at 0.9/1.0, everything else at 1.2. Object at z=0.50. Root at 0.8.
    Identity rotations. The whole scene sits 0.30 m too high."""
    t = torch.zeros(T, 591)
    body = torch.full((T, 52, 3), 1.2)
    body[:, [KNEE_L, KNEE_R], 2] = 0.30
    body[:, [TOE_L, TOE_R], 2] = 0.32
    body[:, [ANKLE_L, ANKLE_R], 2] = 0.35
    body[:, SPINE, 2], body[:, CHEST, 2] = 0.9, 1.0
    t[:, 162:318] = body.reshape(T, -1)
    t[:, 2] = 0.8
    t[:, 320] = 0.50
    for s in (slice(3, 7), slice(321, 325)):
        t[:, s] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rot = torch.zeros(T, 52, 4); rot[:, :, 3] = 1.0
    t[:, 383:591] = rot.reshape(T, -1)
    return t


def _run(src, out, *extra):
    r = subprocess.run([sys.executable, SCRIPT, str(src), "--axis", "x", "--degrees", "0",
                        "--drop-to-floor", "--out", str(out), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout, torch.load(out, weights_only=False)


def test_default_seats_on_the_lowest_of_all_bodies(tmp_path):
    src = tmp_path / "in.pt"; torch.save(kneeling_clip(), src)
    log, d = _run(src, tmp_path / "out.pt")
    body = d[:, 162:318].reshape(-1, 52, 3)
    assert torch.allclose(body[:, [KNEE_L, KNEE_R], 2], torch.zeros(20, 2), atol=1e-6)   # knees on the floor
    assert torch.allclose(d[:, 2], torch.full((20,), 0.5), atol=1e-6)                     # root moved by -0.30
    assert torch.allclose(d[:, 320], torch.full((20,), 0.2), atol=1e-6)                   # object moved with it
    assert "drop-to-floor (all)" in log and "-0.300" in log


def test_feet_mode_uses_the_corrected_foot_indices(tmp_path):
    src = tmp_path / "in.pt"; torch.save(kneeling_clip(), src)
    log, d = _run(src, tmp_path / "out.pt", "--floor-bodies", "feet")
    body = d[:, 162:318].reshape(-1, 52, 3)
    # lowest FOOT body is the toe at 0.32 -> shift -0.32: toes on the floor, knees 2 cm under
    assert torch.allclose(body[:, [TOE_L, TOE_R], 2], torch.zeros(20, 2), atol=1e-6)
    assert torch.allclose(body[:, [KNEE_L, KNEE_R], 2], torch.full((20, 2), -0.02), atol=1e-6)
    # the OLD buggy list [7, 8, 10, 11] would have seated on R_Toe (0.32) too here, but
    # a left-foot-only clip separates them: only the left toe on the ground
    t = kneeling_clip(); b = t[:, 162:318].reshape(-1, 52, 3); b[:, [TOE_R, ANKLE_R], 2] = 0.9
    t[:, 162:318] = b.reshape(20, -1); torch.save(t, src)
    log, d = _run(src, tmp_path / "out2.pt", "--floor-bodies", "feet")
    body = d[:, 162:318].reshape(-1, 52, 3)
    assert torch.allclose(body[:, TOE_L, 2], torch.zeros(20), atol=1e-6)                  # left toe found
    assert "drop-to-floor (feet)" in log

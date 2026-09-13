#!/usr/bin/env python3
"""relabel_contact_soccer.py: feet only, hands untouched, guard and sweep.

Fixture clips are synthetic 591-channel tensors with a known geometry: a ball at
a fixed centre and foot bodies moved onto / off its surface on chosen frames.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_relabel_contact_soccer.py -v
"""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
soc = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("relabel_contact_soccer", os.path.join(REPO, "scripts/relabel_contact_soccer.py")))
soc.__spec__.loader.exec_module(soc)

FEET = [3, 4, 7, 8]          # arbitrary indices for the fixture; the script resolves real ones by name
HANDS = list(range(17, 33)) + list(range(36, 52))
R = 0.11


def make_clip(T=12, touch_frames=(), claim_frames=(), hand_claim_frames=()):
    """Ball at (1,0,0.11). Foot bodies 2 m away except on touch_frames, where the
    L_Toe body sits 1 cm inside the surface. Source flags: +1 on claim_frames for
    the feet, +1 on hand_claim_frames for L_Wrist (must survive untouched)."""
    t = torch.zeros(T, 591)
    bp = torch.full((T, 52, 3), 3.0)
    centre = torch.tensor([1.0, 0.0, R])
    for f in touch_frames:
        bp[f, FEET[1]] = centre + torch.tensor([R - 0.01, 0.0, 0.0])
    t[:, soc.I_BODY] = bp.view(T, -1)
    t[:, soc.I_OBJP] = centre
    ch = torch.zeros(T, 52)
    for f in claim_frames:
        ch[f, FEET] = 1.0
    for f in hand_claim_frames:
        ch[f, 17] = 1.0
    ch[:, 20] = -1.0                    # a must-not-touch on a non-foot body, must survive
    t[:, soc.I_CONTACT_HUMAN] = ch
    t[:, soc.I_CONTACT_OBJ] = torch.tensor([1.0 if f in claim_frames else 0.0 for f in range(T)])
    return t


def test_relabel_rewrites_feet_only():
    t = make_clip(touch_frames=(4, 5, 6), claim_frames=(0, 1, 4, 5, 6, 9), hand_claim_frames=(2, 3))
    out, touching = soc.relabel(t, R, 0.02, smooth=1, keep_contact_obj=False, body_ids=FEET)
    feet = out[:, soc.I_CONTACT_HUMAN][:, FEET]
    assert feet[4:7, 1].tolist() == [1.0, 1.0, 1.0]           # geometry says touching -> +1
    assert feet[[0, 1, 9]].abs().sum() == 0                   # false claims cleared to 0
    hands = out[:, soc.I_CONTACT_HUMAN][:, HANDS]
    assert torch.equal(hands, t[:, soc.I_CONTACT_HUMAN][:, HANDS])   # hands untouched
    assert (out[:, soc.I_CONTACT_HUMAN][:, 20] == -1.0).all()        # -1 elsewhere survives
    assert out[:, soc.I_CONTACT_OBJ].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert torch.equal(out[:, soc.I_BODY], t[:, soc.I_BODY])


def test_census_sweep_separates_claimed_from_free():
    t = make_clip(touch_frames=(4, 5, 6), claim_frames=(4, 5, 6))
    st = soc.census(t, R, 0.02, FEET)
    assert st["ever_touches"] and st["claimed_contact_frames"] == 3 and st["unearnable_frames"] == 0
    assert st["gap_claimed"].max() < 0 and st["gap_free"].min() > 1.0
    assert st["min_gap"] == pytest.approx(-0.01, abs=1e-6)


def test_guard_and_threshold_required(tmp_path):
    """Refuses a clip with no foot contact; refuses to write without --threshold."""
    src = tmp_path / "src"; src.mkdir()
    torch.save(make_clip(touch_frames=(), claim_frames=(2,)), src / "sub405_ballx_000.pt")
    mjcf = tmp_path / "rig.xml"
    names = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"] + \
            [f"B{i}" for i in range(9, 52)]
    names[17], names[36] = "L_Wrist", "R_Wrist"
    body = "".join(f'<body name="{n}" pos="0 0 0"><joint name="{n}_j" type="hinge" axis="1 0 0"/>' for n in names)
    mjcf.write_text(f'<mujoco><worldbody>{body}{"</body>" * len(names)}</worldbody></mujoco>')
    ids = soc.foot_body_ids(str(mjcf))
    assert ids == [3, 4, 7, 8]
    script = os.path.join(REPO, "scripts/relabel_contact_soccer.py")
    r = subprocess.run([sys.executable, script, "--src-dir", str(src), "--dst-dir", str(tmp_path / "dst"),
                        "--mjcf", str(mjcf)], capture_output=True, text=True)
    assert r.returncode != 0 and "--threshold is required" in r.stderr
    r = subprocess.run([sys.executable, script, "--src-dir", str(src), "--dst-dir", str(tmp_path / "dst"),
                        "--mjcf", str(mjcf), "--threshold", "0.02"], capture_output=True, text=True)
    assert r.returncode != 0 and "no foot body ever comes within" in r.stderr
    assert not (tmp_path / "dst").exists()
    r = subprocess.run([sys.executable, script, "--src-dir", str(src), "--mjcf", str(mjcf), "--census"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "THRESHOLD SWEEP" in r.stdout and "would refuse 1 clip" in r.stdout


def test_basketball_script_untouched():
    """The hand script must not have been edited for this feature."""
    out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "scripts/relabel_contact_human.py"],
                         cwd=REPO)
    assert out.returncode == 0, "scripts/relabel_contact_human.py has uncommitted changes"

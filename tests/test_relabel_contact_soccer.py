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

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
soc = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("relabel_contact_soccer", os.path.join(REPO, "scripts/relabel_contact_soccer.py")))
soc.__spec__.loader.exec_module(soc)

FEET = [3, 4, 7, 8]          # arbitrary indices for the fixture; the script resolves real ones by name
HANDS = list(range(17, 33)) + list(range(36, 52))
R = 0.11


TOE_BOX_HALF = 0.02      # the fixture rig's toe geom: a 4 cm cube at the body origin


def make_clip(T=12, touch_frames=(), claim_frames=(), hand_claim_frames=()):
    """Ball at (1,0,0.11). Foot bodies 2 m away except on touch_frames, where the
    L_Toe body ORIGIN sits 1 cm inside the ball surface (so its 2 cm box surface is
    3 cm inside). All body rotations identity. Source flags: +1 on claim_frames
    for the feet, +1 on hand_claim_frames for L_Wrist (must survive untouched)."""
    t = torch.zeros(T, 591)
    bp = torch.full((T, 52, 3), 3.0)
    centre = torch.tensor([1.0, 0.0, R])
    for f in touch_frames:
        bp[f, FEET[1]] = centre + torch.tensor([R - 0.01, 0.0, 0.0])
    t[:, soc.I_BODY] = bp.view(T, -1)
    t[:, soc.I_OBJP] = centre
    rot = torch.zeros(T, 52, 4); rot[:, :, 3] = 1.0                  # identity, xyzw
    t[:, soc.I_BODY_ROT] = rot.view(T, -1)
    ch = torch.zeros(T, 52)
    for f in claim_frames:
        ch[f, FEET] = 1.0
    for f in hand_claim_frames:
        ch[f, 17] = 1.0
    ch[:, 20] = -1.0                    # a must-not-touch on a non-foot body, must survive
    t[:, soc.I_CONTACT_HUMAN] = ch
    t[:, soc.I_CONTACT_OBJ] = torch.tensor([1.0 if f in claim_frames else 0.0 for f in range(T)])
    return t


def fixture_geoms():
    """Each foot body: one cube of half-size TOE_BOX_HALF at its origin, unrotated."""
    g = dict(type="box", pos=np.zeros(3), R=np.eye(3), size=np.full(3, TOE_BOX_HALF))
    return {n: [g] for n in soc.FOOT_BODY_NAMES}


def test_relabel_rewrites_feet_only():
    t = make_clip(touch_frames=(4, 5, 6), claim_frames=(0, 1, 4, 5, 6, 9), hand_claim_frames=(2, 3))
    out, touching = soc.relabel(t, R, 0.02, smooth=1, keep_contact_obj=False, body_ids=FEET,
                                geoms=fixture_geoms())
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
    st = soc.census(t, R, 0.02, FEET, fixture_geoms())
    assert st["ever_touches"] and st["claimed_contact_frames"] == 3 and st["unearnable_frames"] == 0
    assert st["gap_claimed"].max() < 0 and st["gap_free"].min() > 1.0
    # origin 1 cm inside the ball surface, box face 2 cm further out toward the
    # ball: the box SURFACE is 3 cm inside
    assert st["min_gap"] == pytest.approx(-0.01 - TOE_BOX_HALF, abs=1e-6)
    # origin-based fallback (geoms=None) still gives the old -1 cm
    assert soc.census(t, R, 0.02, FEET)["min_gap"] == pytest.approx(-0.01, abs=1e-6)


def test_geom_distances_box_capsule_sphere():
    """Point-to-geom signed distances, and the rotation/offset handling."""
    box = dict(type="box", pos=np.zeros(3), R=np.eye(3), size=np.array([0.1, 0.05, 0.02]))
    q = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.05], [0.1, 0.05, 0.02]])
    d = soc._point_to_geom(q, box)
    assert d[0] == pytest.approx(-0.02)            # deepest inside: nearest face is z
    assert d[1] == pytest.approx(0.2)              # 0.2 beyond the +x face
    assert d[2] == pytest.approx(0.03)             # 0.03 above the top face
    assert d[3] == pytest.approx(0.0)              # on a corner
    cap = dict(type="capsule", pos=np.zeros(3), R=np.eye(3), size=np.array([0.03, 0.1]))
    d = soc._point_to_geom(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.2]]), cap)
    assert d[0] == pytest.approx(-0.03)
    assert d[1] == pytest.approx(np.hypot(0.05, 0.1) - 0.03)
    sph = dict(type="sphere", pos=np.zeros(3), R=np.eye(3), size=np.array([0.04]))
    assert soc._point_to_geom(np.array([[0.0, 0.1, 0.0]]), sph)[0] == pytest.approx(0.06)
    # a body rotated 90 deg about z with a box offset along its local +x: the
    # ball straight ahead in WORLD +y must be found in front of the box face
    t = make_clip(T=1)
    T = 1
    bp = torch.full((T, 52, 3), 3.0); bp[0, FEET[1]] = torch.tensor([1.0, -0.2, R])   # ball at (1,0,R)
    t[:, soc.I_BODY] = bp.view(T, -1)
    rot = torch.zeros(T, 52, 4); rot[:, :, 3] = 1.0
    rot[0, FEET[1]] = torch.tensor(sRot.from_euler("z", 90, degrees=True).as_quat(), dtype=torch.float32)
    t[:, soc.I_BODY_ROT] = rot.view(T, -1)
    geoms = {n: [dict(type="box", pos=np.array([0.1, 0.0, 0.0]), R=np.eye(3), size=np.full(3, 0.02))]
             for n in soc.FOOT_BODY_NAMES}
    gap = soc.surface_gap(t, R, FEET, geoms)[0, 1].item()
    # body local +x points along world +y; box centre sits 0.1 along it, i.e. at
    # world (1, -0.1, R); its +x face at (1, -0.08, R); ball surface at (1, -R, R)
    assert gap == pytest.approx(0.2 - 0.1 - 0.02 - R, abs=1e-6)


def test_foot_geoms_from_rig(tmp_path):
    mjcf = tmp_path / "rig.xml"
    names = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"] + \
            [f"B{i}" for i in range(9, 52)]
    names[17], names[36] = "L_Wrist", "R_Wrist"
    def geom(n):
        if n in ("L_Ankle", "R_Ankle"):
            return '<geom type="box" pos="0.05 0.01 -0.02" size="0.09 0.05 0.02" quat="1 0 0 0"/>'
        if n in ("L_Toe", "R_Toe"):
            return '<geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.02"/>'
        return ""
    body = "".join(f'<body name="{n}" pos="0 0 0">{geom(n)}' for n in names)
    mjcf.write_text(f'<mujoco><worldbody>{body}{"</body>" * len(names)}</worldbody></mujoco>')
    g = soc.foot_geoms(str(mjcf))
    assert set(g) == set(soc.FOOT_BODY_NAMES)
    assert g["L_Ankle"][0]["type"] == "box" and g["L_Ankle"][0]["size"].tolist() == [0.09, 0.05, 0.02]
    cap = g["L_Toe"][0]
    assert cap["type"] == "capsule" and cap["size"].tolist() == pytest.approx([0.02, 0.02])
    assert cap["pos"].tolist() == pytest.approx([0.02, 0, 0])
    assert (cap["R"] @ np.array([0, 0, 1.0])).tolist() == pytest.approx([1, 0, 0], abs=1e-9)


def test_kick_impulse_classifies_kick_miss_and_none():
    """Ball velocity jump = kick; the foot gap at that frame decides OK vs MISS;
    a floor bounce is not a kick; no jump = no kick."""
    T = 12
    # ball rolls slowly then is struck at frame 6 (velocity jumps by 6 m/s)
    def rolling_then_kick(kick_at=6):
        obj = np.zeros((T, 3)); obj[:, 2] = R
        x = 0.0
        for f in range(T):
            obj[f, 0] = x
            x += 0.01 if f < kick_at else 0.21           # 0.3 -> 6.3 m/s at 30 fps
        return obj
    t = make_clip(T=T)
    obj = rolling_then_kick()
    t[:, soc.I_OBJP] = torch.tensor(obj, dtype=torch.float32)
    # foot ON the ball at the kick frame
    bp = t[:, soc.I_BODY].view(T, 52, 3).clone()
    bp[6, FEET[1]] = torch.tensor(obj[6], dtype=torch.float32) + torch.tensor([R - 0.01, 0, 0])
    t[:, soc.I_BODY] = bp.view(T, -1)
    st = soc.census(t, R, 0.02, FEET, fixture_geoms())
    k = st["kick"]
    assert k["verdict"] == "kick" and k["frame"] == 6 and k["jump"] == pytest.approx(6.0, abs=0.05)
    assert k["gap_at"] < 0.02                                  # KICK-OK
    # same impulse, foot 3 m away -> KICK-MISS
    t2 = make_clip(T=T); t2[:, soc.I_OBJP] = torch.tensor(obj, dtype=torch.float32)
    k2 = soc.census(t2, R, 0.02, FEET, fixture_geoms())["kick"]
    assert k2["verdict"] == "kick" and k2["gap_at"] > 1.0
    # a floor bounce (vertical velocity flips at floor height) is not a kick
    t3 = make_clip(T=T)
    obj3 = np.zeros((T, 3)); obj3[:, 2] = R + np.abs(np.linspace(-0.3, 0.3, T)) + 0.0
    obj3[:, 2] = np.where(np.arange(T) < 6, R + 0.3 - 0.06 * np.arange(T), R + 0.06 * (np.arange(T) - 5))
    t3[:, soc.I_OBJP] = torch.tensor(obj3, dtype=torch.float32)
    assert soc.census(t3, R, 0.02, FEET, fixture_geoms())["kick"]["verdict"] == "no-kick"
    # a ball that just rolls: no kick
    t4 = make_clip(T=T); obj4 = np.zeros((T, 3)); obj4[:, 0] = np.arange(T) * 0.02; obj4[:, 2] = R
    t4[:, soc.I_OBJP] = torch.tensor(obj4, dtype=torch.float32)
    assert soc.census(t4, R, 0.02, FEET, fixture_geoms())["kick"]["verdict"] == "no-kick"
    # a tracking GLITCH (5 m teleport at frame 9) must not become the kick: the
    # real 6 m/s kick at frame 6 with the foot on the ball still wins
    t5 = make_clip(T=T)
    obj5 = rolling_then_kick(); obj5[9:, 1] += 5.0
    t5[:, soc.I_OBJP] = torch.tensor(obj5, dtype=torch.float32)
    bp = t5[:, soc.I_BODY].view(T, 52, 3).clone()
    bp[6, FEET[1]] = torch.tensor(obj5[6], dtype=torch.float32) + torch.tensor([R - 0.01, 0, 0])
    t5[:, soc.I_BODY] = bp.view(T, -1)
    k5 = soc.census(t5, R, 0.02, FEET, fixture_geoms())["kick"]
    assert k5["verdict"] == "kick" and k5["frame"] == 6 and k5["gap_at"] < 0.02
    assert k5["glitch"] == pytest.approx(5.0, abs=0.3)
    # a glitch with NO real kick is not a kick either
    t6 = make_clip(T=T); obj6 = np.zeros((T, 3)); obj6[:, 2] = R; obj6[9:, 1] += 5.0
    t6[:, soc.I_OBJP] = torch.tensor(obj6, dtype=torch.float32)
    k6 = soc.census(t6, R, 0.02, FEET, fixture_geoms())["kick"]
    assert k6["verdict"] == "no-kick" and k6["glitch"] > 4.0


def test_guard_and_threshold_required(tmp_path):
    """Refuses a clip with no foot contact; refuses to write without --threshold."""
    src = tmp_path / "src"; src.mkdir()
    torch.save(make_clip(touch_frames=(), claim_frames=(2,)), src / "sub405_ballx_000.pt")
    mjcf = tmp_path / "rig.xml"
    names = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"] + \
            [f"B{i}" for i in range(9, 52)]
    names[17], names[36] = "L_Wrist", "R_Wrist"
    def geom(n):
        return '<geom type="box" pos="0 0 0" size="0.02 0.02 0.02"/>' if n in soc.FOOT_BODY_NAMES else ""
    body = "".join(f'<body name="{n}" pos="0 0 0"><joint name="{n}_j" type="hinge" axis="1 0 0"/>{geom(n)}' for n in names)
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

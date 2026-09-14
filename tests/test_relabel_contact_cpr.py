#!/usr/bin/env python3
"""relabel_contact_cpr.py: -1 -> 0 on the six lower-leg bodies, nothing else.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_relabel_contact_cpr.py -v
"""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
cpr = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("relabel_contact_cpr", os.path.join(REPO, "scripts/relabel_contact_cpr.py")))
cpr.__spec__.loader.exec_module(cpr)

MJCF = os.path.join(REPO, "isaacgym/src/intermimic/data/assets/smplx/omomo.xml")
LEGS = [2, 3, 4, 6, 7, 8]        # L_Knee L_Ankle L_Toe R_Knee R_Ankle R_Toe in omomo.xml order


def make_clip(T=10):
    """Random-ish 591-channel tensor with a known contact_human layout:
    legs -1 on even frames, +1 on frame 1 for L_Knee, 0 elsewhere; a hand (17)
    +1 on odd frames and -1 on frame 0; body 20 -1 everywhere; contact_obj = 1
    on frames 0-4."""
    torch.manual_seed(0)
    t = torch.randn(T, 591)
    ch = torch.zeros(T, 52)
    ch[0::2][:, LEGS] = -1.0
    ch[1, 2] = 1.0
    ch[1::2, 17] = 1.0
    ch[0, 17] = -1.0
    ch[:, 20] = -1.0
    t[:, cpr.I_CONTACT_HUMAN] = ch
    t[:, cpr.I_CONTACT_OBJ] = torch.tensor([1.0] * 5 + [0.0] * (T - 5))
    return t


def test_lower_leg_ids_resolve_by_name_from_the_real_rig():
    assert cpr.lower_leg_body_ids(MJCF) == LEGS
    assert not set(LEGS) & set(cpr.HAND_BODY_IDS)


def test_neutralise_touches_only_minus_one_on_the_legs():
    t = make_clip()
    out, n = cpr.neutralise(t, LEGS)
    ch_in, ch_out = t[:, cpr.I_CONTACT_HUMAN], out[:, cpr.I_CONTACT_HUMAN]
    # every -1 on a leg became 0; the +1 on L_Knee survived
    assert (ch_out[:, LEGS] >= 0).all()
    assert ch_out[1, 2] == 1.0
    assert n == [5] * 6                                      # even frames of a 10-frame clip
    # hands, body 20, contact_obj and every non-contact channel untouched
    others = [i for i in range(52) if i not in LEGS]
    assert torch.equal(ch_out[:, others], ch_in[:, others])
    assert ch_out[0, 17] == -1.0 and (ch_out[:, 20] == -1.0).all()
    mask = torch.ones(591, dtype=torch.bool); mask[cpr.I_CONTACT_HUMAN] = False
    assert torch.equal(out[:, mask], t[:, mask])


def test_cli_writes_relabelled_clips_and_census_writes_nothing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    t = make_clip()
    torch.save(t, src / "sub509_cprd87s109t002a_000.pt")
    torch.save(t.clone(), src / "sub509_cprd87s109t002a_001.pt")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "scripts"))
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/relabel_contact_cpr.py"),
                        "--src-dir", str(src), "--mjcf", MJCF, "--census"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert not dst.exists() and "TOTAL" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/relabel_contact_cpr.py"),
                        "--src-dir", str(src), "--dst-dir", str(dst), "--mjcf", MJCF],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    got = torch.load(dst / "sub509_cprd87s109t002a_000.pt", weights_only=False)
    exp, _ = cpr.neutralise(t, LEGS)
    assert torch.equal(got, exp)
    assert sorted(p.name for p in dst.iterdir()) == sorted(p.name for p in src.iterdir())


def test_wrong_rig_is_fatal(tmp_path):
    bad = tmp_path / "rig.xml"
    bad.write_text('<mujoco><worldbody><body name="Pelvis"><body name="L_Hip"/></body></worldbody></mujoco>')
    with pytest.raises(SystemExit):
        cpr.lower_leg_body_ids(str(bad))

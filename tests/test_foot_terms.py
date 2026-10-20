"""rewardTerms.footWeight and rewardTerms.footSwing (utils/foot_terms.py + the wiring in
intermimic.py): the feet count `weight` times in the position / rotation terms; a toe
below the reference toe on a reference swing frame is penalised.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_foot_terms.py -q
"""
import math
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "isaacgym", "src"))
from intermimic.utils import foot_terms as ft           # noqa: E402  (torch only)

TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")


# --- footSwing -------------------------------------------------------------------
def test_swing_penalises_only_a_short_toe_on_a_swing_frame():
    sim = torch.tensor([[0.05, 0.05]])
    ref = torch.tensor([[0.10, 0.10]])                      # reference toes 5 cm higher
    stance = torch.tensor([[0.0, 0.0]])                     # labels: not swing
    swing = torch.tensor([[-1.0, -1.0]])                    # labels: must not contact = swing
    assert ft.swing_lift_error(sim, ref, stance).item() == 0.0
    assert ft.swing_lift_error(sim, ref, swing).item() == pytest.approx(0.05 ** 2)
    assert ft.swing_factor(sim, ref, swing, 50.0).item() == pytest.approx(math.exp(-50 * 0.0025))   # 0.88
    # a toe AT or ABOVE the reference costs nothing even on a swing frame
    assert ft.swing_factor(ref, ref, swing, 50.0).item() == 1.0
    assert ft.swing_factor(ref + 0.1, ref, swing, 50.0).item() == 1.0


def test_swing_is_per_toe_and_per_env():
    sim = torch.tensor([[0.05, 0.10], [0.0, 0.0]])
    ref = torch.tensor([[0.10, 0.10], [0.10, 0.10]])
    lab = torch.tensor([[-1.0, -1.0], [-1.0, 0.0]])           # env 1: only the left toe is swing
    err = ft.swing_lift_error(sim, ref, lab)
    assert err[0].item() == pytest.approx(0.05 ** 2 / 2)      # one short toe of two
    assert err[1].item() == pytest.approx(0.10 ** 2 / 2)      # right toe short but in stance: free
    f = ft.swing_factor(sim, ref, lab, 50.0)
    assert f.shape == (2,) and torch.all(f > 0) and torch.all(f <= 1)


def test_docstring_numbers():
    swing = torch.tensor([[-1.0]])
    assert ft.swing_factor(torch.tensor([[0.0]]), torch.tensor([[0.05]]), swing, 50.0).item() == pytest.approx(0.8825, abs=1e-3)
    assert ft.swing_factor(torch.tensor([[0.0]]), torch.tensor([[0.10]]), swing, 50.0).item() == pytest.approx(0.6065, abs=1e-3)


# --- the cfg blocks --------------------------------------------------------------
def test_parsers():
    assert ft.parse_foot_weight(None) == {'enable': False, 'weight': 5.0}
    assert ft.parse_foot_weight({'enable': True, 'weight': 3}) == {'enable': True, 'weight': 3.0}
    with pytest.raises(ValueError, match="unknown key"):
        ft.parse_foot_weight({'enable': True, 'lambda': 2})
    with pytest.raises(ValueError, match="> 0"):
        ft.parse_foot_weight({'enable': True, 'weight': 0})
    assert ft.parse_foot_swing(None) == {'enable': False, 'lambda': 50.0}
    assert ft.parse_foot_swing({'enable': True, 'lambda': 20})['lambda'] == 20.0
    with pytest.raises(ValueError, match="unknown key"):
        ft.parse_foot_swing({'enable': True, 'weight': 2})
    with pytest.raises(ValueError, match="> 0"):
        ft.parse_foot_swing({'enable': True, 'lambda': -1})


# --- the wiring (read as text: the task cannot be imported off-cluster) -----------
def test_wiring():
    src = open(TASK).read()
    assert "'footSkate', 'footWeight', 'footSwing'" in src                      # validator admits both
    assert "foot_terms.parse_foot_weight(rt.get('footWeight'))" in src
    assert "foot_terms.parse_foot_swing(rt.get('footSwing'))" in src
    # N under the root counts footSwing (a factor) but NOT footWeight (a weighting inside existing terms)
    assert ".get('footSwing', {}) or {}).get('enable', False)) else 0)" in src
    assert ".get('footWeight', {}) or {}).get('enable', False)) else 0)" not in src
    # the feet in the 52-body order, built after the sim exists
    assert "self._foot_ids52 = self._build_key_body_ids_tensor(['L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe'])" in src
    assert "self._toe_ids = self._build_key_body_ids_tensor(['L_Toe', 'R_Toe'])" in src
    # footWeight multiplies BOTH the position and the rotation weights on those ids
    assert "weight_hp[:, self._foot_ids52] = weight_hp[:, self._foot_ids52] * self._foot_weight['weight']" in src
    assert "weight_hr[:, self._foot_ids52] = weight_hr[:, self._foot_ids52] * self._foot_weight['weight']" in src
    # footSwing reads world toe z from sim and reference obs and the reference's contact label
    assert "factors.append(foot_terms.swing_factor(sim_toe_z, ref_toe_z, ref_toe_label, self._foot_swing['lambda']))" in src
    assert "extract_data_component('contact_human', obs=self._curr_ref_obs)[:, self._toe_ids]" in src
    assert "rewardTerms.footWeight / footSwing need rewardShape" in src
    lines = src.splitlines()
    sup = next(i for i, l in enumerate(lines) if "super().__init__(cfg=cfg," in l)
    ids = next(i for i, l in enumerate(lines) if "self._foot_ids52 = self._build_key_body_ids_tensor" in l)
    assert sup < ids

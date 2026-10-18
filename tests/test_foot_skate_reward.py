"""rewardTerms.footSkate (utils/foot_skate_reward.py + its wiring in intermimic.py): a
planted foot pays for its horizontal speed; a foot in the air pays nothing.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_foot_skate_reward.py -q
"""
import math
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "isaacgym", "src"))
from intermimic.utils import foot_skate_reward as fs          # noqa: E402  (torch only)
from intermimic.utils import reward_shape                      # noqa: E402

TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
FEET = torch.tensor([4, 8])                                     # L_Toe, R_Toe in the 52-body order


def _scene(E=3, B=52):
    """Zero velocities, zero contact forces: nothing planted, nothing moving."""
    return torch.zeros(E, B, 3), torch.zeros(E, B, 3)


def test_still_feet_and_airborne_feet_cost_nothing():
    vel, force = _scene()
    force[:, FEET, 2] = 50.0                                     # both toes planted, not moving
    assert torch.allclose(fs.factor(vel, force, FEET, 2.0, 1.0), torch.ones(3))
    vel, force = _scene()
    vel[:, FEET, 0] = 1.5                                        # both toes flying forward, no contact
    assert torch.allclose(fs.factor(vel, force, FEET, 2.0, 1.0), torch.ones(3))


def test_a_planted_sliding_foot_pays_for_its_horizontal_speed_only():
    vel, force = _scene(E=1)
    force[0, FEET, 2] = 50.0                                     # both planted
    vel[0, 4, 0] = 0.5                                           # left toe slides 0.5 m/s
    vel[0, 4, 2] = 3.0                                           # vertical speed is NOT skating
    err = fs.skate_error(vel, force, FEET, 1.0)
    assert err.item() == pytest.approx((0.5 ** 2 + 0) / 2)       # mean over the two feet
    assert fs.factor(vel, force, FEET, 2.0, 1.0).item() == pytest.approx(math.exp(-2.0 * 0.125))
    # the docstring's numbers: one foot at 0.5 m/s with lambda 2 on its own = exp(-0.5)
    one = torch.tensor([4])
    assert fs.factor(vel, force, one, 2.0, 1.0).item() == pytest.approx(math.exp(-0.5))
    vel[0, 4, 0] = 1.0
    assert fs.factor(vel, force, one, 2.0, 1.0).item() == pytest.approx(math.exp(-2.0), abs=1e-6)


def test_contact_threshold_decides_planted():
    vel, force = _scene(E=1)
    vel[0, FEET, 1] = 1.0
    force[0, FEET, 2] = 0.5                                      # below the 1 N threshold: airborne
    assert fs.factor(vel, force, FEET, 2.0, 1.0).item() == 1.0
    force[0, FEET, 2] = 1.5                                      # above: planted and sliding
    assert fs.factor(vel, force, FEET, 2.0, 1.0).item() < 0.2


def test_factor_is_per_env_and_in_unit_interval():
    vel, force = _scene(E=4)
    force[:, FEET, 2] = 10.0
    vel[:, FEET, 0] = torch.tensor([0.0, 0.2, 1.0, 5.0]).view(4, 1)
    r = fs.factor(vel, force, FEET, 2.0, 1.0)
    assert r.shape == (4,) and r[0] == 1.0 and torch.all(r[1:] < r[:-1]) and torch.all(r > 0)


def test_cfg_parser():
    assert fs.parse_cfg(None) == {'enable': False, 'lambda': 2.0, 'bodies': ['L_Toe', 'R_Toe'], 'contactThreshold': 1.0}
    c = fs.parse_cfg({'enable': True, 'lambda': 3.0, 'bodies': ['L_Toe', 'R_Toe', 'L_Ankle', 'R_Ankle']})
    assert c['enable'] and c['lambda'] == 3.0 and len(c['bodies']) == 4
    with pytest.raises(ValueError, match="unknown key"):
        fs.parse_cfg({'enable': True, 'weight': 2.0})
    with pytest.raises(ValueError, match="> 0"):
        fs.parse_cfg({'enable': True, 'lambda': 0})
    with pytest.raises(ValueError, match="non-empty"):
        fs.parse_cfg({'enable': True, 'bodies': []})
    assert fs.parse_cfg({'enable': False, 'lambda': 0})['enable'] is False      # off: values not checked


def test_combines_as_a_fifth_factor_under_the_root():
    rb, ro, rig, rcg = (torch.tensor([0.35]), torch.tensor([0.73]), torch.tensor([0.79]), torch.tensor([0.58]))
    foot = torch.tensor([0.5])
    r5 = reward_shape.combine([rb, ro, rig, rcg, foot], 'geometric_all')
    assert r5.item() == pytest.approx((0.35 * 0.73 * 0.79 * 0.58 * 0.5) ** 0.2)
    # with a per-env exponent (the object-flight rule at product): plain product of five
    r5p = reward_shape.combine([rb, ro, rig, rcg, foot], 'geometric_all', exponent=torch.tensor([1.0]))
    assert r5p.item() == pytest.approx(0.35 * 0.73 * 0.79 * 0.58 * 0.5)


# --- the wiring (the task cannot be imported off-cluster: read as text) ---------------
def test_wiring():
    src = open(TASK).read()
    assert "'pose', 'freeFlightGate', 'footSkate'" in src                       # validator admits the block
    assert "foot_skate_reward.parse_cfg(rt.get('footSkate'))" in src              # ... and checks its keys
    assert "self._foot_skate_ids = self._build_key_body_ids_tensor(self._foot_skate['bodies'])" in src
    assert "rewardTerms.footSkate.enable needs rewardShape" in src
    # N under the root counts it, both where the exponent rules read N at startup and at reward time
    assert ".get('footSkate', {}) or {}).get('enable', False)) else 0)" in src
    assert "n_factors = len(factors) + (1 if pose_factor is not None else 0)" in src
    assert "factors.append(foot_skate_reward.factor(self._rigid_body_vel, self._contact_forces," in src
    assert "reward = reward_shape.combine(factors, self._reward_shape," in src
    # the foot ids are built after the sim exists (after super().__init__) -- the parse site follows it
    lines = src.splitlines()
    sup = next(i for i, l in enumerate(lines) if "super().__init__(cfg=cfg," in l)
    ids = next(i for i, l in enumerate(lines) if "self._foot_skate_ids = self._build_key_body_ids_tensor" in l)
    assert sup < ids

"""objectFlightReward (utils/object_flight.py + its wiring in intermimic.py): the per-clip
exponent from the clip's object flight.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_object_flight.py -q
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "isaacgym", "src"))
from intermimic.utils import object_flight as of          # noqa: E402  (torch only)

spec = importlib.util.spec_from_file_location("cof", os.path.join(REPO, "scripts", "clip_object_flight_stats.py"))
cof = importlib.util.module_from_spec(spec); spec.loader.exec_module(cof)
TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
REPO_CLIP = os.path.join(REPO, "InterAct", "OMOMO_new", "sub2_largetable_000.pt")


# --- the exponent ------------------------------------------------------------------
def test_exponent_product_below_flat_root_at_and_beyond_root_linear_between():
    e = lambda f: of.exponent(f, 0.6, 0.0, 4)
    assert e(0.0) == 1.0 and e(0.6) == pytest.approx(0.25) and e(1.8) == pytest.approx(0.25)
    assert e(0.3) == pytest.approx(0.625)                               # halfway
    # the two-constant form: flat to 0.3, root at 0.6
    e2 = lambda f: of.exponent(f, 0.6, 0.3, 4)
    assert e2(0.0) == 1.0 and e2(0.3) == 1.0 and e2(0.45) == pytest.approx(0.625) and e2(0.6) == pytest.approx(0.25)
    # N under the root: 5 factors with the pose term -> floor 0.2
    assert of.exponent(9.0, 0.6, 0.0, 5) == pytest.approx(0.2)


def test_exponent_tensor_path_matches_scalar_path():
    f = torch.tensor([0.0, 0.1, 0.3, 0.45, 0.6, 1.2])
    t = of.exponent(f, 0.6, 0.0, 4)
    assert t.shape == f.shape
    for x, y in zip(f.tolist(), t.tolist()):
        assert y == pytest.approx(of.exponent(x, 0.6, 0.0, 4))
    t2 = of.exponent(f, 0.6, 0.3, 4)
    assert t2[:3].tolist() == [1.0, 1.0, 1.0]                           # the flat region


def test_exponent_at_the_measured_dataset_points():
    """The 2026-10-17 measurement (scripts/clip_object_flight_stats.py on OMOMO_new,
    soccer15, bball7) and what each rule gives it -- the numbers the decision was made on."""
    one = lambda f: of.exponent(f, 0.6, 0.0, 4)
    two = lambda f: of.exponent(f, 0.6, 0.3, 4)
    assert one(0.0) == 1.0 and two(0.0) == 1.0                          # OMOMO median / p90 (and soccer p10)
    assert one(0.19) == pytest.approx(0.7625, abs=0.01) and two(0.19) == 1.0   # OMOMO max (the tripod tail)
    assert one(0.088) == pytest.approx(0.89, abs=0.01) and two(0.088) == 1.0   # soccer median
    assert one(0.318) == pytest.approx(0.60, abs=0.01) and two(0.318) == pytest.approx(0.955, abs=0.01)  # soccer p90
    assert one(0.505) == pytest.approx(0.37, abs=0.01) and two(0.505) == pytest.approx(0.49, abs=0.01)   # soccer max
    assert one(0.62) == 0.25 and two(0.62) == 0.25                      # basketball p10
    assert one(0.90) == 0.25 and two(0.90) == 0.25                      # basketball median


def test_exponent_refuses_bad_constants():
    with pytest.raises(ValueError):
        of.exponent(0.1, 0.3, 0.3, 4)          # root must exceed flat
    with pytest.raises(ValueError):
        of.exponent(0.1, 0.6, 0.0, 0)


# --- the cfg block -----------------------------------------------------------------
def test_cfg_parser():
    assert of.parse_cfg(None) == {'enable': False, 'rootFlight': None, 'flatFlight': 0.0}
    assert of.parse_cfg({'enable': True, 'rootFlight': 0.6}) == {'enable': True, 'rootFlight': 0.6, 'flatFlight': 0.0}
    assert of.parse_cfg({'enable': True, 'rootFlight': 0.6, 'flatFlight': 0.3})['flatFlight'] == 0.3
    with pytest.raises(ValueError, match="REQUIRED"):
        of.parse_cfg({'enable': True})
    with pytest.raises(ValueError, match="must be >"):
        of.parse_cfg({'enable': True, 'rootFlight': 0.3, 'flatFlight': 0.3})
    with pytest.raises(ValueError, match="unknown key"):
        of.parse_cfg({'enable': True, 'rootFlight': 0.6, 'referenceStd': 0.5})
    assert of.parse_cfg({'enable': False})['enable'] is False


# --- the per-clip flight -----------------------------------------------------------
def _clip(T, z, contact):
    x = torch.zeros(T, 591)
    x[:, of.I_OBJ_Z] = torch.as_tensor(z, dtype=torch.float32)
    x[:, of.I_CONTACT_OBJ] = torch.as_tensor(contact, dtype=torch.float32)
    return x


def test_clip_flight_on_known_object_behaviour():
    assert of.clip_flight(_clip(120, np.full(120, 1.2), np.ones(120))) == 0.0        # carried: contact on
    assert of.clip_flight(_clip(90, np.full(90, 0.3), np.zeros(90))) == 0.0          # resting, nobody touching
    z = 0.11 + 0.01 * np.sin(np.linspace(0, 20, 300))                                  # rolling on the ground
    assert of.clip_flight(_clip(300, z, (np.arange(300) % 10 == 0))) == 0.0
    phase = (np.arange(300) % 20) / 20.0                                                # dribble: floor <-> hand
    zb = 0.12 + 0.88 * np.sin(np.pi * phase)
    fl = of.clip_flight(_clip(300, zb, np.abs(phase - 0.5) < 0.06))
    assert fl > 0.3
    assert of.clip_flight(_clip(1, [0.5], [0.0])) == 0.0                                # a one-frame clip


def test_clip_flight_equals_the_measurement_script():
    """The task grades with the same number the script reports, on synthetic clips
    and on a real OMOMO clip from the repo."""
    phase = (np.arange(300) % 20) / 20.0
    zb = 0.12 + 0.88 * np.sin(np.pi * phase)
    x = _clip(300, zb, np.abs(phase - 0.5) < 0.06)
    assert of.clip_flight(x) == pytest.approx(cof.flight_stats(x.numpy().astype(np.float64))["flight_m"], abs=1e-6)
    assert os.path.exists(REPO_CLIP), REPO_CLIP
    real = torch.load(REPO_CLIP, map_location="cpu", weights_only=False)
    assert of.clip_flight(real) == pytest.approx(cof.flight_stats(cof.load_clip(REPO_CLIP))["flight_m"], abs=1e-6)


# --- the wiring in the task (cannot be imported off-cluster: read as text) ------------
def _line_of(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    raise AssertionError(f"not found in intermimic.py: {needle!r}")


def test_block_is_parsed_before_the_motion_load_that_reads_it():
    lines = open(TASK).read().splitlines()
    parse = _line_of(lines, "self._object_flight = object_flight.parse_cfg(")
    nfac = _line_of(lines, "self._motion_scale_n_factors = ")
    load = _line_of(lines, "self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)")
    assert parse < load and nfac < load, (parse, nfac, load)


def test_the_two_rules_are_exclusive_and_the_key_is_known():
    src = open(TASK).read()
    assert "objectFlightReward and motionScaleReward are both enabled" in src
    assert "'objectFlightReward'" in src                                 # the env-key allowlist


def test_load_path_computes_flight_from_raw_channels_and_the_reward_reads_it():
    src = open(TASK).read()
    assert "motion_flights.append(object_flight.clip_flight(loaded_dict['hoi_data']))" in src
    assert "self._motion_flight = to_torch(motion_flights" in src
    assert "object_flight.exponent(self._motion_flight[self.data_id]" in src
    a = src.index("    def _print_object_flight_table(self, motion_file, motion_flights):")
    b = src.index("    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):")
    code = "\n".join(l for l in src[a:b].splitlines() if not l.strip().startswith("#"))
    assert "self._pose_term_enable" not in code and "self._reward_shape" not in code
    assert "self._motion_scale_n_factors" in code

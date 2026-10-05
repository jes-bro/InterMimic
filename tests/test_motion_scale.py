#!/usr/bin/env python3
"""Tests for utils/motion_scale.py and the per-env exponent in reward_shape.combine.

Pins the mechanism's load-bearing properties:
  * the clip std is the pooled temporal std of the key bodies' positions;
  * exponent = clamp((ref / std)^2, 1/N, 1): a clip moving the reference amount
    OR LESS is graded as PRODUCT, one moving twice that or more gets today's 1/N
    root, and nothing is ever stricter than product (capped, no floor);
  * combine(..., exponent=1/N) equals today's geometric_all bit for bit and
    exponent=1 equals the plain product, so existing runs are unaffected and
    the shape stays a monotone transform (AND gate preserved);
  * the cfg parser refuses missing constants and unknown keys, and the
    exponent is refused under any shape but geometric_all.

Loaded by file path (the reward_shape test's pattern) because the package's
__init__ chain imports isaacgym, which is not available off-cluster.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_motion_scale.py -q
"""
import importlib.util
import math
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "isaacgym/src/intermimic/utils", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ms = _load("motion_scale")
rs = _load("reward_shape")

KEY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36]
REF = ms.OMOMO_MEDIAN_KEY_BODY_STD_M


def synthetic_clip(std, T=60, seed=0):
    """(T, 52, 3) reference whose KEY bodies oscillate with exactly the given pooled
    std around a fixed pose; the 31 finger bodies stay put."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(52, 3, generator=g)
    direction = torch.randn(52, 3, generator=g)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    wave = torch.sin(2 * math.pi * torch.arange(T, dtype=torch.float) / 20)
    wave = wave / wave.pow(2).mean().sqrt()                     # rms exactly 1
    clip = base.unsqueeze(0) + wave.view(T, 1, 1) * direction.unsqueeze(0) * std
    non_key = [i for i in range(52) if i not in KEY]
    clip[:, non_key] = base[non_key]
    return clip


def test_key_body_std_is_pooled_temporal_std():
    for s in (0.05, 0.5157, 1.0):
        clip = synthetic_clip(s)
        assert abs(ms.key_body_std(clip, KEY) - s) < 1e-4
        assert abs(ms.key_body_std(clip.reshape(clip.shape[0], -1), KEY) - s) < 1e-4   # flat (T, 156) layout
    assert ms.key_body_std(synthetic_clip(0.3)[:1], KEY) == 0.0                         # one frame: no motion
    # moving only non-key bodies is not motion the reward grades -> std 0
    clip = synthetic_clip(0.0)
    clip[:, 18] += torch.linspace(0, 1, clip.shape[0]).view(-1, 1)                   # a finger wanders
    assert ms.key_body_std(clip, KEY) < 1e-6                                         # float32 noise only


def test_exponent_product_at_and_below_reference_root_at_twice():
    N = 4
    assert ms.exponent(REF, REF, N) == pytest.approx(1.0)                # reference motion -> product
    assert ms.exponent(REF / 2, REF, N) == 1.0                           # smaller motion: STILL product (capped)
    assert ms.exponent(0.10, REF, N) == 1.0 and ms.exponent(0.02, REF, N) == 1.0   # CPR-sized, jitter-sized
    assert ms.exponent(0.0, REF, N) == 1.0                               # no motion at all: the cap, no divide
    assert ms.exponent(2 * REF, REF, N) == pytest.approx(0.25)           # twice the motion -> today's 4th root
    assert ms.exponent(10 * REF, REF, N) == pytest.approx(0.25)          # never softer than the root
    assert ms.exponent(REF * 2 ** 0.5, REF, N) == pytest.approx(0.5)     # in between
    assert ms.exponent(3 * REF, REF, 5) == pytest.approx(0.2)            # 5 factors: floor is 1/5
    assert ms.exponent(2 * REF, REF, 5) == pytest.approx(0.25)
    # measured datasets (2026-09-30): bball median 1.02 -> just above the root; soccer 0.45, OMOMO small -> product
    assert ms.exponent(1.02, REF, N) == pytest.approx((REF / 1.02) ** 2)
    assert ms.exponent(0.45, REF, N) == 1.0 and ms.exponent(0.18, REF, N) == 1.0
    assert ms.MAX_EXPONENT == 1.0
    with pytest.raises(ValueError):
        ms.exponent(0.5, REF, 0)


def test_exponent_never_exceeds_product_or_drops_below_root():
    stds = torch.tensor([0.0, 0.02, 0.10, 0.18, 0.5157, 0.73, 1.02, 3.0])
    for n in (4, 5):
        e = ms.exponent(stds, REF, n)
        assert float(e.max()) <= 1.0 and float(e.min()) >= 1.0 / n - 1e-7
        assert bool((e[1:] <= e[:-1] + 1e-7).all())                      # more motion never means a stricter exponent


def test_exponent_tensor_path_matches_scalar_path():
    stds = torch.tensor([0.02, 0.10, 0.18, 0.5157, 0.73, 1.02, 3.0])
    t = ms.exponent(stds, REF, 4)
    for s_, e in zip(stds.tolist(), t.tolist()):
        assert e == pytest.approx(ms.exponent(s_, REF, 4), rel=1e-6)


def test_combine_exponent_reproduces_root_and_product_exactly():
    g = torch.Generator().manual_seed(1)
    f = [torch.rand(64, generator=g) * 0.9 + 0.05 for _ in range(4)]
    today = rs.combine(f, "geometric_all")
    root = rs.combine(f, "geometric_all", exponent=torch.full((64,), 0.25))
    assert torch.equal(today, root)                                           # bit-identical, not approx
    prod = rs.combine(f, "product")
    e1 = rs.combine(f, "geometric_all", exponent=torch.ones(64))
    assert torch.allclose(prod, e1, atol=1e-7)                                # product (clamp floor aside)
    # per-env exponents: strict clips score lower than lenient ones for the same factors
    mixed = rs.combine(f, "geometric_all", exponent=torch.tensor([0.25, 1.0, 8.0] + [1.0] * 61))
    assert mixed[0] > mixed[1] > mixed[2]
    # monotone in every factor at every exponent (AND gate preserved)
    f2 = [x.clone() for x in f]; f2[3] = f2[3] * 0.5
    assert bool((rs.combine(f2, "geometric_all", exponent=torch.full((64,), 8.0))
                 <= rs.combine(f, "geometric_all", exponent=torch.full((64,), 8.0))).all())


def test_combine_refuses_exponent_under_other_shapes_and_folds_pose():
    f = [torch.full((3,), 0.5) for _ in range(4)]
    for shape in ("product", "geometric"):
        with pytest.raises(ValueError, match="geometric_all"):
            rs.combine(f, shape, exponent=torch.ones(3))
    pose = torch.full((3,), 0.5)
    r = rs.combine(f, "geometric_all", pose=pose, exponent=torch.ones(3))     # 5 factors, exponent 1
    assert torch.allclose(r, torch.full((3,), 0.5 ** 5))


def test_cfg_parser():
    assert ms.parse_cfg(None) == {'enable': False, 'referenceStd': None}
    assert ms.parse_cfg({'enable': True, 'referenceStd': 0.5157}) == {'enable': True, 'referenceStd': 0.5157}
    with pytest.raises(ValueError, match="REQUIRED"):
        ms.parse_cfg({'enable': True})                                        # no reference
    with pytest.raises(ValueError, match="> 0"):
        ms.parse_cfg({'enable': True, 'referenceStd': 0})
    with pytest.raises(ValueError, match="unknown key"):
        ms.parse_cfg({'enable': True, 'referenceStd': 0.5157, 'floorStd': 0.18})   # the floor is gone
    with pytest.raises(ValueError, match="unknown key"):
        ms.parse_cfg({'enable': True, 'referenceStd': 0.5157, 'A_ref': 1.0})
    assert ms.parse_cfg({'enable': False})['enable'] is False                 # off: the constant is optional


# --- startup ORDER in the task (it cannot be imported off-cluster, so read as text) ---
# The first draft parsed motionScaleReward next to rewardShape, AFTER the motion load
# that needs it: every run would have died with an AttributeError at load. These pin it.

TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")


def _line_of(src_lines, needle, start=0):
    for i in range(start, len(src_lines)):
        if needle in src_lines[i]:
            return i
    raise AssertionError(f"not found in intermimic.py: {needle!r}")


def test_block_is_parsed_before_the_motion_load_that_reads_it():
    lines = open(TASK).read().splitlines()
    parse = _line_of(lines, "self._motion_scale = motion_scale.parse_cfg(")
    nfac = _line_of(lines, "self._motion_scale_n_factors = ")
    load = _line_of(lines, "self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)")
    assert parse < load and nfac < load, (parse, nfac, load)


def test_load_path_reads_nothing_that_is_set_after_the_load():
    src = open(TASK).read()
    a = src.index("    def _print_motion_scale_table(self, motion_file, motion_stds):")
    b = src.index("    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):")
    table = src[a:b]
    code = "\n".join(l for l in table.splitlines() if not l.strip().startswith("#"))
    assert "self._pose_term_enable" not in code          # parsed long after the load
    assert "self._reward_shape" not in code
    assert "self._motion_scale_n_factors" in code

#!/usr/bin/env python3
"""Tests for isaacgym/src/intermimic/utils/reward_shape.py.

The load-bearing property is BACKWARD COMPATIBILITY: 'product' and 'geometric'
must compute exactly what they computed before this module existed, because
r7, r8, r10 and r11-r14 were run and are judged on those numbers. Only the new
'geometric_all' behaves differently, and only by moving the pose factor inside
the root.

Loaded by file path (the psi_update test's pattern) because the package's
__init__ chain imports isaacgym, which is not available off-cluster.

Run:  python tests/test_reward_shape.py   (exit 0 = all green)
  or: pytest tests/test_reward_shape.py
"""
import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "reward_shape",
    os.path.join(REPO, "isaacgym/src/intermimic/utils/reward_shape.py"))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


# The bball arm's held-frame values, from the rationale in the source.
RB, RO, RIG, RCG = (torch.tensor([0.304]), torch.tensor([0.402]),
                    torch.tensor([0.422]), torch.tensor([0.240]))
BASE = [RB, RO, RIG, RCG]
POSE = torch.tensor([0.7])


def close(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol)


def test_product_matches_the_original_expression():
    # exactly `reward = rb * ro * rig * rcg`, unclamped
    assert close(rs.combine(BASE, 'product'), RB * RO * RIG * RCG)
    assert close(rs.combine(BASE, 'product', pose=POSE), RB * RO * RIG * RCG * POSE)
    print("ok: 'product' is the original unclamped product")


def test_geometric_matches_the_original_expression_including_pose_outside():
    """The behaviour r7..r14 ran with: 4th root of the four base terms, pose
    multiplied in AFTER. This is the row that must never change."""
    want = torch.stack(BASE, dim=0).clamp_min(1e-8).prod(dim=0) ** 0.25
    assert close(rs.combine(BASE, 'geometric'), want)
    assert close(rs.combine(BASE, 'geometric', pose=POSE), want * POSE)
    print("ok: 'geometric' keeps the pose factor outside the root, as run")


def test_geometric_all_puts_pose_inside_the_root():
    want = (torch.stack(BASE + [POSE], dim=0).clamp_min(1e-8).prod(dim=0)) ** (1.0 / 5)
    assert close(rs.combine(BASE, 'geometric_all', pose=POSE), want)
    print("ok: 'geometric_all' takes the 5th root over all factors")


def test_the_two_roots_differ_only_by_the_pose_weighting():
    """In log space, 'geometric' weights pose 4x the other terms and
    'geometric_all' weights everything equally. Pin that quantitatively."""
    g = rs.combine(BASE, 'geometric', pose=POSE)
    ga = rs.combine(BASE, 'geometric_all', pose=POSE)
    logs = [torch.log(f) for f in BASE]
    assert close(torch.log(g), 0.25 * sum(logs) + 1.0 * torch.log(POSE))
    assert close(torch.log(ga), 0.2 * (sum(logs) + torch.log(POSE)))
    assert not close(g, ga)
    print("ok: geometric weights pose 4x; geometric_all weights all terms equally")


def test_geometric_all_equals_geometric_when_pose_is_off():
    """The root counts the factors present rather than assuming four, so with
    the pose term disabled the new shape is a drop-in for the old one."""
    assert close(rs.combine(BASE, 'geometric_all'), rs.combine(BASE, 'geometric'))
    print("ok: with pose off, geometric_all == geometric")


def test_a_zero_factor_collapses_the_reward_under_every_shape():
    """The AND gate: a zero in any factor must destroy the reward.

    'product' is unclamped so it reaches exactly 0. The roots clamp at 1e-8 for
    gradient finiteness, so they FLOOR at (1e-8 * rest)**(1/N) rather than
    reaching zero -- ~0.003 for the 4-factor root, ~0.012 for the 5-factor one.
    Both are ~2 orders below the ~0.33 healthy value, so the gate holds in
    effect, but 'zero' is not literal under a root and the floor is higher the
    more factors the root spans."""
    for shape in rs.VALID_SHAPES:
        healthy = rs.combine(BASE, shape, pose=POSE)
        for i in range(len(BASE)):
            factors = list(BASE)
            factors[i] = torch.tensor([0.0])
            out = rs.combine(factors, shape, pose=POSE)
            assert out.item() < 0.05 * healthy.item(), (shape, i, out, healthy)
        out = rs.combine(BASE, shape, pose=torch.tensor([0.0]))
        assert out.item() < 0.05 * healthy.item(), (shape, "pose", out)

    # exact values, so a change to the clamp or the root order is visible
    assert rs.combine([torch.tensor([0.0])] + BASE[1:], 'product').item() == 0.0
    geo_floor = rs.combine([torch.tensor([0.0])] + BASE[1:], 'geometric').item()
    all_floor = rs.combine([torch.tensor([0.0])] + BASE[1:], 'geometric_all',
                           pose=POSE).item()
    assert 0.002 < geo_floor < 0.005, geo_floor
    assert all_floor > geo_floor, (all_floor, geo_floor)   # wider root, higher floor
    print(f"ok: a zero factor collapses the reward "
          f"(product 0.0, geometric {geo_floor:.4f}, geometric_all {all_floor:.4f})")


def test_roots_are_monotone_in_every_factor():
    """A root is a monotone transform, so improving any term must not lower the
    reward -- the optimum is unchanged."""
    for shape in rs.VALID_SHAPES:
        base = rs.combine(BASE, shape, pose=POSE)
        for i in range(len(BASE)):
            better = list(BASE)
            better[i] = BASE[i] + 0.1
            assert rs.combine(better, shape, pose=POSE) > base, (shape, i)
        assert rs.combine(BASE, shape, pose=POSE + 0.1) > base, shape
    print("ok: every shape is monotone increasing in every factor")


def test_the_root_lifts_a_collapsed_product():
    """The reason the shape exists: at the held-frame values the product is
    ~0.012 and the root ~0.33."""
    prod = rs.combine(BASE, 'product')
    geo = rs.combine(BASE, 'geometric')
    assert abs(prod.item() - 0.0124) < 5e-4, prod
    assert abs(geo.item() - 0.334) < 5e-3, geo
    print(f"ok: product {prod.item():.4f} -> geometric {geo.item():.3f}")


def test_batched_inputs_stay_elementwise():
    n = 7
    factors = [torch.rand(n) * 0.9 + 0.05 for _ in range(4)]
    pose = torch.rand(n) * 0.9 + 0.05
    for shape in rs.VALID_SHAPES:
        out = rs.combine(factors, shape, pose=pose)
        assert out.shape == (n,)
        # elementwise: row k must equal the scalar computation on row k
        k = 3
        one = rs.combine([f[k:k + 1] for f in factors], shape, pose=pose[k:k + 1])
        assert close(out[k:k + 1], one), shape
    print("ok: shapes are elementwise over a batch")


def test_unknown_shape_and_empty_factors_raise():
    for bad in ("geometric5", "GEOMETRIC", "mean", ""):
        try:
            rs.combine(BASE, bad)
        except ValueError as exc:
            assert "expected one of" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should have raised")
    try:
        rs.combine([], 'product')
    except ValueError as exc:
        assert "no factors" in str(exc)
    else:
        raise AssertionError("empty factors should have raised")
    print("ok: unknown shapes and empty factors raise, never fall back")


def test_task_validates_against_this_module():
    """The task must accept exactly the shapes this module defines -- a config
    naming a valid shape the task rejects (or vice versa) is a silent trap."""
    src = open(os.path.join(
        REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")).read()
    assert "reward_shape.VALID_SHAPES" in src, "task should validate against the module"
    assert "reward_shape.combine(" in src, "task should shape via the module"
    # the pose factor must be handed to combine(), not multiplied in afterwards
    assert "reward = reward * self._compute_pose_reward()" not in src
    assert "pose=pose_factor" in src
    print("ok: the task validates and shapes through this module")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")

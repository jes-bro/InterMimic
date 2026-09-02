#!/usr/bin/env python3
"""How the reward's factors are combined: product, or a root of the product.

InterMimic's reward is an AND gate -- a product of per-aspect factors, each in
(0, 1], so a policy that tracks the body and drops the object cannot score. The
shape decides how that product is presented to the optimizer:

  'product'        rb * ro * rig * rcg            (original)
  'geometric'      (rb*ro*rig*rcg) ** 1/4         (r7 onward)
  'geometric_all'  every enabled factor, including pose, inside the root

WHY A ROOT AT ALL. The product makes each term's gradient proportional to the
others: at the bball arm's held-frame values (rb .304 ro .402 rig .422 rcg .240)
the product is 0.012 and dR/d(rcg) = rb*ro*rig = 0.052, so improving contact
barely moves the reward BECAUSE everything else is bad -- nothing can be fixed
first. A root is a monotone transform, so the optimum and the AND property are
preserved, but the value lands at 0.334 and dR/d(rcg) = R/(N*rcg) = 0.35, ~7x
larger, and it no longer collapses when the other factors are weak.

ONE CAVEAT ON "AND". The clamp below keeps the root's gradient finite, which
means a zero factor FLOORS the reward rather than zeroing it: ~0.003 under the
4-factor root, ~0.012 under the 5-factor one, against ~0.33 healthy. Two orders
down, so the gate holds in practice -- but 'product' is the only shape where a
zero is literally zero, and the floor rises as the root spans more factors.

WHY 'geometric' AND 'geometric_all' BOTH EXIST. The opt-in pose factor
(rewardTerms.pose) was added 2026-06-29, when the reward was a plain product --
where a factor's position does not matter, multiplication being commutative.
'geometric' arrived 2026-08-27 and took the root of the four terms it was
written against, leaving the pose factor multiplied in afterwards, OUTSIDE the
root. In log space that gives it 4x the weight of every other term:

    log R = 0.25*(log rb + log ro + log rig + log rcg) + 1.0*log r_pose

That was not a decision -- it is what happens when a non-linear step is inserted
into a pipeline that used to be purely multiplicative. But r7, r8, r10 and
r11-r14 all ran that way and are judged on those numbers, so 'geometric' keeps
that behaviour EXACTLY and 'geometric_all' is the corrected form, chosen per
config. With the pose term disabled the two are identical, because the root
counts the factors actually present rather than assuming four.
"""
import torch

VALID_SHAPES = ('product', 'geometric', 'geometric_all')

# The terms are exp(-x) and so strictly positive, but a zero from underflow
# would make the root's gradient non-finite.
_FLOOR = 1e-8


def combine(factors, shape, pose=None):
    """Combine reward factors under `shape`.

    factors  the base per-aspect factors, in order [rb, ro, rig, rcg]
    shape    one of VALID_SHAPES
    pose     the optional pose factor, or None when rewardTerms.pose is off

    Returns the shaped reward. 'product' and 'geometric' apply `pose` OUTSIDE
    the combination, byte-identically to how they have always run;
    'geometric_all' folds it in and takes the root over all of them.
    """
    if shape not in VALID_SHAPES:
        raise ValueError(f"reward shape {shape!r}; expected one of {VALID_SHAPES}")
    if not factors:
        raise ValueError("reward shape: no factors given")

    if shape == 'geometric_all':
        allf = list(factors) + ([] if pose is None else [pose])
        return torch.stack(allf, dim=0).clamp_min(_FLOOR).prod(dim=0) ** (1.0 / len(allf))

    if shape == 'geometric':
        out = torch.stack(list(factors), dim=0).clamp_min(_FLOOR).prod(dim=0) ** 0.25
    else:
        # 'product': multiply in order, unclamped -- the original expression.
        out = factors[0]
        for f in factors[1:]:
            out = out * f
    return out if pose is None else out * pose

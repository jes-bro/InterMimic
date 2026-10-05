#!/usr/bin/env python3
"""Per-clip reward exponent from the clip's own motion scale.

THE QUESTION (Jess, 2026-10-01): "no root for OMOMO, geometric mean for
basketball, each motion its own weight, one universal method." The reward is a
product of per-aspect factors under a root (reward_shape.py). The product is
what InterMimic was built and validated on with OMOMO; the 1/N root was added
because the product collapses when every factor is poor at once, which is the
free-flight ball's case (basketball). Which one a clip needs tracks how much it
moves, so the exponent is made a function of the clip's own motion:

    R = (rb * ro * rig * rcg) ** e          e = clamp( (ref / std)^2 , 1/N , 1 )

    std   rms deviation of the clip's KEY bodies from their time-mean position
          (= the pooled temporal standard deviation of key-body position), metres
    ref   OMOMO's median key-body std, 0.5157 m over all 4421 OMOMO_new clips.
          The one constant: the amount of motion at and below which a clip is
          graded as plain PRODUCT. It is applied identically to every dataset.
    N     number of factors under the root (4 with the pose term off), so e is
          never below 1/N = today's geometric mean.

What each clip gets (std measured 2026-09-30):
    std <= 0.52 m   (half of OMOMO, all of CPR/kitchen)   e = 1      product
    std  = 0.73 m                                          e = 0.5
    std >= 1.03 m   (basketball median 1.02)               e = 0.25   the 4th root

WHY THE EXPONENT IS CAPPED AT 1 (decided 2026-10-05, from measurement). An
earlier draft let e exceed 1 for small-motion clips ("stricter than product"),
with a floor on std to cap it. Measured against the real policies it backfires:
PPO climbs total reward, so a clip's pull on the policy is the SLOPE of its
reward, d(P^e)/dP = e * P^(e-1). That exceeds the plain product's slope only
when the product P of the four factors is already high (P > 0.63 at e = 4,
> 0.75 at e = 8, > 0.88 at e = 27). The trained OMOMO teachers sit at P ~ 0.12
(rb .35, ro .73, rig .79, rcg .58), where an e = 8 clip pulls about a millionth
as hard as a median clip -- the small-motion clips would be IGNORED, the
opposite of the intent. Product (e = 1) is the strictest setting that is known
to train. So there is no floor and no knob beyond `ref`.

Consequence, stated plainly: this does NOT by itself make a 5 cm motion visible
to the reward (the flat-landscape problem measured on CPR). It restores the
product for every clip that moves no more than OMOMO's median and keeps the
root where the motion is large; small-motion visibility needs a different
mechanism and real small-motion data to design it against.

Cfg (env block; absent => every existing run is byte-identical):
    motionScaleReward:
      enable: true
      referenceStd: 0.5157      # metres; OMOMO_new median key-body std. REQUIRED.
Requires rewardShape: geometric_all (the exponent replaces the 1/N root).
"""
import torch

OMOMO_MEDIAN_KEY_BODY_STD_M = 0.5157   # all 4421 OMOMO_new clips, 21 key bodies, 2026-10-01
MAX_EXPONENT = 1.0                     # plain product; see "WHY THE EXPONENT IS CAPPED AT 1"


def parse_cfg(block):
    """{'enable', 'referenceStd'}; unknown keys / a missing reference are errors."""
    block = block or {}
    unknown = set(block) - {'enable', 'referenceStd'}
    if unknown:
        raise ValueError(f"motionScaleReward: unknown key(s) {sorted(unknown)}; valid: enable, "
                         f"referenceStd (there is no floor: the exponent is capped at product)")
    enable = bool(block.get('enable', False))
    ref = block.get('referenceStd', None)
    if enable:
        if ref is None:
            raise ValueError("motionScaleReward.enable is true: referenceStd (metres) is REQUIRED "
                             "(no silent default; see utils/motion_scale.py)")
        ref = float(ref)
        if not ref > 0:
            raise ValueError(f"motionScaleReward: referenceStd={ref} must be > 0")
    return {'enable': enable, 'referenceStd': ref}


def key_body_std(body_pos, key_body_ids):
    """Pooled temporal std of the key bodies' positions over ONE clip, metres.

    body_pos      (T, 52*3) or (T, 52, 3) reference body positions
    key_body_ids  indices of the reward's keyBodies
    = sqrt(mean over frames and key bodies of |p - time-mean(p)|^2). T == 1 -> 0.
    """
    kp = body_pos.reshape(body_pos.shape[0], -1, 3)[:, key_body_ids]
    dev = kp - kp.mean(dim=0, keepdim=True)
    return float(dev.pow(2).sum(dim=-1).mean().sqrt())


def exponent(std, reference_std, n_factors):
    """Exponent on the whole product: clamp((ref / std)^2, 1/N, 1).

    std may be a float or a tensor; the result has the same shape. A clip with
    no motion at all (std 0) gets the cap, 1 -- there is nothing to divide by
    and product is the answer for every small clip anyway.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    lo = 1.0 / n_factors
    if torch.is_tensor(std):
        ratio = float(reference_std) / torch.clamp(std, min=1e-9)
        return torch.clamp(ratio ** 2, min=lo, max=MAX_EXPONENT)
    s = float(std)
    if s <= 0.0:
        return MAX_EXPONENT
    return min(max((float(reference_std) / s) ** 2, lo), MAX_EXPONENT)

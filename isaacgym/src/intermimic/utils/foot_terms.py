#!/usr/bin/env python3
"""Two opt-in foot terms (Jess, 2026-10-20): make the feet matter to the reward.

THE FAULT. Every OMOMO policy slides or taps its feet instead of stepping. The reward
never asked for feet: the position term averages a foot's error with 20 other key
bodies, the rotation term with 51 other bodies, the velocity terms are 0, and the
contact term is a step function (a toe 1 mm off the floor on a swing frame scores
the same as one 10 cm up). Under the strict product the policy obeys the swing labels
with the smallest possible lift (the vibrating of the flight arms); under the soft
root it ignores them and glides (the skating of the msexp arms). The ankles roll
because a foot's tilt is 1/52 of the rotation error.

  rewardTerms.footWeight: {enable: true, weight: 5.0}
      multiply the ankles' and toes' error inside the EXISTING position and rotation
      terms by `weight` (the hook: compute_humanoid_reward's weight_hp / weight_hr).
      One knob, no new definitions: a 10 cm foot error stops being < 1% of the
      reward, the reference's lift and placement get tracked, a rolled ankle is
      pulled flat by the rotation term. Changes nothing when the feet already match.

  rewardTerms.footSwing: {enable: true, lambda: 50.0}
      on frames where the REFERENCE labels a toe as swing (contact_human < -0.1: "must
      touch nothing"), penalise the sim toe for being LOWER than the reference toe:
          err  = mean over the toes of [swing] * max(0, z_ref - z_sim)^2      [m^2]
          r    = exp(-lambda * err)
      one more factor under the root (N counts it). A toe at or above the reference
      height costs nothing; 5 cm short at lambda 50 costs exp(-0.125) = 0.88, 10 cm
      short 0.61. Stance frames are untouched (footWeight covers placement and tilt).

Both default OFF => every run byte-identical. Both need rewardShape geometric_all so
the factor count and the per-clip exponent rules stay consistent.
"""
import torch

DEFAULT_FOOT_WEIGHT = 5.0
DEFAULT_SWING_LAMBDA = 50.0
SWING_LABEL = -0.1          # contact_human below this = the reference says "must not contact"


def parse_foot_weight(block):
    """{'enable', 'weight'}; unknown keys / weight <= 0 are errors."""
    block = block or {}
    unknown = set(block) - {'enable', 'weight'}
    if unknown:
        raise ValueError(f"rewardTerms.footWeight: unknown key(s) {sorted(unknown)}; valid: enable, weight")
    enable = bool(block.get('enable', False))
    weight = float(block.get('weight', DEFAULT_FOOT_WEIGHT))
    if enable and not weight > 0:
        raise ValueError(f"rewardTerms.footWeight: weight={weight} must be > 0")
    return {'enable': enable, 'weight': weight}


def parse_foot_swing(block):
    """{'enable', 'lambda'}; unknown keys / lambda <= 0 are errors."""
    block = block or {}
    unknown = set(block) - {'enable', 'lambda'}
    if unknown:
        raise ValueError(f"rewardTerms.footSwing: unknown key(s) {sorted(unknown)}; valid: enable, lambda")
    enable = bool(block.get('enable', False))
    lam = float(block.get('lambda', DEFAULT_SWING_LAMBDA))
    if enable and not lam > 0:
        raise ValueError(f"rewardTerms.footSwing: lambda={lam} must be > 0")
    return {'enable': enable, 'lambda': lam}


def swing_lift_error(sim_z, ref_z, ref_label):
    """mean over the toes of [ref says swing] * max(0, ref_z - sim_z)^2, per env.
    sim_z, ref_z, ref_label: (num_envs, n_toes)."""
    swing = (ref_label < SWING_LABEL).float()
    short = (ref_z - sim_z).clamp_min(0.0)
    return (swing * short ** 2).mean(dim=-1)


def swing_factor(sim_z, ref_z, ref_label, lam):
    """exp(-lambda * swing_lift_error), per env, in (0, 1]."""
    return torch.exp(-lam * swing_lift_error(sim_z, ref_z, ref_label))

#!/usr/bin/env python3
"""Foot-skate reward factor: a planted foot must not slide.

Jess, 2026-10-18: every OMOMO policy slides its feet instead of stepping (renders of
2026-10-16/18, scripts/foot_skate.py: skating 30-34% of planted frames at 2x the
reference speed over a whole clip). The reward never sees it: the velocity terms
(pv, rv) are 0 in the stock recipe, the position term averages a foot's error over
21 bodies, and the contact term penalises a MISSING contact but not a foot that
stays down and slides. This factor measures skating directly, the way
scripts/foot_skate.py reads it:

    skate_i = |v_xy(foot i)|^2 * [foot i is in contact in the sim]
    r_foot  = exp(-lambda * mean_i skate_i)                        in (0, 1]

A foot in the air costs nothing (it may move as fast as it likes), a planted foot
pays for its horizontal speed. Contact is the SIM's contact force on that body
(|F| > contactThreshold), not the reference's label, so the term grades what the
policy does, not what the clip says. Stepping (lift, swing, plant) is free; sliding
(planted, moving) is not. Units: v in m/s, so at lambda 2 a foot sliding at 0.5 m/s
while planted costs exp(-0.5) = 0.61 on its own and 1.0 m/s costs 0.14; the
reference's own 15% of planted frames above 0.1 m/s (foot_skate.py on
sub6_woodchair_000) costs ~0.98.

Folded into the reward PRODUCT as one more factor under the geometric_all root
(N = 5 with it on, like the pose term), so the per-clip exponent rules
(motionScaleReward / objectFlightReward) count it. Default OFF => every run
byte-identical.

Cfg (env block):
    rewardTerms:
      footSkate:
        enable: true
        lambda: 2.0                     # per (m/s)^2 of planted horizontal speed
        bodies: [L_Toe, R_Toe]          # default: the toes
        contactThreshold: 1.0           # N; |contact force| above this = planted
Requires rewardShape: geometric_all.
"""
import torch

DEFAULT_BODIES = ("L_Toe", "R_Toe")
DEFAULT_LAMBDA = 2.0
DEFAULT_CONTACT_N = 1.0


def parse_cfg(block):
    """{'enable', 'lambda', 'bodies', 'contactThreshold'}; unknown keys / bad values are errors."""
    block = block or {}
    unknown = set(block) - {'enable', 'lambda', 'bodies', 'contactThreshold'}
    if unknown:
        raise ValueError(f"rewardTerms.footSkate: unknown key(s) {sorted(unknown)}; valid: enable, "
                         f"lambda, bodies, contactThreshold")
    enable = bool(block.get('enable', False))
    lam = float(block.get('lambda', DEFAULT_LAMBDA))
    bodies = list(block.get('bodies', DEFAULT_BODIES))
    thr = float(block.get('contactThreshold', DEFAULT_CONTACT_N))
    if enable:
        if not lam > 0:
            raise ValueError(f"rewardTerms.footSkate: lambda={lam} must be > 0")
        if not bodies or not all(isinstance(b, str) and b for b in bodies):
            raise ValueError(f"rewardTerms.footSkate: bodies must be a non-empty list of body names, got {bodies}")
        if not thr > 0:
            raise ValueError(f"rewardTerms.footSkate: contactThreshold={thr} must be > 0")
    return {'enable': enable, 'lambda': lam, 'bodies': bodies, 'contactThreshold': thr}


def skate_error(body_vel, contact_forces, foot_ids, contact_threshold):
    """mean_i |v_xy(foot i)|^2 * [|F(foot i)| > contact_threshold], per env.
    body_vel, contact_forces: (num_envs, num_bodies, 3); foot_ids: long tensor."""
    v = body_vel[:, foot_ids, :2]                                 # (E, F, 2)
    planted = (contact_forces[:, foot_ids, :].norm(dim=-1) > contact_threshold).float()   # (E, F)
    return ((v ** 2).sum(dim=-1) * planted).mean(dim=-1)            # (E,)


def factor(body_vel, contact_forces, foot_ids, lam, contact_threshold):
    """r_foot = exp(-lambda * skate_error), per env, in (0, 1]."""
    return torch.exp(-lam * skate_error(body_vel, contact_forces, foot_ids, contact_threshold))

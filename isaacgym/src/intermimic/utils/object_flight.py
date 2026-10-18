#!/usr/bin/env python3
"""Per-clip reward exponent from the clip's own OBJECT FLIGHT.

THE QUESTION (Jess, 2026-10-17): the reward exponent must be a FUNCTION of the clip
-- not a dataset label -- with OMOMO and soccer graded at (or near) the product and
basketball at the 1/N root. The product is what InterMimic was built on; the root was
added because the product collapses when every factor is poor at once, which is the
free-flight ball's case (reward_shape.py). Motion magnitude cannot pick the clips
(utils/motion_scale.py: OMOMO walking and basketball overlap at ~1 m of key-body
std, so the walking clips were graded at the root -- the sliding feet). What makes
basketball different is that the ball is out of anyone's control most of the time,
IN THE AIR between hand and floor. An OMOMO object is held or resting; a soccer ball
mostly rolls on the ground, where its pose is easy.

    flight = airborne_share * mean height above rest while airborne       [metres]
    airborne frame = no human-object contact (contact_obj ~ 0)
                     AND z - ground > AIR_HEIGHT (0.15 m), ground = 5th-percentile z
    e = 1 - (1 - 1/N) * clamp((flight - flatFlight) / (rootFlight - flatFlight), 0, 1)

so e = 1 (PRODUCT) for flight <= flatFlight, the 1/N root for flight >= rootFlight,
linear in between. Measured 2026-10-17 (scripts/clip_object_flight_stats.py):

    dataset      clips   flight p10 / median / p90 / max  (m)
    OMOMO_new    3356      0    /  0     /  0    / 0.19   (the max: a tripod raised with
                                                           no contact label, not flight)
    soccer15       61      0    /  0.09  / 0.32 / 0.51   (juggling / lobbed clips at the top)
    bball7         48     0.62  /  0.90  / 1.15 / 1.77

With rootFlight 0.6 and flatFlight 0 (the one-constant rule, arms tagged _flight):
OMOMO product on every clip (its tail at 0.77), soccer median 0.89 and top tenth
~0.6, basketball at the root on ~90% of clips. flatFlight 0.3 (arms tagged _flight2)
keeps soccer's bulk and OMOMO's tail at exactly product; its only effect on OMOMO and
basketball is that tail. The constant is the slope of the softening: how many metres
of uncontrolled flight per frame it takes to go from strict to soft.

Cfg (env block; absent => every existing run is byte-identical):
    objectFlightReward:
      enable: true
      rootFlight: 0.6       # metres of flight at which the 1/N root is reached. REQUIRED.
      flatFlight: 0.0       # product up to here (default 0)
Requires rewardShape: geometric_all (the exponent replaces the 1/N root) and is
exclusive with motionScaleReward. The clip channels are those of the 591-channel
layout (scripts/retarget_contact.py header): obj_pos 318:321 (z = 320), contact_obj 330.
"""
import torch

I_OBJ_Z = 320
I_CONTACT_OBJ = 330
AIR_HEIGHT = 0.15          # metres above the resting height = airborne
CONTACT_THRES = 0.1        # |contact_obj| below this = no human-object contact
GROUND_QUANTILE = 0.05     # the object's resting height: 5th percentile of its z


def parse_cfg(block):
    """{'enable', 'rootFlight', 'flatFlight'}; unknown keys / a missing rootFlight are errors."""
    block = block or {}
    unknown = set(block) - {'enable', 'rootFlight', 'flatFlight'}
    if unknown:
        raise ValueError(f"objectFlightReward: unknown key(s) {sorted(unknown)}; valid: enable, "
                         f"rootFlight, flatFlight")
    enable = bool(block.get('enable', False))
    root = block.get('rootFlight', None)
    flat = block.get('flatFlight', 0.0)
    if enable:
        if root is None:
            raise ValueError("objectFlightReward.enable is true: rootFlight (metres) is REQUIRED "
                             "(no silent default; see utils/object_flight.py)")
        root, flat = float(root), float(flat)
        if not flat >= 0:
            raise ValueError(f"objectFlightReward: flatFlight={flat} must be >= 0")
        if not root > flat:
            raise ValueError(f"objectFlightReward: rootFlight={root} must be > flatFlight={flat}")
    return {'enable': enable, 'rootFlight': root, 'flatFlight': flat}


def clip_flight(hoi_data):
    """Flight of ONE clip, metres: a (T, 591) tensor's airborne share times its mean
    airborne height above rest. 0 for a held or resting object; T < 2 -> 0."""
    z = hoi_data[:, I_OBJ_Z].float()
    contact = hoi_data[:, I_CONTACT_OBJ].float()
    if z.numel() < 2:
        return 0.0
    ground = torch.quantile(z, GROUND_QUANTILE)        # linear interpolation, = numpy.percentile
    airborne = (contact.abs() < CONTACT_THRES) & (z - ground > AIR_HEIGHT)
    if not bool(airborne.any()):
        return 0.0
    share = airborne.float().mean()
    height = (z - ground)[airborne].mean()
    return float(share * height)


def exponent(flight, root_flight, flat_flight, n_factors):
    """Exponent on the whole product: 1 - (1 - 1/N) * clamp((flight - flat) / (root - flat), 0, 1).
    flight may be a float or a tensor; the result has the same shape."""
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    if not root_flight > flat_flight >= 0:
        raise ValueError(f"need rootFlight {root_flight} > flatFlight {flat_flight} >= 0")
    lo = 1.0 / n_factors
    if torch.is_tensor(flight):
        t = ((flight - flat_flight) / (root_flight - flat_flight)).clamp(0.0, 1.0)
        return 1.0 - (1.0 - lo) * t
    t = min(1.0, max(0.0, (float(flight) - flat_flight) / (root_flight - flat_flight)))
    return 1.0 - (1.0 - lo) * t

#!/usr/bin/env python3
"""Prove the objectAug 'hold' reward is gated on the REFERENCE grip.

The bug this guards against: the proximity ('keep a grip') term used to be applied
every frame, so on frames where the source clip is NOT holding the object (free
flight, at rest, idle hand) the reward still pulled the wrists onto the object.

This test reproduces the exact per-hand gating logic from
InterMimic._compute_hold_reward (pure torch, no Isaac Gym) and asserts:
  1. reference not holding EITHER hand  -> reward == 1 (neutral), regardless of
     where the hands are.
  2. reference holding, hand ON object  -> reward high (~1).
  3. reference holding, hand FAR/off    -> reward pulled down (< neutral).
  4. one-handed reference hold: the idle (non-reference) hand never drags the
     reward down, even parked far from the object.
"""
import sys
import torch

LAMBDA = 5.0
LEFT, RIGHT = list(range(17, 33)), list(range(36, 52))
NLINKS = 52


def hold_reward(hand_pos, obj_points, ref_contact, live_contact, lam=LAMBDA):
    """Mirror of _compute_hold_reward (per-hand ref-gated)."""
    min_d = torch.cdist(hand_pos, obj_points).min(dim=-1)[0]      # (E, 2)
    r_prox = torch.exp(-lam * min_d)                             # (E, 2)
    factors = []
    for h, ids in enumerate((LEFT, RIGHT)):
        ref_any = (ref_contact[:, ids] > 0.1).any(dim=-1).float()
        live_any = (live_contact[:, ids] > 0.1).any(dim=-1).float()
        shaped = r_prox[:, h] * (0.5 + 0.5 * live_any)
        factors.append(ref_any * shaped + (1.0 - ref_any))
    return torch.stack(factors, dim=-1).mean(dim=-1)


def mk(E=1, P=8):
    # object as a small cloud around the origin; hands placed relative to it.
    obj = torch.zeros(E, P, 3)
    obj[:, :, 0] = torch.linspace(-0.05, 0.05, P)
    ref = torch.zeros(E, NLINKS)
    live = torch.zeros(E, NLINKS)
    return obj, ref, live


fails = 0
def check(name, ok):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name)
    fails += (0 if ok else 1)


# 1) reference holding NEITHER hand -> neutral 1, even with hands jammed on the object.
obj, ref, live = mk()
hands_on = torch.zeros(1, 2, 3)                     # both wrists at object center
r = hold_reward(hands_on, obj, ref, live)
check("ref not holding -> reward == 1 (neutral) even with hands on object",
      torch.allclose(r, torch.ones(1), atol=1e-5))

# 2) reference holding both; hands ON object and in contact -> ~1.
obj, ref, live = mk()
ref[:, LEFT] = 1; ref[:, RIGHT] = 1
live[:, LEFT] = 1; live[:, RIGHT] = 1
r_on = hold_reward(torch.zeros(1, 2, 3), obj, ref, live)
check("ref holding + hands on + in contact -> high (>0.95)", float(r_on) > 0.95)

# 3) reference holding both; hands FAR and not in contact -> pulled well below neutral.
obj, ref, live = mk()
ref[:, LEFT] = 1; ref[:, RIGHT] = 1                # ref holds, live contact stays 0
hands_far = torch.zeros(1, 2, 3); hands_far[:, :, 2] = 0.5   # 0.5 m above object
r_far = hold_reward(hands_far, obj, ref, live)
check("ref holding but hands far/off -> low (<0.2)", float(r_far) < 0.2)
check("holding+on beats holding+far", float(r_on) > float(r_far))

# 4) one-handed reference hold (left only); RIGHT wrist parked far must NOT hurt.
obj, ref, live = mk()
ref[:, LEFT] = 1                                    # only left hand used by source
live[:, LEFT] = 1
hands = torch.zeros(1, 2, 3); hands[:, 1, 2] = 1.0  # right wrist parked 1 m away
r_1h = hold_reward(hands, obj, ref, live)
# left factor ~1 (on+contact), right factor neutral 1 -> mean ~1.
check("one-handed ref hold: idle far hand doesn't drag reward down (>0.95)",
      float(r_1h) > 0.95)

print()
if fails:
    print(f"FAILED ({fails})"); sys.exit(1)
print("ALL GREEN -- hold reward is correctly gated on the reference grip.")

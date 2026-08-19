#!/usr/bin/env python3
"""Diagnose the CARI4D basketball clip's reference quality in one pass:

  1. contact timeline   -- the recon's contact_obj flag as spans (does contact
                           ever RESUME after the dribble bounce?)
  2. hand-ball distance -- min wrist-to-ball distance per frame (is the recon
                           ball ever back NEAR the hand, even if the flag says
                           no? flags-broken vs trajectory-broken)
  3. floor offset       -- lowest body point per frame (grounded frames at
                           z ~= 0.23 = the known monocular floor offset is live)
  4. ball height        -- ball z per frame (bounce profile vs the sim floor)

Everything prints as aligned per-frame/per-span tables; no plots, no GPU, no
Isaac Gym -- runs anywhere the clip file and the subject MJCF exist (cluster).

  python3 scripts/inspect_bball_clip.py
  python3 scripts/inspect_bball_clip.py --clip InterAct/behave_cari4d/sub100_bball_000.pt --every 2
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402  (the validated MJCF parser)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Clip layout (see retarget_contact.py header; verified against OMOMO_new):
I_ROOT = slice(0, 3)          # root position -- what the SIM actually drives FK from
I_BODY = slice(162, 318)      # 52 bodies x 3, world frame (stored kinematics)
I_OBJP = slice(318, 321)      # object position
I_CONTACT_OBJ = 330           # 1.0 where the recon says object is in contact


def contact_spans(flags):
    """[(start, end, bool)] runs of the contact flag."""
    out, i = [], 0
    for k, g in itertools.groupby(flags.tolist()):
        n = len(list(g))
        out.append((i, i + n - 1, bool(k)))
        i += n
    return out


def wrist_indices(mjcf_path):
    """L_Wrist / R_Wrist indices in the clip's 52-body order, from the MJCF
    itself -- no hardcoded indices (they'd silently rot if the order changed)."""
    names = [n for n, _, _ in _parse_mjcf_tree(mjcf_path)]
    try:
        return [names.index("L_Wrist"), names.index("R_Wrist")], names
    except ValueError:
        raise SystemExit(f"FATAL: L_Wrist/R_Wrist not in {mjcf_path} body names; "
                         f"got {names[:8]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(
        REPO, "InterAct/behave_cari4d/sub100_bball_000.pt"))
    ap.add_argument("--mjcf", default=os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml"),
        help="the clip subject's MJCF (for wrist indices by NAME)")
    ap.add_argument("--every", type=int, default=4,
                    help="print every Nth frame in the per-frame tables")
    args = ap.parse_args()

    if not os.path.exists(args.clip):
        raise SystemExit(f"FATAL: clip not found: {args.clip} (cluster-only data?)")
    c = torch.load(args.clip, map_location="cpu")
    c = c.detach() if hasattr(c, "detach") else c
    T = c.shape[0]
    bp = c[:, I_BODY].view(T, 52, 3)
    obj = c[:, I_OBJP]
    flags = c[:, I_CONTACT_OBJ] > 0.5
    wrists, _names = wrist_indices(args.mjcf)
    hand_d = (bp[:, wrists, :] - obj[:, None, :]).norm(dim=-1).min(dim=1).values
    lowest = bp[:, :, 2].min(dim=1).values

    print(f"clip: {args.clip}  ({T} frames)")

    print(f"\n== 1. contact_obj flag timeline ==")
    for s, e, k in contact_spans(flags):
        print(f"  frames {s:3d}-{e:3d} ({e-s+1:3d}f): {'CONTACT' if k else 'free'}")
    n_resume = sum(1 for i, (s, e, k) in enumerate(contact_spans(flags)) if k and i > 0)
    print(f"  -> contact spans after the first: {n_resume} "
          f"({'contact RESUMES' if n_resume else 'contact NEVER resumes -- no catch in the supervision'})")

    print(f"\n== 2/3/4. per-frame: hand-ball dist | lowest body z | ball z | flag ==")
    print(f"  {'frame':>5s} {'hand-ball(m)':>12s} {'lowest z(m)':>11s} {'ball z(m)':>9s}  flag")
    for i in range(0, T, args.every):
        print(f"  {i:5d} {hand_d[i]:12.3f} {lowest[i]:11.3f} {obj[i,2]:9.3f}  "
              f"{'CONTACT' if flags[i] else 'free'}")

    # 5. FK-vs-stored consistency. The sim drives the humanoid by FK from
    # root_pos+root_rot+dof_pos; hand-ball above is measured from the STORED
    # body_pos channels. If a conversion transform (rotate / drop-to-floor)
    # touched one set and not the other, body_pos can say "ball in hand" while
    # the FK-driven reference stands somewhere else. root_pos vs body_pos[0]
    # (the pelvis) catches the translation/rotation forms of that mismatch.
    root_delta = (c[:, I_ROOT] - bp[:, 0, :]).norm(dim=-1)
    print(f"\n== 5. root_pos vs stored pelvis (FK/body_pos consistency) ==")
    print(f"  |root_pos - body_pos[0]|: mean {root_delta.mean():.3f} m  "
          f"max {root_delta.max():.3f} m at frame {int(root_delta.argmax())}")
    print(f"  read: ~0.0x m (fixed pelvis offset) = consistent; growing or")
    print(f"        decimeter+ deltas = conversion transformed the channel sets")
    print(f"        differently -- the sim's reference does NOT match these tables.")

    grounded = lowest[lowest < lowest.median() + 0.05]
    print(f"\n== summary ==")
    print(f"  grounded-frame lowest-body z: median {grounded.median():.3f} m "
          f"(~0.00 = floor ok, ~0.23 = the known monocular floor offset is LIVE)")
    print(f"  hand-ball distance: min {hand_d.min():.3f} m at frame {int(hand_d.argmin())}; "
          f"post-frame-50 min {hand_d[50:].min():.3f} m at frame {50+int(hand_d[50:].argmin())}")
    print(f"  ball z: min {obj[:,2].min():.3f}  max {obj[:,2].max():.3f} m")
    print(f"  read: flags say no resume + hand-ball gets small again -> FLAGS broken (relabel);")
    print(f"        hand-ball never gets small post-bounce -> ball TRAJECTORY broken (rebuild).")


if __name__ == "__main__":
    main()

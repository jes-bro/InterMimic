#!/usr/bin/env python3
"""Re-derive the reference's PER-BODY hand contact flags (contact_human, channels
331..382) from geometry, so the contact reward becomes earnable.

WHY. relabel_contact_flags.py rewrote channel 330 (contact_obj) only -- it says
so in its own docstring. But compute_cg_reward's rcg_hand does not read channel
330; it grades the sim's per-body contact against contact_human. So the _cf build
fixed the channel the free-flight gate and difficulty bucketing use, and left the
one that drives the contact reward carrying the ORIGINAL recon's story.

Measured on behave_cari4d_optj3d_cf/sub100_bball_000.pt (2026-08-26):
    frames where contact_human flags a hand body:              53 of 101
    of those, frames with NO hand body touching the ball:      21 (40%)
    worst: frame 11, nearest hand body +0.187 m from the surface
    contact_obj vs contact_human(hand) disagree on:            15 of 101
rcg_hand cannot be earned on those 21 frames by ANY policy -- to score, the
humanoid would have to be where the reference says AND touching the ball, and
those are different places. r3's eval measured rcg = 0.141 on held frames.

WHAT THIS DOES. For each frame, for each of the 32 hand bodies rcg_hand grades
(compute_cg_reward's range(17,33) + range(36,52), finger bodies included), set
the flag from the body's distance to the ball SURFACE. Non-hand bodies are left
untouched -- they carry floor and self contact that rcg_other/rcg_all read, and
this script has no business rewriting those. Positions are NEVER touched.

By default contact_obj is re-derived from the SAME criterion, so the two channels
agree by construction; that inconsistency is half the reported problem. Pass
--keep-contact-obj to leave channel 330 alone.

THE TRI-STATE. contact_human is not boolean: compute_cg_reward's ecg_all keys on
`ref_human_contact < -contact_thres`, i.e. negative means "should be free" and
positive "should be in contact". Rather than hardcode +1/-1, this script reads
the values ALREADY PRESENT in the source clip's hand channels and writes those
same values back, so whatever convention the data uses is preserved. --census
prints what it found without writing anything.

THE GUARD (Jess, 2026-08-26). Relabelling is only the right fix if the recon
actually achieves contact somewhere. If no frame has any hand body within the
threshold, the flags are not the problem -- the reconstruction is -- and this
script REFUSES rather than emitting an all-free clip that would silently delete
the contact supervision entirely.

  # look before you leap: what convention is in there, and what would change?
  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf \
      --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --census

  python3 scripts/relabel_contact_human.py \
      --src-dir InterAct/behave_cari4d_optj3d_cf \
      --dst-dir InterAct/behave_cari4d_optj3d_cf2 \
      --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml \
      --threshold 0.02
"""
import argparse
import itertools
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402  (the validated MJCF parser)

I_BODY = slice(162, 318)              # 52 bodies x 3, world frame
I_OBJP = slice(318, 321)              # object position
I_CONTACT_OBJ = 330
I_CONTACT_HUMAN = slice(331, 383)     # 52 per-body flags (intermimic.py:765)
# compute_cg_reward's own hand lists, in the clip's 52-body order. 17 = L_Wrist,
# 36 = R_Wrist (confirmed independently by inspect_bball_clip.wrist_indices).
HAND_BODY_IDS = list(range(17, 33)) + list(range(36, 52))


def spans(flags):
    """Compact run-length view of a 0/1 timeline, for before/after diffing."""
    out, i = [], 0
    for k, g in itertools.groupby(list(flags)):
        n = len(list(g))
        out.append(f"{i}-{i+n-1}:{'C' if k else 'f'}")
        i += n
    return " ".join(out)


def surface_gap(t, ball_radius):
    """Per-frame, per-hand-body distance to the ball SURFACE.

    Negative = interpenetrating (a solid grip in a rigid-body approximation of a
    hand), ~0 = touching, positive = not touching. Shape [T, 32]."""
    T = t.shape[0]
    bp = t[:, I_BODY].view(T, 52, 3)
    obj = t[:, I_OBJP]
    return (bp[:, HAND_BODY_IDS, :] - obj[:, None, :]).norm(dim=-1) - ball_radius


def majority_smooth(x, window=3):
    """Majority filter along time, per body -- kills single-frame flicker the
    way relabel_contact_flags.py does, but vectorised over the 32 bodies."""
    if window <= 1:
        return x
    half, T = window // 2, x.shape[0]
    out = x.clone()
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        out[i] = (x[lo:hi].float().mean(dim=0) >= 0.5)
    return out


def observed_levels(t):
    """The 'in contact' and 'free' values ALREADY used by this clip's hand
    channels, so the tri-state convention is preserved rather than guessed.

    Returns (contact_value, free_value, sorted_distinct_values)."""
    ch = t[:, I_CONTACT_HUMAN][:, HAND_BODY_IDS]
    vals = sorted({round(float(v), 4) for v in torch.unique(ch)})
    pos = [v for v in vals if v > 0.1]
    neg = [v for v in vals if v < -0.1]
    # Fall back to the convention compute_cg_reward implies (+ = should touch,
    # - = should be free) only if the clip itself shows no example.
    contact_v = max(pos) if pos else 1.0
    free_v = min(neg) if neg else (0.0 if 0.0 in vals else -1.0)
    return contact_v, free_v, vals


def census(t, ball_radius, threshold):
    """What is in the source, and what would change -- no writes."""
    gap = surface_gap(t, ball_radius)
    ch = t[:, I_CONTACT_HUMAN][:, HAND_BODY_IDS]
    old_any = (ch > 0.1).any(dim=1)
    new_any = (gap < threshold).any(dim=1)
    contact_v, free_v, vals = observed_levels(t)
    claimed = int(old_any.sum())
    unearnable = int(((gap.min(dim=1).values > 0) & old_any).sum())
    return {
        'frames': t.shape[0],
        'distinct_values': vals,
        'contact_value': contact_v,
        'free_value': free_v,
        'claimed_contact_frames': claimed,
        'unearnable_frames': unearnable,
        'ever_touches': bool((gap < threshold).any()),
        'old_any': old_any.numpy().astype(int),
        'new_any': new_any.numpy().astype(int),
        'min_gap': float(gap.min()),
    }


def relabel(t, ball_radius, threshold, smooth, keep_contact_obj):
    """Return a NEW tensor with hand contact_human (and optionally contact_obj)
    re-derived from geometry. Positions and non-hand bodies untouched."""
    out = t.clone()
    gap = surface_gap(t, ball_radius)
    touching = majority_smooth(gap < threshold, smooth)          # [T, 32] bool
    contact_v, free_v, _ = observed_levels(t)

    ch = out[:, I_CONTACT_HUMAN].clone()
    hand = ch[:, HAND_BODY_IDS]
    hand[touching] = contact_v
    hand[~touching] = free_v
    ch[:, HAND_BODY_IDS] = hand
    out[:, I_CONTACT_HUMAN] = ch

    if not keep_contact_obj:
        # Same criterion, so the two channels cannot disagree: the object is in
        # contact exactly when some hand body is touching it.
        out[:, I_CONTACT_OBJ] = touching.any(dim=1).to(out.dtype)
    return out, touching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", help="required unless --census")
    ap.add_argument("--mjcf", required=True,
                    help="subject MJCF -- used only to VERIFY the body order")
    ap.add_argument("--ball-radius", type=float, default=0.13,
                    help="object radius in m (bball recon sphere is 0.26 m dia)")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="hand-body-to-ball-SURFACE contact distance (m). Much "
                         "tighter than relabel_contact_flags.py's 0.25 because "
                         "that measured WRIST-to-CENTER; here the finger bodies "
                         "are included and the ball radius is already removed.")
    ap.add_argument("--smooth", type=int, default=3,
                    help="majority-filter window in frames (1 = off)")
    ap.add_argument("--keep-contact-obj", action="store_true",
                    help="leave channel 330 alone (default: re-derive it from "
                         "the same criterion so the channels agree)")
    ap.add_argument("--census", action="store_true",
                    help="report the source's convention and what would change; "
                         "write nothing")
    args = ap.parse_args()

    src = Path(args.src_dir)
    if not src.is_dir():
        sys.exit(f"FATAL: src dir not found: {src}")
    if not args.census:
        if not args.dst_dir:
            sys.exit("FATAL: --dst-dir is required unless --census")
        dst = Path(args.dst_dir)
        if dst.exists():
            sys.exit(f"FATAL: dst dir already exists: {dst} -- refusing to overwrite")

    # The body order is load-bearing: HAND_BODY_IDS are positional. Verify
    # against the MJCF rather than trusting it, because a body-order change
    # would silently relabel the wrong 32 channels.
    names = [n for n, _, _ in _parse_mjcf_tree(args.mjcf)]
    if len(names) != 52 or names[17] != "L_Wrist" or names[36] != "R_Wrist":
        sys.exit(f"FATAL: unexpected body order in {args.mjcf}: "
                 f"{len(names)} bodies, [17]={names[17] if len(names) > 17 else '?'}, "
                 f"[36]={names[36] if len(names) > 36 else '?'}. "
                 f"HAND_BODY_IDS is positional and would relabel the wrong channels.")

    clips = sorted(src.glob("*.pt"))
    if not clips:
        sys.exit(f"FATAL: no .pt clips in {src}")

    # --- Pass 1: census + the guard, on every clip, before writing anything. ---
    stats = {}
    for f in clips:
        t = torch.load(f, map_location="cpu", weights_only=False).detach()
        st = census(t, args.ball_radius, args.threshold)
        stats[f.name] = st
        print(f"\n{f.name}: {st['frames']} frames")
        print(f"  contact_human hand values present: {st['distinct_values']}")
        print(f"  -> writing contact={st['contact_value']}  free={st['free_value']}")
        print(f"  frames claiming hand contact (source): {st['claimed_contact_frames']}")
        print(f"  of those, UNEARNABLE (no hand body touching): {st['unearnable_frames']}")
        print(f"  closest any hand body ever gets to the surface: {st['min_gap']:+.3f} m")
        print(f"  old: {spans(st['old_any'])}")
        print(f"  new: {spans(st['new_any'])}   (threshold {args.threshold} m)")

    # Jess's guard: relabelling presumes the recon DOES make contact somewhere.
    # If it never does, an all-free relabel would silently delete the contact
    # supervision and look like a successful fix.
    dead = [n for n, st in stats.items() if not st['ever_touches']]
    if dead:
        sys.exit(
            f"\nFATAL: no hand body ever comes within {args.threshold} m of the "
            f"ball surface in: {', '.join(dead)}\n"
            f"  Relabelling would produce an all-free clip and delete the contact\n"
            f"  supervision entirely. This is a RECONSTRUCTION problem, not a\n"
            f"  labelling one -- fix the recon, or raise --threshold deliberately.")

    if args.census:
        print("\ncensus only -- nothing written.")
        return

    # --- Pass 2: write. ---
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".pt":
            shutil.copy2(f, dst / f.name)

    for f in clips:
        t = torch.load(f, map_location="cpu", weights_only=False).detach()
        out, touching = relabel(t, args.ball_radius, args.threshold, args.smooth,
                                args.keep_contact_obj)
        # Positions must be byte-identical: this script relabels, never moves.
        assert torch.equal(out[:, I_BODY], t[:, I_BODY]), "positions changed!"
        assert torch.equal(out[:, I_OBJP], t[:, I_OBJP]), "object moved!"
        # Non-hand contact_human channels must be untouched too.
        others = [i for i in range(52) if i not in HAND_BODY_IDS]
        assert torch.equal(out[:, I_CONTACT_HUMAN][:, others],
                           t[:, I_CONTACT_HUMAN][:, others]), "non-hand flags changed!"
        torch.save(out, dst / f.name)
        st = stats[f.name]
        print(f"\n{f.name}: wrote {dst / f.name}")
        print(f"  hand-contact frames {st['claimed_contact_frames']} -> "
              f"{int(touching.any(dim=1).sum())}")
        if not args.keep_contact_obj:
            print(f"  contact_obj re-derived from the same criterion "
                  f"(channels now agree by construction)")
    print(f"\ndone -> {dst} (flags only; positions and non-hand bodies untouched)")
    print("receipt: re-run inspect_bball_clip.py section 8 on the new dir -- "
          "'frames claiming contact with NO hand body touching' should be 0.")


if __name__ == "__main__":
    main()

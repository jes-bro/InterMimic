#!/usr/bin/env python3
"""Re-derive the reference's FOOT contact flags from geometry, for the soccer
reconstructions. The soccer twin of relabel_contact_human.py -- that script is
hands-only (it rewrites the 32 hand bodies the reward's rcg_hand grades) and is
deliberately NOT modified; this one rewrites the 4 foot bodies and nothing else.

WHY FEET. In a kick the contact is foot-to-ball. compute_cg_reward grades the
hands through rcg_hand and EVERY other body through rcg_other wherever the
reference says that body touches the object (intermimic.py ~2442-2446), so the
foot flags (channels 331..382, L_Ankle/L_Toe/R_Ankle/R_Toe) are what the
contact reward reads on a soccer clip. If the recon says "foot on ball" on a
frame where the foot is 10 cm away, that reward is unearnable by any policy.

WHAT IS MIRRORED from the hand script, unchanged:
  * gap = distance(body origin, ball centre) - ball radius, per frame per body;
    radius per clip from its own mesh (--ball-radius-from-mesh)
  * touching = gap < --threshold, then a 3-frame majority filter
  * tri-state preserved: touching -> the clip's own +1; a false +1 -> the least
    committal value the clip already uses (0); existing 0 / -1 left as found;
    -1 ("must not touch") is never invented (--free-value negative opts in)
  * contact_obj (channel 330) re-derived from the same criterion, so the two
    channels agree (--keep-contact-obj leaves it)
  * the guard: a clip in which NO foot body ever comes within the threshold is
    REFUSED, not silently written all-free
  * --census: measure and print, write nothing
WHAT DIFFERS: the body set (4 feet, by NAME from the rig, not by hard-coded
index), and the body-order check looks for the ankle/toe names.

THE THRESHOLD IS NOT COPIED. 2 cm suited hands because reconstructed grips put
finger joints 8-11 cm INSIDE the ball. Foot body origins sit inside the shoe
(toe joint) or above the sole (ankle), so a real kick can register as a small
positive gap. --census therefore also prints a SWEEP: at each candidate
threshold, the fraction of frames the recon itself labelled foot-contact that
fall inside it, vs the fraction of recon-free frames that do. Pick the value
where those separate, then run the relabel with --threshold set explicitly.

    python3 scripts/relabel_contact_soccer.py --src-dir InterAct/behave_cari4d_soccer \\
        --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub405.xml \\
        --ball-radius-from-mesh isaacgym/src/intermimic/data/assets/objects/objects --census
    python3 scripts/relabel_contact_soccer.py --src-dir ... --dst-dir ..._cf \\
        --mjcf ... --ball-radius-from-mesh ... --threshold 0.03
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree                                   # noqa: E402
from relabel_contact_human import (I_BODY, I_OBJP, I_CONTACT_OBJ, I_CONTACT_HUMAN,  # noqa: E402
                                   spans, majority_smooth, radius_from_mesh)

FOOT_BODY_NAMES = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
SWEEP_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]


def foot_body_ids(mjcf):
    """Indices of the 4 foot bodies in the clip's 52-body order, from the rig by
    NAME; a rig with a different order or missing feet is a hard error."""
    names = [n for n, _, _ in _parse_mjcf_tree(mjcf)]
    if len(names) != 52:
        raise SystemExit(f"FATAL: {mjcf} has {len(names)} bodies, expected 52")
    missing = [n for n in FOOT_BODY_NAMES if n not in names]
    if missing:
        raise SystemExit(f"FATAL: {mjcf} lacks foot bodies {missing}; names: {names[:12]}...")
    return [names.index(n) for n in FOOT_BODY_NAMES]


def surface_gap(t, ball_radius, body_ids):
    """[T, len(body_ids)] distance from each foot body origin to the ball SURFACE
    (negative = inside)."""
    T = t.shape[0]
    bp = t[:, I_BODY].view(T, 52, 3)
    obj = t[:, I_OBJP]
    return (bp[:, body_ids, :] - obj[:, None, :]).norm(dim=-1) - ball_radius


def observed_levels(t, body_ids):
    """(contact_value, clear_value, distinct values) on the FOOT channels -- the
    same rule as the hand script: write the clip's own +1, clear a false claim to
    its least committal value (0 if present), never invent -1."""
    ch = t[:, I_CONTACT_HUMAN][:, body_ids]
    vals = sorted({round(float(v), 4) for v in torch.unique(ch)})
    pos = [v for v in vals if v > 0.1]
    contact_v = max(pos) if pos else 1.0
    clear_v = 0.0 if any(abs(v) < 0.1 for v in vals) else (
        min([v for v in vals if v < -0.1], default=-1.0))
    return contact_v, clear_v, vals


def census(t, ball_radius, threshold, body_ids):
    gap = surface_gap(t, ball_radius, body_ids)
    ch = t[:, I_CONTACT_HUMAN][:, body_ids]
    old_any = (ch > 0.1).any(dim=1)
    new_any = (gap < threshold).any(dim=1)
    contact_v, clear_v, vals = observed_levels(t, body_ids)
    per_frame_min = gap.min(dim=1).values
    return {
        "frames": t.shape[0], "distinct_values": vals,
        "contact_value": contact_v, "clear_value": clear_v,
        "claimed_contact_frames": int(old_any.sum()),
        "unearnable_frames": int(((per_frame_min > 0) & old_any).sum()),
        "ever_touches": bool((gap < threshold).any()),
        "old_any": old_any.numpy().astype(int), "new_any": new_any.numpy().astype(int),
        "min_gap": float(gap.min()),
        # for the threshold sweep: per-frame closest foot, split by the source's own claim
        "gap_claimed": per_frame_min[old_any].numpy(),
        "gap_free": per_frame_min[~old_any].numpy(),
    }


def relabel(t, ball_radius, threshold, smooth, keep_contact_obj, body_ids,
            free_value="minimal"):
    out = t.clone()
    gap = surface_gap(t, ball_radius, body_ids)
    touching = majority_smooth(gap < threshold, smooth)                # [T, 4] bool
    contact_v, clear_v, vals = observed_levels(t, body_ids)
    neg_v = min([v for v in vals if v < -0.1], default=-1.0)
    ch = out[:, I_CONTACT_HUMAN].clone()
    feet = ch[:, body_ids].clone()
    if free_value == "negative":
        feet[~touching] = neg_v
    else:
        false_claim = (~touching) & (feet > 0.1)
        feet[false_claim] = clear_v
    feet[touching] = contact_v
    ch[:, body_ids] = feet
    out[:, I_CONTACT_HUMAN] = ch
    if not keep_contact_obj:
        out[:, I_CONTACT_OBJ] = touching.any(dim=1).to(out.dtype)
    return out, touching


def print_sweep(stats):
    import numpy as np
    claimed = np.concatenate([s["gap_claimed"] for s in stats.values()]) if stats else np.zeros(0)
    free = np.concatenate([s["gap_free"] for s in stats.values()]) if stats else np.zeros(0)
    print(f"\nTHRESHOLD SWEEP over {len(claimed)} recon-labelled foot-contact frames and "
          f"{len(free)} recon-free frames (closest foot body to the surface):")
    print(f"  {'thr (m)':>7} {'claimed inside':>15} {'free inside':>12}")
    for thr in SWEEP_THRESHOLDS:
        ci = float((claimed < thr).mean()) if len(claimed) else float("nan")
        fi = float((free < thr).mean()) if len(free) else float("nan")
        print(f"  {thr:>7.2f} {100 * ci:>14.1f}% {100 * fi:>11.1f}%")
    if len(claimed):
        q = np.percentile(claimed, [10, 50, 90])
        print(f"  claimed-contact frames: closest-foot gap p10 {100*q[0]:+.1f} cm, "
              f"median {100*q[1]:+.1f} cm, p90 {100*q[2]:+.1f} cm")
    print("  Pick the smallest threshold that captures most claimed frames while "
          "leaving most free frames outside, then rerun with --threshold.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", help="required unless --census")
    ap.add_argument("--mjcf", required=True, help="subject MJCF -- foot body indices by name")
    ap.add_argument("--ball-radius-from-mesh", metavar="OBJECTS_DIR",
                    help="per-clip radius from <OBJECTS_DIR>/<object>/<object>.obj")
    ap.add_argument("--ball-radius", type=float, default=0.11,
                    help="one radius for all clips if --ball-radius-from-mesh is not given")
    ap.add_argument("--threshold", type=float, default=None,
                    help="foot-to-surface contact threshold (m). REQUIRED to write; "
                         "--census runs at 0.02 for the per-clip table plus the sweep.")
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--keep-contact-obj", action="store_true")
    ap.add_argument("--free-value", choices=["minimal", "negative"], default="minimal")
    ap.add_argument("--census", action="store_true", help="measure only, write nothing")
    args = ap.parse_args()

    src = Path(args.src_dir)
    if not src.is_dir():
        sys.exit(f"FATAL: src dir not found: {src}")
    if not args.census:
        if not args.dst_dir:
            sys.exit("FATAL: --dst-dir is required unless --census")
        if args.threshold is None:
            sys.exit("FATAL: --threshold is required to write (choose it from --census's sweep)")
        dst = Path(args.dst_dir)
        if dst.exists():
            sys.exit(f"FATAL: dst dir already exists: {dst} -- refusing to overwrite")
    threshold = 0.02 if args.threshold is None else args.threshold

    body_ids = foot_body_ids(args.mjcf)
    clips = sorted(src.glob("*.pt"))
    if not clips:
        sys.exit(f"FATAL: no .pt clips in {src}")

    radius = {}
    for f in clips:
        if args.ball_radius_from_mesh:
            obj = f.stem.split("_")[-2]
            mesh = Path(args.ball_radius_from_mesh) / obj / f"{obj}.obj"
            if not mesh.is_file():
                sys.exit(f"FATAL: --ball-radius-from-mesh: no mesh for {f.name} at {mesh}")
            radius[f.name] = radius_from_mesh(mesh)
        else:
            radius[f.name] = args.ball_radius

    stats = {}
    for f in clips:
        t = torch.load(f, map_location="cpu", weights_only=False).detach()
        st = census(t, radius[f.name], threshold, body_ids)
        stats[f.name] = st
        print(f"\n{f.name}: {st['frames']} frames  (ball radius {radius[f.name]:.4f} m)")
        print(f"  foot contact_human values present: {st['distinct_values']}")
        print(f"  frames claiming foot contact (source): {st['claimed_contact_frames']}")
        print(f"  of those, UNEARNABLE at 0 cm (no foot body inside the surface): {st['unearnable_frames']}")
        print(f"  closest any foot body ever gets to the surface: {st['min_gap']:+.3f} m")
        print(f"  old: {spans(st['old_any'])}")
        print(f"  new: {spans(st['new_any'])}   (threshold {threshold} m)")

    if args.census:
        print_sweep(stats)
        dead = [n for n, st in stats.items() if not st["ever_touches"]]
        print(f"\ncensus only -- nothing written. At {threshold} m the guard would refuse "
              f"{len(dead)} clip(s): {', '.join(dead) or 'none'}")
        return

    dead = [n for n, st in stats.items() if not st["ever_touches"]]
    if dead:
        sys.exit(f"\nFATAL: no foot body ever comes within {threshold} m of the ball surface in: "
                 f"{', '.join(dead)}\n  Relabelling would produce an all-free clip and delete the "
                 f"contact supervision entirely. Drop the clip, fix the recon, or raise "
                 f"--threshold deliberately (see --census's sweep).")

    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".pt":
            shutil.copy2(f, dst / f.name)
    others = [i for i in range(52) if i not in body_ids]
    for f in clips:
        t = torch.load(f, map_location="cpu", weights_only=False).detach()
        out, touching = relabel(t, radius[f.name], threshold, args.smooth,
                                args.keep_contact_obj, body_ids, args.free_value)
        assert torch.equal(out[:, I_BODY], t[:, I_BODY]), "positions changed!"
        assert torch.equal(out[:, I_OBJP], t[:, I_OBJP]), "object moved!"
        assert torch.equal(out[:, I_CONTACT_HUMAN][:, others],
                           t[:, I_CONTACT_HUMAN][:, others]), "non-foot flags changed!"
        torch.save(out, dst / f.name)
        st = stats[f.name]
        print(f"\n{f.name}: wrote {dst / f.name}")
        print(f"  foot-contact frames {st['claimed_contact_frames']} -> {int(touching.any(dim=1).sum())}")
    print(f"\ndone -> {dst} (flags only; positions, hands and every non-foot body untouched)")


if __name__ == "__main__":
    main()

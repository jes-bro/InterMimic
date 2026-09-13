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
index), the body-order check looks for the ankle/toe names, AND THE GAP IS
MEASURED FROM THE FOOT'S COLLISION SURFACE, not from the body origin.

WHY THE SURFACE. The hand script measures joint ORIGIN to ball surface, which
works because reconstructed grips put finger joints 8-11 cm inside the ball. A
kicking foot does not penetrate: its surface touches, and the ankle origin
(~8 cm above the sole) and toe origin (inside the shoe) stay 5-12 cm away.
Measured on the first 50 soccer clips (2026-09-12), origin-to-surface gaps on
recon-labelled contact frames had median +12 cm and no threshold separated
them from free frames. So this script reads each foot body's collision geoms
(box / capsule / sphere: type, size, offset, orientation) from the rig, poses
them with the clip's per-body rotations, and takes the distance from the ball
centre to the nearest geom surface minus the ball radius -- the quantity the
simulator's contacts are actually decided on. 2 cm then means for feet what
it means for hands.

THE THRESHOLD IS STILL CHOSEN FROM DATA. --census prints a SWEEP: at each
candidate threshold, the fraction of frames the recon itself labelled
foot-contact that fall inside it, vs the fraction of recon-free frames that do.
Note the recon's own labels over-claim: the upstream converter flags contact
whenever the ball is airborne and not falling and then marks the nearest body,
so "claimed" frames include the ball rising after a kick. Pick the value where
the two populations separate, then write with an explicit --threshold.

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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree                                   # noqa: E402
from relabel_contact_human import (I_BODY, I_OBJP, I_CONTACT_OBJ, I_CONTACT_HUMAN,  # noqa: E402
                                   spans, majority_smooth, radius_from_mesh)

I_BODY_ROT = slice(383, 591)          # 52 x 4 per-body quaternions, xyzw (rotate_pt.py:32)
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


def _floats(s, n=None):
    v = np.array([float(x) for x in s.split()], dtype=np.float64)
    if n is not None and v.shape[0] != n:
        raise ValueError(f"expected {n} numbers, got {s!r}")
    return v


def foot_geoms(mjcf):
    """{foot body name: [geom, ...]} read from the rig, each geom a dict with
    'type' (box|capsule|sphere), 'pos' (3,), 'R' (3,3) rotation of the geom in
    the body frame, and its size: box half-extents (3,), capsule (radius,
    half_length) along the geom's local z, sphere radius. MJCF quats are wxyz;
    capsules may be given as fromto instead of pos/size."""
    root = ET.parse(mjcf).getroot()
    out = {}
    for b in root.iter("body"):
        if b.get("name") not in FOOT_BODY_NAMES:
            continue
        geoms = []
        for g in b.findall("geom"):
            gtype = g.get("type", "sphere")
            if gtype not in ("box", "capsule", "sphere"):
                raise SystemExit(f"FATAL: {mjcf} {b.get('name')}: geom type {gtype!r} not supported")
            q = _floats(g.get("quat", "1 0 0 0"), 4)                    # wxyz
            R = sRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # -> xyzw
            size = _floats(g.get("size", "0"))
            pos = _floats(g.get("pos", "0 0 0"), 3)
            if gtype == "capsule" and g.get("fromto"):
                ft = _floats(g.get("fromto"), 6)
                a, c = ft[:3], ft[3:]
                pos = (a + c) / 2
                axis = c - a
                half = np.linalg.norm(axis) / 2
                z = axis / (2 * half) if half > 0 else np.array([0, 0, 1.0])
                # rotation taking local z onto the segment axis
                R = sRot.align_vectors([z], [[0, 0, 1.0]])[0].as_matrix()
                size = np.array([size[0], half])
            geoms.append(dict(type=gtype, pos=pos, R=R, size=size))
        if not geoms:
            raise SystemExit(f"FATAL: {mjcf} {b.get('name')} has no collision geom")
        out[b.get("name")] = geoms
    missing = [n for n in FOOT_BODY_NAMES if n not in out]
    if missing:
        raise SystemExit(f"FATAL: {mjcf} lacks foot bodies {missing}")
    return out


def _point_to_geom(q, g):
    """Signed distance (T,) from points q (T,3), already in the GEOM frame, to the
    geom surface. Negative = inside."""
    if g["type"] == "sphere":
        return np.linalg.norm(q, axis=1) - g["size"][0]
    if g["type"] == "box":
        half = g["size"][:3]
        d = np.abs(q) - half
        outside = np.linalg.norm(np.maximum(d, 0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0)            # <= 0 when inside
        return outside + inside
    r, h = g["size"][0], g["size"][1]                          # capsule along z
    axis_pt = np.zeros_like(q)
    axis_pt[:, 2] = np.clip(q[:, 2], -h, h)                    # nearest point ON the axis segment
    return np.linalg.norm(q - axis_pt, axis=1) - r


def surface_gap(t, ball_radius, body_ids, geoms=None):
    """[T, len(body_ids)] distance from each foot body's collision SURFACE to the
    ball SURFACE (negative = interpenetrating). With geoms=None (tests only), the
    body origin is used instead."""
    T = t.shape[0]
    bp = t[:, I_BODY].view(T, 52, 3).numpy().astype(np.float64)
    obj = t[:, I_OBJP].numpy().astype(np.float64)
    if geoms is None:
        return torch.tensor(np.linalg.norm(bp[:, body_ids, :] - obj[:, None, :], axis=-1) - ball_radius)
    br = t[:, I_BODY_ROT].view(T, 52, 4).numpy().astype(np.float64)   # xyzw
    cols = []
    for name, b in zip(FOOT_BODY_NAMES, body_ids):
        Rb = sRot.from_quat(br[:, b]).as_matrix()                     # (T,3,3) body->world
        local = np.einsum("tji,tj->ti", Rb, obj - bp[:, b])          # world -> body frame
        best = None
        for g in geoms[name]:
            q = (local - g["pos"]) @ g["R"]                            # body -> geom frame
            d = _point_to_geom(q, g)
            best = d if best is None else np.minimum(best, d)
        cols.append(best)
    return torch.tensor(np.stack(cols, axis=1) - ball_radius)


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


IMPULSE_MIN = 0.5        # m/s: smallest frame-to-frame ball speed change we call a kick
FPS = 30.0


def kick_impulse(t, ball_radius, per_frame_gap):
    """Is there a KICK in this clip, and was the foot there when it happened?

    A kick is an impulse: the ball's velocity jumps between two frames. A floor
    bounce is also an impulse, so jumps that happen with the ball at floor
    height and its vertical velocity flipping sign are set aside. Returns the
    largest remaining jump, its frame, the ball speed before/after, and the
    foot-surface gap at that frame (min over the frame and its neighbours).
    Verdicts: 'no-kick' (no jump >= IMPULSE_MIN), else 'kick' with the gap for
    the caller to judge."""
    obj = t[:, I_OBJP].numpy().astype(np.float64)
    T = obj.shape[0]
    if T < 3:
        return dict(verdict="no-kick", jump=0.0, frame=-1, v_before=0.0, v_after=0.0, gap_at=float("nan"), ball_z=float("nan"))
    v = (obj[1:] - obj[:-1]) * FPS                                  # (T-1, 3) velocity per interval
    dv = np.linalg.norm(v[1:] - v[:-1], axis=1)                     # (T-2,) jump at frame i+1
    z = obj[1:-1, 2]
    bounce = (z < ball_radius + 0.03) & (v[:-1, 2] < 0) & (v[1:, 2] > 0)
    dv_kick = np.where(bounce, 0.0, dv)
    i = int(np.argmax(dv_kick))
    jump = float(dv_kick[i])
    frame = i + 1
    if jump < IMPULSE_MIN:
        return dict(verdict="no-kick", jump=jump, frame=frame,
                    v_before=float(np.linalg.norm(v[i])), v_after=float(np.linalg.norm(v[i + 1])),
                    gap_at=float(per_frame_gap[max(0, frame - 1):frame + 2].min()), ball_z=float(obj[frame, 2]))
    gap_at = float(per_frame_gap[max(0, frame - 1):frame + 2].min())
    return dict(verdict="kick", jump=jump, frame=frame,
                v_before=float(np.linalg.norm(v[i])), v_after=float(np.linalg.norm(v[i + 1])),
                gap_at=gap_at, ball_z=float(obj[frame, 2]))


def census(t, ball_radius, threshold, body_ids, geoms=None):
    gap = surface_gap(t, ball_radius, body_ids, geoms)
    ch = t[:, I_CONTACT_HUMAN][:, body_ids]
    old_any = (ch > 0.1).any(dim=1)
    new_any = (gap < threshold).any(dim=1)
    contact_v, clear_v, vals = observed_levels(t, body_ids)
    per_frame_min = gap.min(dim=1).values
    kick = kick_impulse(t, ball_radius, per_frame_min.numpy())
    return {
        "kick": kick,
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
            free_value="minimal", geoms=None):
    out = t.clone()
    gap = surface_gap(t, ball_radius, body_ids, geoms)
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
          f"{len(free)} recon-free frames (closest foot SURFACE to the ball surface):")
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
    geoms = foot_geoms(args.mjcf)
    print("foot collision geoms from the rig: " + "; ".join(
        f"{n}: " + ", ".join(f"{g['type']} size={np.round(g['size'], 3).tolist()}" for g in geoms[n])
        for n in FOOT_BODY_NAMES))
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
        st = census(t, radius[f.name], threshold, body_ids, geoms)
        stats[f.name] = st
        print(f"\n{f.name}: {st['frames']} frames  (ball radius {radius[f.name]:.4f} m)")
        print(f"  foot contact_human values present: {st['distinct_values']}")
        print(f"  frames claiming foot contact (source): {st['claimed_contact_frames']}")
        print(f"  of those, UNEARNABLE at 0 cm (no foot surface touching the ball): {st['unearnable_frames']}")
        print(f"  closest any foot SURFACE ever gets to the ball surface: {st['min_gap']:+.3f} m")
        k = st["kick"]
        if k["verdict"] == "no-kick":
            print(f"  kick: NONE (largest non-bounce ball velocity jump {k['jump']:.2f} m/s < {IMPULSE_MIN})")
        else:
            verdict = "KICK-OK" if k["gap_at"] < threshold else "KICK-MISS"
            print(f"  kick: frame {k['frame']}, ball {k['v_before']:.1f} -> {k['v_after']:.1f} m/s "
                  f"(jump {k['jump']:.1f}), ball z {k['ball_z']:.2f} m, foot gap there {100 * k['gap_at']:+.1f} cm "
                  f"-> {verdict}")
        print(f"  old: {spans(st['old_any'])}")
        print(f"  new: {spans(st['new_any'])}   (threshold {threshold} m)")

    if args.census:
        print_sweep(stats)
        groups = {"KICK-OK": [], "KICK-MISS": [], "NO-KICK": []}
        for n, st in stats.items():
            k = st["kick"]
            key = "NO-KICK" if k["verdict"] == "no-kick" else ("KICK-OK" if k["gap_at"] < threshold else "KICK-MISS")
            groups[key].append((n, k))
        print(f"\nKICK CHECK at threshold {threshold} m (largest non-bounce ball velocity jump per clip, "
              f"foot gap at that frame):")
        print(f"  KICK-OK   {len(groups['KICK-OK']):>2}  impulse present, foot on the ball when it happens")
        print(f"  KICK-MISS {len(groups['KICK-MISS']):>2}  impulse present, foot NOT on the ball: "
              f"the ball was kicked by nothing -- body and ball tracks contradict")
        print(f"  NO-KICK   {len(groups['NO-KICK']):>2}  no impulse >= {IMPULSE_MIN} m/s: the section contains no kick")
        for key in ("KICK-MISS", "NO-KICK"):
            for n, k in sorted(groups[key], key=lambda x: -x[1]["gap_at"] if x[1]["gap_at"] == x[1]["gap_at"] else 0):
                print(f"    {key:<9} {n:<40} jump {k['jump']:>4.1f} m/s  gap {100 * k['gap_at']:>+6.1f} cm")
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
                                args.keep_contact_obj, body_ids, args.free_value, geoms)
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

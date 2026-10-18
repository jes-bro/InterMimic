#!/usr/bin/env python3
"""Per-clip OBJECT-FLIGHT statistics, the candidate signal for a per-clip reward exponent.

Jess, 2026-10-17: the reward exponent must be a FUNCTION of the clip, with OMOMO and
soccer graded at (or near) product and basketball at the 4th root. Motion magnitude
cannot do it (OMOMO walking and basketball overlap at ~1 m of key-body std). What
makes basketball different is that the ball is out of anyone's control most of the
time -- in the AIR between hand and floor -- which is exactly the case the root was
introduced for (reward_shape.py: every factor poor at once). An OMOMO object is held
or resting; a soccer ball mostly rolls on the ground, where its pose is easy.

Per clip, from the 591-channel reference (obj_pos 318:321, contact_obj 330):
  free_share      share of frames with NO human-object contact (contact_obj ~ 0)
  ground_z        the object's resting height: 5th percentile of its z over the clip
  airborne_share  share of frames with no contact AND z - ground_z > --air-height
                  (default 0.15 m): the object is in flight, controlled by nobody
  air_run_s       longest unbroken airborne run, seconds (a dribble bounce ~0.3 s,
                  a shot ~1 s, a rolling ball 0)
  key_std         key-body motion std (utils/motion_scale.py), for comparison

A candidate exponent is reported alongside, e = max(1/N, 1 - airborne_share / A_REF)
with A_REF the --air-ref (default 0.6: a clip airborne 60% of the time is graded at the
4th root, OMOMO at product). The constant is a PLACEHOLDER until the three datasets
have been measured; the point of this script is to pick it from their distributions.

    python3 scripts/clip_object_flight_stats.py --motion-dir InterAct/OMOMO_new --sources sub2 sub6 [--top 10]
    python3 scripts/clip_object_flight_stats.py --motion-dir InterAct/behave_cari4d_bball7_cf2 --csv bball.csv
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

FPS = 30.0
STEM = re.compile(r"^(sub\d+)_([a-z0-9]+)_(\d+)$")
N_FACTORS = 4
I_BODY = slice(162, 318)
I_OBJ_Z = 320
I_CONTACT_OBJ = 330

# the reward's keyBodies in InterMimic's 52-body order (scripts/motion_feedback.py BODY_NAMES)
_BODY = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
         "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Index1", "L_Index2", "L_Index3",
         "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3",
         "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Index1", "R_Index2",
         "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2",
         "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3"]
_KEY = ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest",
        "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"]
KEY_IDS = [_BODY.index(k) for k in _KEY]


def load_clip(path):
    """(T, 591) float64 from a .pt clip."""
    import torch
    x = torch.load(path, map_location="cpu", weights_only=False)
    x = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 591:
        raise SystemExit(f"{path}: expected (T, 591), got {tuple(x.shape)}")
    return x.astype(np.float64)


def key_body_std(body_pos_flat):
    """Pooled temporal std of the key bodies, metres (= utils/motion_scale.key_body_std)."""
    kp = body_pos_flat.reshape(len(body_pos_flat), -1, 3)[:, KEY_IDS]
    dev = kp - kp.mean(axis=0, keepdims=True)
    return float(np.sqrt((dev ** 2).sum(-1).mean()))


def candidate_exponent(airborne_share, air_ref, n_factors=N_FACTORS):
    """e = max(1/N, 1 - airborne_share / air_ref): product for a grounded/held object,
    the 1/N root once the object is airborne air_ref of the time."""
    return float(max(1.0 / n_factors, 1.0 - airborne_share / air_ref))


def flight_stats(x, air_height=0.15, contact_thres=0.1, fps=FPS, air_ref=0.6):
    """The per-clip numbers from a (T, 591) clip array."""
    T = len(x)
    free = np.abs(x[:, I_CONTACT_OBJ]) < contact_thres
    z = x[:, I_OBJ_Z]
    ground = float(np.percentile(z, 5))
    airborne = free & (z - ground > air_height)
    longest = run = 0
    for a in airborne:
        run = run + 1 if a else 0
        longest = max(longest, run)
    air_share = float(airborne.mean()) if T else 0.0
    return {
        "frames": T,
        "free_share": float(free.mean()) if T else 0.0,
        "ground_z": ground,
        "airborne_share": air_share,
        "air_run_s": longest / fps,
        "max_height": float((z - ground).max()) if T else 0.0,
        "key_std": key_body_std(x[:, I_BODY]),
        "exponent": candidate_exponent(air_share, air_ref),
    }


def collect(motion_dir, objects=None, sources=None, **kw):
    rows = []
    for f in sorted(os.listdir(motion_dir)):
        if not f.endswith(".pt"):
            continue
        m = STEM.match(f[:-3])
        if not m:
            continue
        src, obj, _ = m.groups()
        if objects and obj not in objects:
            continue
        if sources and src not in sources:
            continue
        s = flight_stats(load_clip(os.path.join(motion_dir, f)), **kw)
        s.update(clip=f[:-3], source=src, object=obj)
        rows.append(s)
    return rows


def table(rows):
    out = [f"{'clip':26s} {'frames':>6s} {'free%':>6s} {'air%':>6s} {'air_run_s':>9s} {'max_h_m':>7s} {'key_std':>7s} {'e':>5s}"]
    for r in rows:
        out.append(f"{r['clip']:26s} {r['frames']:6d} {100 * r['free_share']:6.1f} {100 * r['airborne_share']:6.1f} "
                   f"{r['air_run_s']:9.2f} {r['max_height']:7.2f} {r['key_std']:7.3f} {r['exponent']:5.2f}")
    return "\n".join(out)


def summary(rows, air_ref):
    a = np.array([r["airborne_share"] for r in rows]); s = np.array([r["key_std"] for r in rows])
    e = np.array([r["exponent"] for r in rows])
    return (f"{len(rows)} clips | airborne share: median {100 * np.median(a):.1f}%  p90 {100 * np.percentile(a, 90):.1f}%  "
            f"max {100 * a.max():.1f}% | key_std median {np.median(s):.3f} m | "
            f"candidate e (air_ref {air_ref}): product {np.mean(e >= 1):.0%}, <= 0.5 {np.mean(e <= 0.5):.0%}, "
            f"at the floor {np.mean(e <= 1 / N_FACTORS + 1e-9):.0%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--air-height", type=float, default=0.15, help="object this far above its resting height = airborne (m)")
    ap.add_argument("--air-ref", type=float, default=0.6, help="airborne share at which the candidate exponent reaches 1/N")
    ap.add_argument("--sort", default="airborne_share", choices=["airborne_share", "free_share", "air_run_s", "key_std", "frames"])
    ap.add_argument("--top", type=int, default=0, help="print only the top N after sorting (0 = all)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = collect(a.motion_dir, a.objects, a.sources, air_height=a.air_height, air_ref=a.air_ref)
    if not rows:
        raise SystemExit("no clips matched")
    rows.sort(key=lambda r: r[a.sort], reverse=True)
    print(summary(rows, a.air_ref))
    print(table(rows[:a.top] if a.top else rows))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clip", "source", "object", "frames", "free_share", "ground_z",
                                               "airborne_share", "air_run_s", "max_height", "key_std", "exponent"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    sys.exit(main())

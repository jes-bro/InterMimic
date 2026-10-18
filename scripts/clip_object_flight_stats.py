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
  air_height_m    mean height above rest over the AIRBORNE frames (0 if never airborne)
  flight_m        airborne_share x air_height_m: metres of uncontrolled flight per
                  frame, averaged over the clip. Measured 2026-10-17: airborne share
                  alone does NOT separate soccer (median 30%, p90 69%) from basketball
                  (median 47%, p90 62%), but a kicked ball skims the ground (peaks
                  mostly < 1 m) while a basketball flies at hand height or to the hoop
                  (peaks 1.7-3 m), so the height-weighted share is the candidate.
  key_std         key-body motion std (utils/motion_scale.py), for comparison

Two candidate exponents are reported, each max(1/N, 1 - stat / ref):
  e_air     stat = airborne_share, ref = --air-ref (default 0.6)
  e_flight  stat = flight_m,       ref = --flight-ref (default 0.5 m)
The constants are PLACEHOLDERS until the three datasets have been measured; the point
of this script is to pick the statistic and its constant from their distributions.
OMOMO_new (13 trained sources, 3356 clips): airborne share median 0%, p90 0%; the 4%
tail is objects raised with no contact flag (resting on a table / missed labels).

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
FIELDS = ["clip", "source", "object", "frames", "free_share", "ground_z", "airborne_share", "air_run_s",
          "max_height", "air_height_m", "flight_m", "key_std", "e_air", "e_flight"]


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


def candidate_exponent(stat, ref, n_factors=N_FACTORS):
    """e = max(1/N, 1 - stat / ref): product at stat 0 (a held or resting object), the 1/N
    root once the statistic reaches ref."""
    return float(max(1.0 / n_factors, 1.0 - stat / ref))


def flight_stats(x, air_height=0.15, contact_thres=0.1, fps=FPS, air_ref=0.6, flight_ref=0.5):
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
    air_h = float((z - ground)[airborne].mean()) if airborne.any() else 0.0
    flight = air_share * air_h
    return {
        "frames": T,
        "free_share": float(free.mean()) if T else 0.0,
        "ground_z": ground,
        "airborne_share": air_share,
        "air_run_s": longest / fps,
        "max_height": float((z - ground).max()) if T else 0.0,
        "air_height_m": air_h,
        "flight_m": flight,
        "key_std": key_body_std(x[:, I_BODY]),
        "e_air": candidate_exponent(air_share, air_ref),
        "e_flight": candidate_exponent(flight, flight_ref),
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
    out = [f"{'clip':30s} {'frames':>6s} {'free%':>6s} {'air%':>6s} {'run_s':>6s} {'max_h':>6s} {'air_h':>6s} "
           f"{'flight':>7s} {'kstd':>6s} {'e_air':>5s} {'e_fl':>5s}"]
    for r in rows:
        out.append(f"{r['clip']:30s} {r['frames']:6d} {100 * r['free_share']:6.1f} {100 * r['airborne_share']:6.1f} "
                   f"{r['air_run_s']:6.2f} {r['max_height']:6.2f} {r['air_height_m']:6.2f} {r['flight_m']:7.3f} "
                   f"{r['key_std']:6.3f} {r['e_air']:5.2f} {r['e_flight']:5.2f}")
    return "\n".join(out)


def summary(rows, air_ref, flight_ref):
    def col(k):
        return np.array([r[k] for r in rows])
    def dist(v, pct=False):
        m, u, d = (100, "%", 0) if pct else (1, "", 3)
        return (f"median {m * np.median(v):.{d}f}{u}  p10 {m * np.percentile(v, 10):.{d}f}{u}  "
                f"p90 {m * np.percentile(v, 90):.{d}f}{u}  max {m * v.max():.{d}f}{u}")
    def shares(e):
        return f"product {np.mean(e >= 1):.0%}, <= 0.5 {np.mean(e <= 0.5):.0%}, at the floor {np.mean(e <= 1 / N_FACTORS + 1e-9):.0%}"
    return (f"{len(rows)} clips\n"
            f"  airborne share : {dist(col('airborne_share'), True)}\n"
            f"  air height (m) : {dist(col('air_height_m'))}\n"
            f"  flight (m)     : {dist(col('flight_m'))}\n"
            f"  key_std (m)    : {dist(col('key_std'))}\n"
            f"  e_air    (air_ref {air_ref}):       {shares(col('e_air'))}\n"
            f"  e_flight (flight_ref {flight_ref}): {shares(col('e_flight'))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--air-height", type=float, default=0.15, help="object this far above its resting height = airborne (m)")
    ap.add_argument("--air-ref", type=float, default=0.6, help="airborne share at which e_air reaches 1/N")
    ap.add_argument("--flight-ref", type=float, default=0.5, help="flight metres at which e_flight reaches 1/N")
    ap.add_argument("--sort", default="flight_m",
                    choices=["flight_m", "airborne_share", "air_height_m", "free_share", "air_run_s", "key_std", "frames"])
    ap.add_argument("--top", type=int, default=0, help="print only the top N after sorting (0 = all)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = collect(a.motion_dir, a.objects, a.sources, air_height=a.air_height, air_ref=a.air_ref, flight_ref=a.flight_ref)
    if not rows:
        raise SystemExit("no clips matched")
    rows.sort(key=lambda r: r[a.sort], reverse=True)
    print(summary(rows, a.air_ref, a.flight_ref))
    print(table(rows[:a.top] if a.top else rows))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in FIELDS})
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    sys.exit(main())

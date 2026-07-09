#!/usr/bin/env python3
"""Profile source motions by difficulty-relevant properties.

Reads InterMimic 591-channel .pt motion clips and extracts, per clip, the
properties that actually drive InterMimic difficulty -- above all the CONTACT
STRUCTURE (sustained grip is easy; release / free-flight is hard, because the
object goes ballistic and the policy can't control it).

Channel layout (from intermimic.py):
    obj_pos      = hoi_data[:, 318:321]
    contact_obj  = round(hoi_data[:, 330])     # 1 = object in contact, 0 = free

Per clip it reports:
    duration, object
    contact_frac     fraction of frames the object is held        (high = easy)
    n_freeflight     # of release/free-flight spans
    max_ff_sec       longest continuous free-flight (s)            (long = hard)
    obj_speed        mean object translation speed (m/s)           (dynamics)
    obj_vert_range   vertical travel of the object (m)             (lift extent)

Usage:
    python scripts/profile_source_motions.py --subject sub9 \
        --data-dir ~/new_one/OMOMO_new
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch


def zero_runs(free):
    runs, c = [], 0
    for v in free:
        if v:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return runs


def profile_clip(path, fps=30):
    x = torch.load(path, map_location="cpu", weights_only=False).detach().float().numpy()
    T = x.shape[0]
    obj_pos = x[:, 318:321]
    contact = np.round(x[:, 330])
    ff = zero_runs(contact < 0.5)
    speed = float(np.linalg.norm(np.diff(obj_pos, axis=0), axis=1).mean() * fps) if T > 1 else 0.0
    vert = float(obj_pos[:, 2].max() - obj_pos[:, 2].min())
    return dict(T=T, sec=T / fps, contact_frac=float(contact.mean()),
                n_ff=len(ff), max_ff_sec=(max(ff) / fps if ff else 0.0),
                obj_speed=speed, obj_vert_range=vert)


def difficulty(p):
    # sustained grip = easy; low contact or long release = hard
    if p["contact_frac"] >= 0.85 and p["max_ff_sec"] < 0.4:
        return "easy (sustained)"
    if p["contact_frac"] < 0.6 or p["max_ff_sec"] >= 1.0:
        return "hard (release/free-flight)"
    return "medium (intermittent)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="e.g. sub9")
    ap.add_argument("--data-dir", default="~/new_one/OMOMO_new")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--top", type=int, default=8, help="show N hardest/easiest clips")
    args = ap.parse_args()

    d = os.path.expanduser(args.data_dir)
    files = sorted(glob.glob(f"{d}/{args.subject}_*.pt"))
    if not files:
        raise SystemExit(f"no clips for {args.subject} in {d}")

    rows = []
    for f in files:
        p = profile_clip(f, args.fps)
        p["obj"] = os.path.basename(f).split(f"{args.subject}_")[1].rsplit("_", 1)[0]
        p["clip"] = os.path.basename(f)
        p["diff"] = difficulty(p)
        rows.append(p)

    print(f"\n=== {args.subject}: {len(rows)} clips ===")
    from collections import Counter
    dc = Counter(r["diff"] for r in rows)
    for k in ["easy (sustained)", "medium (intermittent)", "hard (release/free-flight)"]:
        print(f"  {k:32} {dc.get(k,0):3}  ({100*dc.get(k,0)/len(rows):.0f}%)")

    print(f"\n--- per OBJECT (means), sorted by contact_frac (easy->hard) ---")
    byo = defaultdict(list)
    for r in rows:
        byo[r["obj"]].append(r)
    print(f"{'object':13} {'n':>3} {'sec':>5} {'contact%':>8} {'#ff':>4} {'maxFF_s':>7} {'objSpd':>6} {'vert_m':>6}")
    for o in sorted(byo, key=lambda o: -np.mean([r["contact_frac"] for r in byo[o]])):
        a = byo[o]
        print(f"{o:13} {len(a):>3} {np.mean([r['sec'] for r in a]):>5.1f} "
              f"{100*np.mean([r['contact_frac'] for r in a]):>7.0f}% {np.mean([r['n_ff'] for r in a]):>4.1f} "
              f"{np.mean([r['max_ff_sec'] for r in a]):>7.2f} {np.mean([r['obj_speed'] for r in a]):>6.2f} "
              f"{np.mean([r['obj_vert_range'] for r in a]):>6.2f}")

    rows.sort(key=lambda r: (r["contact_frac"], -r["max_ff_sec"]))
    print(f"\n--- {args.top} HARDEST clips (least contact / longest free-flight) ---")
    for r in rows[:args.top]:
        print(f"  {r['clip']:26} contact={100*r['contact_frac']:3.0f}% maxFF={r['max_ff_sec']:.2f}s "
              f"spd={r['obj_speed']:.2f} [{r['diff']}]")
    print(f"\n--- {args.top} EASIEST clips (most sustained contact) ---")
    for r in rows[-args.top:][::-1]:
        print(f"  {r['clip']:26} contact={100*r['contact_frac']:3.0f}% maxFF={r['max_ff_sec']:.2f}s "
              f"spd={r['obj_speed']:.2f} [{r['diff']}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How much WALKING each clip contains, from the reference motion alone.

Jess, 2026-10-16: the skating / "feet don't lift" read came from ONE pinned clip
(sub6_woodchair_000, later 059). Before treating it as a general problem, pick
other clips to render -- ones that actually contain walking, so a render can
show whether the policy walks, and a couple with little walking as contrast.

Per clip, from the root position channel (cols 0:3 of the 591-channel layout):
  frames        clip length (30 fps)
  path_m        horizontal path length of the pelvis (sum of per-frame steps)
  net_m         straight-line horizontal displacement, first frame to last
  walk_share    share of frames with pelvis horizontal speed > --walk-speed
                (default 0.3 m/s: a slow walk; standing/reaching is ~0.1)
  walk_s        those frames in seconds
  longest_walk_s  longest unbroken run of walking frames, in seconds -- the
                thing a 50-frame (1.67 s) window does or does not contain

Speed is from a 5-frame central difference (mocap jitter otherwise inflates the
share). Filters: --objects, --sources (clip stems are sub<N>_<object>_<idx>).

    python3 scripts/clip_walk_stats.py --motion-dir ~/new_one/OMOMO_new \\
        --objects woodchair whitechair --sources sub2 sub3 sub5 sub6 sub7 sub12 sub14 sub15 sub17 \\
        --top 15 [--csv out.csv]
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

FPS = 30.0
STEM = re.compile(r"^(sub\d+)_([a-z]+)_(\d+)$")


def root_xy(path):
    """(T, 2) pelvis x,y from a 591-channel .pt clip."""
    import torch
    x = torch.load(path, map_location="cpu", weights_only=False)
    x = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 591:
        raise SystemExit(f"{path}: expected (T, 591), got {tuple(x.shape)}")
    return x[:, 0:2].astype(np.float64)


def walk_stats(xy, walk_speed=0.3, fps=FPS, half_window=2):
    """The per-clip numbers from a (T, 2) horizontal pelvis track."""
    T = len(xy)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if T > 1 else np.zeros(0)
    path = float(steps.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0])) if T > 1 else 0.0
    # central difference over +-half_window frames, clipped at the ends
    i = np.arange(T)
    lo, hi = np.clip(i - half_window, 0, T - 1), np.clip(i + half_window, 0, T - 1)
    dt = np.maximum(hi - lo, 1) / fps
    speed = np.linalg.norm(xy[hi] - xy[lo], axis=1) / dt if T > 1 else np.zeros(T)
    walking = speed > walk_speed
    # longest unbroken run of walking frames
    longest = run = 0
    for w in walking:
        run = run + 1 if w else 0
        longest = max(longest, run)
    return {
        "frames": T,
        "path_m": path,
        "net_m": net,
        "walk_share": float(walking.mean()) if T else 0.0,
        "walk_s": float(walking.sum() / fps),
        "longest_walk_s": longest / fps,
        "peak_speed": float(speed.max()) if T else 0.0,
    }


def collect(motion_dir, objects=None, sources=None, walk_speed=0.3):
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
        s = walk_stats(root_xy(os.path.join(motion_dir, f)), walk_speed)
        s.update(clip=f[:-3], source=src, object=obj)
        rows.append(s)
    return rows


def table(rows):
    hdr = f"{'clip':24s} {'frames':>6s} {'path_m':>7s} {'net_m':>6s} {'walk%':>6s} {'walk_s':>6s} {'longest_s':>9s} {'peak_m/s':>8s}"
    out = [hdr]
    for r in rows:
        out.append(f"{r['clip']:24s} {r['frames']:6d} {r['path_m']:7.2f} {r['net_m']:6.2f} "
                   f"{100 * r['walk_share']:6.1f} {r['walk_s']:6.1f} {r['longest_walk_s']:9.1f} {r['peak_speed']:8.2f}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--walk-speed", type=float, default=0.3, help="pelvis horizontal speed that counts as walking (m/s)")
    ap.add_argument("--sort", default="walk_s", choices=["walk_s", "walk_share", "path_m", "net_m", "longest_walk_s", "frames"])
    ap.add_argument("--top", type=int, default=0, help="print only the top N after sorting (0 = all)")
    ap.add_argument("--bottom", type=int, default=0, help="also print the bottom N (least walking)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = collect(a.motion_dir, a.objects, a.sources, a.walk_speed)
    if not rows:
        raise SystemExit("no clips matched")
    rows.sort(key=lambda r: r[a.sort], reverse=True)
    n = len(rows)
    ws = np.array([r["walk_share"] for r in rows])
    print(f"{n} clips; walking frames (> {a.walk_speed} m/s): median {100 * np.median(ws):.1f}%, "
          f"clips with >= 3 s of walking: {sum(r['walk_s'] >= 3 for r in rows)}, "
          f"with a walk longer than a 50-frame window (1.67 s): {sum(r['longest_walk_s'] > 50 / FPS for r in rows)}")
    shown = rows[:a.top] if a.top else rows
    print(table(shown))
    if a.bottom:
        print(f"... least walking ({a.bottom}):")
        print(table(rows[-a.bottom:]))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clip", "source", "object", "frames", "path_m", "net_m",
                                               "walk_share", "walk_s", "longest_walk_s", "peak_speed"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    sys.exit(main())

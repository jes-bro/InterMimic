#!/usr/bin/env python3
"""Add bodies to a heights json, computed the same way the existing entries were.

WHY. bodyNormalizedReward divides the reward by the subject's height, and
intermimic.py builds that map as SUBJECT_HEIGHTS updated with the arm's
subjectHeightsFile -- then does heights_map[int(sub[3:])] for every body in
subjectBodies. A body with no entry is a hard KeyError at startup, which is what
every zero-shot pair hit ("KeyError: 300", 140/140 rows crashed).

HOW. Same function as scripts/generate_synthetic_bodies.py: the SMPL-X NEUTRAL
T-pose mesh built from the body's betas, height = max(y) - min(y). Using the same
computation matters -- a height measured some other way (MJCF joint extent, say)
would put these bodies on a slightly different scale than every body already in
the file, and the normalization would silently differ between them.

ADDITIVE by design: existing keys are never changed, only new ones added, so the
arms already using the file are unaffected and their runs stay reproducible. A
key that already exists with a DIFFERENT value is refused rather than overwritten.

    python3 scripts/add_body_heights.py --betas scripts/hodome_subject_betas.npz \\
        --heights scripts/synthetic_heights.json
    python3 scripts/add_body_heights.py --betas scripts/hodome_subject_betas.npz \\
        --heights scripts/synthetic_heights.json --dry-run

NOTE for non-neutral betas: BEHAVE's fits are SMPL-H, so heights computed from
them through the SMPL-X neutral model are approximate. Fine for a reward
normalizer (it scales a distance), but say so rather than implying a measurement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def neutral_height(models_dir, betas):
    """T-pose SMPL-X NEUTRAL mesh height in metres (copy of
    generate_synthetic_bodies.neutral_height, kept identical on purpose)."""
    z = np.load(Path(models_dir) / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    sd = z["shapedirs"][:, :, :len(betas)].astype(np.float64)
    V = z["v_template"].astype(np.float64) + np.einsum("vni,i->vn", sd, betas)
    return float(V[:, 1].max() - V[:, 1].min())          # SMPL-X up = +y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", required=True, help="npz of {sub<N>: betas}")
    ap.add_argument("--heights", required=True, help="json to extend")
    ap.add_argument("--models-dir", default=str(Path.home() / "Downloads" / "models" / "smplx"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    betas = np.load(a.betas, allow_pickle=True)
    heights = json.load(open(a.heights)) if Path(a.heights).exists() else {}

    added, kept, conflict = {}, [], []
    for k in sorted(betas.files):
        if not k.startswith("sub"):
            continue                                      # _genders / _source
        n = k[3:]
        h = round(neutral_height(a.models_dir, np.asarray(betas[k], dtype=np.float64)), 4)
        if n in heights:
            (kept if abs(heights[n] - h) < 1e-3 else conflict).append((n, heights[n], h))
            continue
        added[n] = h
        print(f"  {k}: {h:.4f} m")

    if conflict:
        print("\nERROR: these bodies already have a DIFFERENT height in "
              f"{a.heights} -- refusing to overwrite (a run used those values):",
              file=sys.stderr)
        for n, old, new in conflict:
            print(f"    {n}: {old} vs {new}", file=sys.stderr)
        return 2
    if kept:
        print(f"  ({len(kept)} already present with the same value, left alone)")
    if not added:
        print("nothing to add")
        return 0
    if a.dry_run:
        print(f"\n(dry run) would add {len(added)} bodies to {a.heights}")
        return 0

    heights.update(added)
    with open(a.heights, "w") as fh:
        json.dump({k: heights[k] for k in sorted(heights, key=int)}, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {a.heights}: {len(added)} added, {len(heights)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())

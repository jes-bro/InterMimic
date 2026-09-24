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


def mesh_height(models_dir, betas, gender="neutral"):
    """T-pose SMPL-X mesh height in metres, built with the subject's OWN model.

    GENDER MATTERS, and getting it wrong is silent. generate_per_subject_mjcfs
    builds each body with its gendered model, so a height computed from
    SMPLX_NEUTRAL describes a different body. Measured on the 8 BEHAVE bodies:
    against the MJCFs the gendered heights correlate r=1.00 (offset 0.064 +/-
    0.003 m, the mesh-vs-capsule gap), the neutral ones r=0.21 (offset 0.128 +/-
    0.13) -- with the shortest body computing as the tallest. The synthetic
    bodies and HODome are neutral fits, so nothing there changes.
    """
    f = Path(models_dir) / f"SMPLX_{gender.strip().upper()}.npz"
    if not f.is_file():
        f = Path(models_dir) / "SMPLX_NEUTRAL.npz"
    z = np.load(f, allow_pickle=True)
    sd = z["shapedirs"][:, :, :len(betas)].astype(np.float64)
    V = z["v_template"].astype(np.float64) + np.einsum("vni,i->vn", sd, betas)
    return float(V[:, 1].max() - V[:, 1].min())          # SMPL-X up = +y


def neutral_height(models_dir, betas):                   # back-compat
    return mesh_height(models_dir, betas, "neutral")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", required=True, help="npz of {sub<N>: betas}")
    ap.add_argument("--heights", required=True, help="json to extend")
    ap.add_argument("--models-dir", default=str(Path.home() / "Downloads" / "models" / "smplx"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing entries. Needed once, to correct the 8 "
                         "BEHAVE heights written with the neutral model before "
                         "gender was handled; otherwise leave it off, since a "
                         "changed height changes what a finished run meant")
    a = ap.parse_args()

    betas = np.load(a.betas, allow_pickle=True)
    genders = {}
    for e in np.atleast_1d(betas["_genders"]) if "_genders" in betas.files else []:
        t = (e.decode() if isinstance(e, bytes) else str(e))
        if ":" in t:
            k, v = t.split(":", 1)
            genders[k] = v
    heights = json.load(open(a.heights)) if Path(a.heights).exists() else {}

    added, kept, conflict = {}, [], []
    for k in sorted(betas.files):
        if not k.startswith("sub"):
            continue                                      # _genders / _source
        n = k[3:]
        g = genders.get(k, "neutral")
        h = round(mesh_height(a.models_dir, np.asarray(betas[k], dtype=np.float64), g), 4)
        if n in heights and not a.overwrite:
            (kept if abs(heights[n] - h) < 1e-3 else conflict).append((n, heights[n], h))
            continue
        added[n] = h
        print(f"  {k} ({g}): {h:.4f} m"
              + (f"   [was {heights[n]}]" if n in heights and heights[n] != h else ""))

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

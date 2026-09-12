#!/usr/bin/env python3
"""One body per person: average each subject's per-clip SMPL-H betas.

CARI4D fits the 10 shape parameters per CLIP. interact2mimic writes the
subject's MJCF from whatever betas the sequence carries, and computes the
clip's joint positions with the same rig. Converting a subject's clips one by
one with their own betas therefore leaves the LAST clip's body on disk and
every earlier clip's positions computed against a rig that no longer exists.

This averages the betas over all of a subject's clips (measured spread on the
2026-09-12 set: 0.16-0.26 L2 within a person, vs 0.4-1.5 between people) and
writes one vector per subject. cari4d_to_interact.py --betas-npz then uses it
for every clip of that subject, so one MJCF is written repeatedly with
identical content and all clips share its bone lengths.

    python3 scripts/cari4d_subject_betas.py --manifest ~/cari4d_bball7/manifest.csv \\
        --bundles-root ~/cari4d_bball7 --out scripts/bball7_subject_betas.npz
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cari4d_to_interact import _load_bundle  # noqa: E402


def subject_means(manifest, bundles_root):
    """{'sub401': {'mean': (10,), 'n_clips': int, 'spread': float, 'clips': [...]}}"""
    per = {}
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            b = _load_bundle(Path(bundles_root) / r["bundle"])["pr"]["betas"]
            b = b.detach().cpu().numpy()
            if b.ndim != 2 or b.shape[1] != 10:
                raise SystemExit(f"{r['clip']}: betas {b.shape}, expected (T, 10)")
            per.setdefault(f"sub{r['subject_id']}", []).append((r["clip"], b.mean(0)))
    out = {}
    for s, items in per.items():
        arr = np.stack([v for _, v in items])
        mean = arr.mean(0)
        spread = float(np.mean(np.linalg.norm(arr - mean, axis=1))) if len(arr) > 1 else 0.0
        out[s] = dict(mean=mean, n_clips=len(arr), spread=spread, clips=[c for c, _ in items])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--bundles-root", required=True)
    p.add_argument("--out", required=True, help=".npz with one (10,) array per sub<id> key")
    a = p.parse_args(argv)

    stats = subject_means(a.manifest, os.path.expanduser(a.bundles_root))
    print(f"{'subject':>7} {'clips':>5} {'spread':>7}  betas[0:4]")
    for s in sorted(stats, key=lambda k: int(k[3:])):
        st = stats[s]
        print(f"{s:>7} {st['n_clips']:>5} {st['spread']:>7.3f}  {np.round(st['mean'][:4], 3)}")
    np.savez(a.out, **{s: st["mean"].astype(np.float32) for s, st in stats.items()})
    print(f"wrote {a.out} ({len(stats)} subjects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

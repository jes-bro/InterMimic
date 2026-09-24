#!/usr/bin/env python3
"""HODome subjects -> a betas npz that generate_per_subject_mjcfs.py can read.

WHY. The zero-shot body test scores our policies on people from a DATASET THEY
WERE NEVER TRAINED ON. To put a HODome person in the simulator we need an MJCF
built from that person's shape, and the MJCF generator reads a npz of the OMOMO
form:

    {'sub<N>': betas (16,) float32, ..., '_genders': ['sub<N>:male', ...]}

HODome ships one .npz per (subject, object) sequence -- subject01_box.npz etc --
with the shape parameters inside each. Shape is a property of the PERSON, not the
sequence, so any one sequence per subject is enough; with several we average and
report the spread, which is also the check that they really are one body (a large
spread means the fits disagree and the "person" we build is nobody).

    python3 scripts/hodome_subject_betas.py --inspect ~/Downloads/hodome_smplx/smplx/subject01_box.npz
    python3 scripts/hodome_subject_betas.py --src ~/Downloads/hodome_smplx/smplx \\
        --out scripts/hodome_subject_betas.npz --first-id 300

--inspect prints the keys, shapes and dtypes of one file and exits: run it FIRST,
because the key names below are an assumption until a real file confirms them.

SUBJECT IDS. HODome's subject01..10 become sub<first-id + n>, default 300, so
they cannot collide with OMOMO (1-17), synthetics (100+) or CARI4D (204, 401+).
The mapping is written into the npz as '_source' so a body id can always be
traced back to the person it came from.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Key names seen in SMPL-X/SMPL-H sequence fits, most specific first. Checked
# against the file rather than assumed -- --inspect prints what is actually there.
BETAS_KEYS = ["betas", "shape", "shape_params", "smplx_betas", "person_betas"]
GENDER_KEYS = ["gender", "sex", "meta_gender"]


def inspect(path):
    d = np.load(path, allow_pickle=True)
    print(f"{path}:")
    for k in d.files:
        v = d[k]
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        head = np.array2string(v.ravel()[:6], precision=3) if getattr(v, "size", 0) else str(v)
        print(f"  {k:24s} shape={str(shape):14s} dtype={str(dtype):10s} {head}")
    return 0


def _pick(d, candidates, what, path):
    for k in candidates:
        if k in d.files:
            return k
    raise SystemExit(
        f"ERROR: none of {candidates} in {path} (has: {list(d.files)}). "
        f"Run --inspect and tell the script the real {what} key.")


def betas_of(path):
    """(betas (16,), gender str or None) for one sequence file."""
    d = np.load(path, allow_pickle=True)
    b = np.asarray(d[_pick(d, BETAS_KEYS, "betas", path)], dtype=np.float32)
    # A per-frame fit is (T, n); shape is constant per person, so collapse it and
    # say so rather than silently taking frame 0.
    if b.ndim == 2:
        b = b.mean(axis=0)
    b = b.ravel()
    gender = None
    for k in GENDER_KEYS:
        if k in d.files:
            gender = str(d[k]).strip().lower().strip("[]'\" ")
            break
    return b, gender


def subject_files(src):
    """{'subject01': [paths...]} from names like subject01_box.npz."""
    per = {}
    for p in sorted(Path(src).glob("*.npz")):
        m = re.match(r"(subject\d+)_", p.name)
        if not m:
            continue
        per.setdefault(m.group(1), []).append(p)
    if not per:
        raise SystemExit(f"ERROR: no subject*_*.npz under {src}")
    return per


def build(src, first_id, genders_override=None, n_betas=16):
    per = subject_files(src)
    out, meta, source = {}, [], []
    for i, (subj, paths) in enumerate(sorted(per.items())):
        vecs, gender = [], None
        for p in paths:
            b, g = betas_of(p)
            vecs.append(b)
            gender = gender or g
        V = np.stack(vecs)
        mean = V.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(V - mean, axis=1))) if len(V) > 1 else 0.0
        # The generator wants 16; a 10-vector fit is zero-padded (the extra
        # coefficients are genuinely zero for that fit, not unknown).
        if mean.shape[0] < n_betas:
            mean = np.concatenate([mean, np.zeros(n_betas - mean.shape[0], np.float32)])
        elif mean.shape[0] > n_betas:
            mean = mean[:n_betas]
        sub_id = f"sub{first_id + i}"
        out[sub_id] = mean.astype(np.float32)
        g = (genders_override or {}).get(subj) or gender or "neutral"
        meta.append(f"{sub_id}:{g}")
        source.append(f"{sub_id}:{subj}:{len(paths)}seq:spread{spread:.3f}")
        print(f"  {sub_id} <- {subj}  {len(paths)} seq  |betas|={np.linalg.norm(mean):.2f}  "
              f"spread={spread:.3f}  gender={g}")
    return out, meta, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", metavar="NPZ", help="print one file's keys and exit")
    ap.add_argument("--src", help="directory of HODome subject*_*.npz files")
    ap.add_argument("--out", default="scripts/hodome_subject_betas.npz")
    ap.add_argument("--first-id", type=int, default=300,
                    help="subject01 becomes sub<first-id>; default 300 keeps HODome "
                         "clear of OMOMO (1-17), synthetics (100+) and CARI4D (204, 401+)")
    ap.add_argument("--gender", nargs="*", default=[],
                    help="subject01=female ... when the npz carries no gender")
    a = ap.parse_args()

    if a.inspect:
        return inspect(a.inspect)
    if not a.src:
        ap.error("--src (or --inspect)")

    overrides = dict(kv.split("=", 1) for kv in a.gender)
    out, meta, source = build(a.src, a.first_id, overrides)
    np.savez(a.out, _genders=np.array(meta), _source=np.array(source), **out)
    print(f"\nwrote {a.out}: {len(out)} bodies")
    print("next (on the cluster, where the SMPL-X models are):")
    print(f"  python3 scripts/generate_per_subject_mjcfs.py --betas-npz {a.out} "
          f"--subjects {' '.join(sorted(out))} --dataset-tag omomo")
    return 0


if __name__ == "__main__":
    sys.exit(main())

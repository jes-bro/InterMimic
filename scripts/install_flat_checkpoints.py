#!/usr/bin/env python3
"""Install FLAT-named checkpoint files into the checkpoints/<exp>/nn/ tree.

A collaborator exports a run's checkpoint as one file named after the
experiment, e.g. (Google Drive, 2026-09-15):

    smplx_teacher_g3_omomo_geoall_src11__f0_mimic.pth
    smplx_teacher_g3_omomo_geoall__f0_mimic_00030000.pth

collect_g3_teachers.py (and every launcher's resume block) reads
checkpoints/<exp>/nn/mimic.pth | mimic_<n>.pth, so each file goes to

    <root>/smplx_teacher_g3_omomo_geoall_src11__f0/nn/mimic.pth
    <root>/smplx_teacher_g3_omomo_geoall__f0/nn/mimic_00030000.pth

The split point is the LAST "_mimic" in the name; a name without it is
refused. Use --root to stage into a fresh tree (recommended on the laptop:
the real checkpoints/ is 735 GB and the upload glob should not have to skip
it) and --move instead of copy when the source is a download you do not need
twice. Never overwrites an existing destination file.

    python3 scripts/install_flat_checkpoints.py ~/Downloads/smplx_teacher_g3_*.pth \\
        --root ~/gcp_stage/checkpoints --move
"""
import argparse
import os
import re
import shutil
import sys

NAME = re.compile(r"^(?P<exp>.+)_(?P<file>mimic(?:_\d+)?\.pth)$")


def split_name(basename):
    """'<exp>_mimic[_NNNN].pth' -> (exp, 'mimic[_NNNN].pth'), or None."""
    m = NAME.match(basename)
    return (m.group("exp"), m.group("file")) if m else None


def plan(paths):
    out, bad = [], []
    for p in paths:
        s = split_name(os.path.basename(p))
        (out if s else bad).append((p, *s) if s else p)
    if bad:
        raise SystemExit("ERROR: not '<exp>_mimic[_N].pth' names, refusing to guess: " + ", ".join(bad))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--root", required=True, help="checkpoints tree to install into")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    items = plan(a.files)
    for src, exp, fname in items:
        if not os.path.isfile(src):
            raise SystemExit(f"ERROR: not a file: {src}")
        dst = os.path.join(a.root, exp, "nn", fname)
        if os.path.exists(dst):
            raise SystemExit(f"ERROR: {dst} exists -- refusing to overwrite a checkpoint")
        print(f"  {os.path.basename(src)}  ->  {os.path.relpath(dst, a.root)}   ({os.path.getsize(src) / 1048576:.1f} MB)")
        if a.dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        (shutil.move if a.move else shutil.copy2)(src, dst)
    print(f"{'DRY RUN: would install' if a.dry_run else 'installed'} {len(items)} file(s) under {a.root}")
    return items


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect cross-pair teacher checkpoints into a flat directory for distillation.

InterMimic_CrossPair (the distillation env) expects all teacher .pth files
in a single directory, named with the cross-pair slug:

    checkpoints/teachers/crosspair_<object>/
      b10_s2_largetable.pth
      b10_s6_largetable.pth
      b17_s2_largetable.pth
      ...

But trainees save to:

    checkpoints/smplx_crosspair_b10_s2_largetable/nn/mimic.pth
    checkpoints/smplx_crosspair_b10_s2_largetable/nn/mimic_00005000.pth
    ...

This script finds each trained teacher, picks the latest checkpoint (highest
epoch snapshot, falling back to mimic.pth), and creates a symlink in the
collection dir under the cross-pair slug name.

By default, only default-reward teachers are collected. Use --include-normreward
to also collect the body-normalized-reward variants (writes to a separate
subdir so they don't get mixed in with the default-reward distillation set).

Usage from repo root:
    python scripts/collect_crosspair_teachers.py --object largetable
    python scripts/collect_crosspair_teachers.py --object woodchair
    python scripts/collect_crosspair_teachers.py --object largetable --include-normreward
"""
import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO / "checkpoints"
TEACHER_RUN_RE = re.compile(
    r'^smplx_crosspair_b(\d+)_s(\d+)_([^_]+)(_normreward)?$'
)
SNAPSHOT_RE = re.compile(r'^mimic_(\d+)\.pth$')


def find_latest_checkpoint(nn_dir):
    """Return the highest-epoch snapshot in nn_dir, or mimic.pth if no snapshot."""
    if not nn_dir.is_dir():
        return None
    snapshots = []
    for f in nn_dir.iterdir():
        m = SNAPSHOT_RE.match(f.name)
        if m:
            snapshots.append((int(m.group(1)), f))
    if snapshots:
        snapshots.sort()
        return snapshots[-1][1]
    mimic = nn_dir / "mimic.pth"
    if mimic.exists():
        return mimic
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True,
                        help="Object name to collect teachers for. Use a single name "
                             "(largetable, woodchair) for that object's teachers, or "
                             "'both' to collect teachers for ALL objects into one dir.")
    parser.add_argument("--include-normreward", action="store_true",
                        help="Also collect normreward teachers into a separate subdir.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: checkpoints/teachers/crosspair_<object>)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (uses ~50MB per teacher).")
    args = parser.parse_args()

    out_default = args.output_dir and Path(args.output_dir) or \
                  CHECKPOINTS_DIR / "teachers" / f"crosspair_{args.object}"
    out_norm = CHECKPOINTS_DIR / "teachers" / f"crosspair_{args.object}_normreward"

    out_default.mkdir(parents=True, exist_ok=True)
    if args.include_normreward:
        out_norm.mkdir(parents=True, exist_ok=True)

    # Discover all matching teacher run dirs
    default_collected, norm_collected, skipped = [], [], []
    for d in sorted(CHECKPOINTS_DIR.iterdir() if CHECKPOINTS_DIR.is_dir() else []):
        if not d.is_dir():
            continue
        m = TEACHER_RUN_RE.match(d.name)
        if not m:
            continue
        body, source, obj, normreward_tag = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        # --object largetable -> only largetable teachers
        # --object both       -> all object teachers (for multi-object distillation)
        if args.object != "both" and obj != args.object:
            continue
        is_normreward = normreward_tag is not None
        if is_normreward and not args.include_normreward:
            continue

        ckpt = find_latest_checkpoint(d / "nn")
        if ckpt is None:
            skipped.append((d.name, "no checkpoint found in nn/"))
            continue

        slug = f"b{body}_s{source}_{obj}"
        out_dir = out_norm if is_normreward else out_default
        dst = out_dir / f"{slug}.pth"

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copy(ckpt, dst)
        else:
            # Use absolute path for symlink so it works regardless of where the env is invoked from
            dst.symlink_to(ckpt.resolve())

        target_list = norm_collected if is_normreward else default_collected
        target_list.append((slug, ckpt.relative_to(REPO), dst.relative_to(REPO)))

    # Report
    print(f"Default-reward teachers collected: {len(default_collected)}")
    for slug, src, dst in default_collected:
        print(f"  {slug:30s}  {src} -> {dst}")
    if args.include_normreward:
        print(f"\nNormreward teachers collected: {len(norm_collected)}")
        for slug, src, dst in norm_collected:
            print(f"  {slug:30s}  {src} -> {dst}")
    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    print(f"\nOutput dir: {out_default}")
    if args.include_normreward:
        print(f"Normreward output dir: {out_norm}")

    if not default_collected and not (args.include_normreward and norm_collected):
        print(f"\nWARNING: no teachers collected. Check that checkpoints/ has "
              f"smplx_crosspair_*_{args.object}/ dirs with snapshots in nn/.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

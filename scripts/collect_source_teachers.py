#!/usr/bin/env python3
"""Gather trained source-teacher checkpoints into a teacherPolicy dir for distill.

InterMimic_All selects a teacher per env by the SOURCE subject id, parsed from the
checkpoint filename `sub{S}.pth` (intermimic_all.py: subid = int(name[3:])). So we
copy each source-teacher's latest checkpoint into one dir, renamed by its source:

  checkpoints/smplx_teacher_src{S}_xf_aug/nn/mimic.pth  ->  <out>/sub{S}.pth

Usage (from repo root, on the cluster where the checkpoints live):
  python scripts/collect_source_teachers.py --sources 1 2 3 5 6 7 8 9 11 12 14 15 17 \
      --out checkpoints/teachers/source_xf_aug_noheldout
  python scripts/collect_source_teachers.py --sources 1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 \
      --out checkpoints/teachers/source_xf_aug_no14

Fails loudly if any requested source has no checkpoint -- a missing teacher would
silently shrink the ensemble and mis-map the source->teacher index.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def latest_ckpt(exp):
    """Newest numbered snapshot checkpoints/<exp>/nn/mimic_<n>.pth, else mimic.pth."""
    nn = REPO / "checkpoints" / exp / "nn"
    if not nn.is_dir():
        return None
    snaps = []
    for p in nn.glob("mimic_*.pth"):
        m = re.search(r"mimic_(\d+)\.pth$", p.name)
        if m:
            snaps.append((int(m.group(1)), p))
    if snaps:
        return max(snaps)[1]
    mp = nn / "mimic.pth"
    return mp if mp.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=int, nargs="+", required=True,
                    help="source subject ids to include as teachers")
    ap.add_argument("--out", required=True, help="output teacherPolicy dir")
    ap.add_argument("--suffix", default="xf_aug",
                    help="teacher experiment suffix: smplx_teacher_src{S}_{suffix}")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = REPO / a.out
    plan, missing = [], []
    for s in a.sources:
        exp = f"smplx_teacher_src{s}_{a.suffix}"
        ck = latest_ckpt(exp)
        (plan if ck else missing).append((s, exp, ck))

    for s, exp, ck in plan:
        print(f"  sub{s:<3} <- {ck.relative_to(REPO)}")
    if missing:
        print("\nMISSING checkpoints (no teacher trained yet?):", file=sys.stderr)
        for s, exp, _ in missing:
            print(f"  sub{s}: checkpoints/{exp}/nn/ has no mimic*.pth", file=sys.stderr)
        sys.exit(f"\nERROR: {len(missing)} of {len(a.sources)} sources have no checkpoint. "
                 f"Refusing to build a partial teacher set (would mis-map source->teacher).")

    if a.dry_run:
        print(f"\nDRY RUN: would copy {len(plan)} teachers into {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    for s, exp, ck in plan:
        shutil.copy2(ck, out / f"sub{s}.pth")
    print(f"\nwrote {len(plan)} teacher checkpoints -> {out}")


if __name__ == "__main__":
    main()

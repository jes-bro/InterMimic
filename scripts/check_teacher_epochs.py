#!/usr/bin/env python3
"""Inspect every cross-pair teacher checkpoint on the cluster and print
epoch + size for each. Used to spot stale/under-trained teachers
(e.g., an epoch=100 leftover from an early failed run) that would silently
corrupt distillation by providing bad BC supervision.

Run from the InterMimic repo root on the cluster:
    python scripts/check_teacher_epochs.py
"""
import os
import glob
import torch
from pathlib import Path

# Search every plausible teacher dir under checkpoints/ and report.
DIRS = [
    "checkpoints/teachers/crosspair_both",
    "checkpoints/teachers/crosspair_largetable",
    "checkpoints/teachers/crosspair_woodchair",
]

# Heuristic — anything below this is suspicious (under-trained).
MIN_HEALTHY_EPOCH = 1500

print(f"{'epoch':>6}  {'size(MB)':>9}  status   path")
print("-" * 110)
for d in DIRS:
    if not Path(d).exists():
        print(f"{'?':>6}  {'?':>9}  MISSING  {d}/ (directory does not exist)")
        continue
    paths = sorted(glob.glob(f"{d}/*.pth"))
    if not paths:
        print(f"{'?':>6}  {'?':>9}  EMPTY    {d}/")
        continue
    for p in paths:
        try:
            ck = torch.load(p, map_location="cpu")
            epoch = ck.get("epoch", "?")
            size_mb = os.path.getsize(p) / 1e6
            if isinstance(epoch, int) and epoch < MIN_HEALTHY_EPOCH:
                status = "STALE!"
            else:
                status = "ok"
            print(f"{epoch!s:>6}  {size_mb:>9.1f}  {status:<7}  {p}")
        except Exception as e:
            print(f"{'?':>6}  {'?':>9}  FAIL     {p}: {e}")

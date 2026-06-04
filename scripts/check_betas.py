#!/usr/bin/env python3
"""Print body-shape betas for each subject and the pairwise distance
between sub1 and sub5 (the held-out bodies whose renders look similar).

Run from the repo root:
    python scripts/check_betas.py
"""
import numpy as np

b = np.load('scripts/omomo_betas.npz')

print("Per-subject betas norm (larger = more pronounced body shape):")
for k in ['sub2', 'sub3', 'sub9', 'sub10', 'sub17', 'sub1', 'sub5']:
    if k in b:
        n = float(np.linalg.norm(b[k]))
        print(f"  {k}: norm={n:.3f}, first 5 = {b[k][:5]}")
    else:
        print(f"  {k}: NOT in betas npz")

print()
if 'sub1' in b and 'sub5' in b:
    diff = b['sub5'] - b['sub1']
    print(f"sub5 - sub1 diff: norm={float(np.linalg.norm(diff)):.3f}")
    print(f"  raw diff: {diff[:8]}")

print()
print("Pairwise distance between bodies (lower = more similar):")
subs = ['sub2', 'sub3', 'sub9', 'sub10', 'sub17', 'sub1', 'sub5']
subs = [s for s in subs if s in b]
print(f"      {' '.join(f'{s:>6s}' for s in subs)}")
for a in subs:
    row = []
    for c in subs:
        if a == c:
            row.append("  -   ")
        else:
            row.append(f"{float(np.linalg.norm(b[a]-b[c])):6.3f}")
    print(f"  {a:5s} {' '.join(row)}")

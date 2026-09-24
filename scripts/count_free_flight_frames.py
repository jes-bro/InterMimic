#!/usr/bin/env python3
"""How often does rewardTerms.freeFlightGate actually fire on a set of clips?

WHY. The gate neutralises object reward and/or resets on frames where the
REFERENCE has no hand-object contact. Two eval configs that differ only in the
gate are comparable exactly to the extent that the gate never fires -- so before
re-running a sweep to equalise it, measure whether there is anything to equalise.

Contact is channels 331:383 of the 591-channel motion tensor (52 binary
per-body-part flags, scripts/retarget_contact.py:58). A frame is FREE FLIGHT
when all 52 are zero.

    python3 scripts/count_free_flight_frames.py ~/new_one/OMOMO_new
    python3 scripts/count_free_flight_frames.py <dir> --pattern "sub15_*.pt"

Prints the per-clip and overall fraction of free-flight frames. Near 0% means the
gate is inert for these clips and a config with it and one without give the same
answer; a large fraction means the two are not comparable.
"""
import argparse
import sys
from pathlib import Path

import torch

CONTACT = slice(331, 383)          # 52 binary human-contact flags
N_CHANNELS = 591


def free_flight_fraction(path):
    """-> (n_free, n_frames) for one clip. Raises SystemExit on a bad tensor."""
    d = torch.load(str(path), map_location="cpu")
    if d.ndim != 2 or d.shape[-1] != N_CHANNELS:
        raise SystemExit(f"{path.name}: shape {tuple(d.shape)}, want (T, {N_CHANNELS})")
    contact = d[:, CONTACT]
    # >0.5 rather than !=0: these are stored as floats and a retargeted clip can
    # carry tiny non-zero values where the original had a hard 0/1.
    per_frame = (contact > 0.5).any(dim=1)
    return int((~per_frame).sum()), int(d.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="a clip, or a directory of .pt clips")
    ap.add_argument("--pattern", default="*.pt")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    a = ap.parse_args()

    if a.path.is_dir():
        clips = sorted(a.path.glob(a.pattern))
    else:
        clips = [a.path]
    if not clips:
        raise SystemExit(f"no clips matching {a.pattern} under {a.path}")
    if a.limit:
        clips = clips[:a.limit]

    tot_free = tot_frames = 0
    worst = []
    for c in clips:
        free, n = free_flight_fraction(c)
        tot_free += free
        tot_frames += n
        worst.append((free / n if n else 0.0, c.name, free, n))
        if not a.quiet:
            print(f"  {c.name}: {free}/{n} free-flight frames ({100.0 * free / max(n, 1):.1f}%)")

    worst.sort(reverse=True)
    print(f"\n{len(clips)} clip(s): {tot_free}/{tot_frames} frames are free flight "
          f"({100.0 * tot_free / max(tot_frames, 1):.2f}%)")
    print("worst clips:")
    for frac, name, free, n in worst[:5]:
        print(f"  {100.0 * frac:5.1f}%  {name}  ({free}/{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

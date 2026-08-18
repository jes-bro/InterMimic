#!/usr/bin/env python3
"""Drop the first N frames of converted InterMimic motion clips (591-channel
.pt tensors), writing a NEW dataset dir -- the source dir is never touched.

Why: Start-mode init copies reference frame 0 verbatim, so a clip whose first
frames show the ball in free flight spawns a detached falling ball the policy
must catch before anything else. If the pickup happens a few frames in, the
honest fix is to start the segment there ("segments begin on a contact frame"),
not to edit poses. Found on the rectinj3 bball export: frame 0 has the ball
0.72 m from the hand; by frame 4 it is in hand (0.27 m, CONTACT).

Every channel is per-frame (positions, rotations, dofs, contact flags), so a
row slice keeps everything aligned -- nothing is recomputed.

Usage (from the repo root):
    python3 scripts/trim_pt_start.py \
        --src-dir InterAct/behave_cari4d_rectinj3 \
        --dst-dir InterAct/behave_cari4d_rectinj3_t4 \
        --start 4

Verify the result with scripts/inspect_bball_clip.py --clip <dst>/<clip>.pt
(frame 0 should now be CONTACT with a small hand-ball distance).
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="dataset dir with .pt clips")
    ap.add_argument("--dst-dir", required=True,
                    help="output dir; must NOT exist (new-names rule)")
    ap.add_argument("--start", type=int, required=True,
                    help="number of leading frames to drop (new frame 0 = old frame N)")
    args = ap.parse_args()

    src, dst = Path(args.src_dir), Path(args.dst_dir)
    if not src.is_dir():
        sys.exit(f"FATAL: src dir not found: {src}")
    if dst.exists():
        sys.exit(f"FATAL: dst dir already exists: {dst} -- refusing to overwrite; "
                 f"pick a new name")
    if args.start < 1:
        sys.exit("FATAL: --start must be >= 1 (0 would be a pure copy)")

    clips = sorted(src.glob("*.pt"))
    if not clips:
        sys.exit(f"FATAL: no .pt clips in {src}")

    dst.mkdir(parents=True)
    # Copy everything that is not a clip (manifests etc.) unchanged.
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".pt":
            shutil.copy2(f, dst / f.name)
            print(f"  copied  {f.name}")

    for f in clips:
        t = torch.load(f, map_location="cpu")
        if t.ndim != 2:
            sys.exit(f"FATAL: {f.name}: expected a (T, C) tensor, got shape {tuple(t.shape)}")
        if args.start >= t.shape[0]:
            sys.exit(f"FATAL: {f.name}: --start {args.start} >= clip length {t.shape[0]}")
        torch.save(t[args.start:].clone(), dst / f.name)
        print(f"  trimmed {f.name}: {t.shape[0]} -> {t.shape[0] - args.start} frames "
              f"(dropped frames 0-{args.start - 1})")

    print(f"done -> {dst}")


if __name__ == "__main__":
    main()

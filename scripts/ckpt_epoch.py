#!/usr/bin/env python3
"""Print the epoch (and frame count) stored INSIDE rl_games checkpoints.

A file named mimic.pth carries no epoch in its name, and a collaborator's
export may be renamed anyway. The checkpoint dict itself records 'epoch'
(and 'frame'), so read that instead of trusting the filename.

    python3 scripts/ckpt_epoch.py ~/Downloads/smplx_teacher_g3_omomo_geoall_src6__f0_mimic.pth
    python3 scripts/ckpt_epoch.py checkpoints/smplx_teacher_g3_*/nn/mimic.pth
"""
import os
import sys

import torch


def ckpt_epoch(path):
    """(epoch, frame) from the checkpoint dict; None for a key it lacks."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f"{path}: not a checkpoint dict")
    return ck.get("epoch"), ck.get("frame")


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        ep, fr = ckpt_epoch(p)
        mb = os.path.getsize(p) / 1048576
        print(f"epoch {ep!s:>8}  frames {fr!s:>14}  {mb:7.1f} MB  {p}")


if __name__ == "__main__":
    main()

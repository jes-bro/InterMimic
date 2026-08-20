#!/usr/bin/env python3
"""Re-derive contact_obj flags from geometry: contact = nearest wrist within
--threshold of the ball CENTER. Fixes flag mislabels like optj3d frames 20-26
(flagged CONTACT with 0.28-0.70m of air -- the ball still approaching), which
poison the contact-gated reward and any contact-colored visualization.

Threshold default 0.25 m = ball radius (0.13) + a real hand's reach past the
wrist (~0.08-0.10) + slack; it reproduces the clip's true hold/carry spans and
rejects the approach frames. A 3-frame majority filter removes single-frame
flicker. Positions are NEVER touched -- flags only (channel 330).

Writes a NEW dataset dir (trim_pt_start.py conventions: dst must not exist,
non-.pt files copied along).

  python3 scripts/relabel_contact_flags.py \
      --src-dir InterAct/behave_cari4d_optj3d \
      --dst-dir InterAct/behave_cari4d_optj3d_cf \
      --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml
"""
import argparse
import itertools
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402

I_BODY = slice(162, 318)
I_OBJP = slice(318, 321)
I_CONTACT_OBJ = 330


def spans(flags):
    out, i = [], 0
    for k, g in itertools.groupby(list(flags)):
        n = len(list(g))
        out.append(f"{i}-{i+n-1}:{'C' if k else 'f'}")
        i += n
    return " ".join(out)


def relabel(t, wrists, threshold):
    T = t.shape[0]
    bp = t[:, I_BODY].view(T, 52, 3)
    obj = t[:, I_OBJP]
    d = (bp[:, wrists, :] - obj[:, None, :]).norm(dim=-1).min(dim=1).values
    raw = (d < threshold).numpy().astype(int)
    # 3-frame majority filter against single-frame flicker
    sm = raw.copy()
    for i in range(1, T - 1):
        sm[i] = int(raw[i - 1] + raw[i] + raw[i + 1] >= 2)
    return sm, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--mjcf", required=True, help="subject MJCF (wrist indices by name)")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="wrist-to-ball-CENTER contact distance (m)")
    args = ap.parse_args()

    src, dst = Path(args.src_dir), Path(args.dst_dir)
    if not src.is_dir():
        sys.exit(f"FATAL: src dir not found: {src}")
    if dst.exists():
        sys.exit(f"FATAL: dst dir already exists: {dst} -- refusing to overwrite")
    names = [n for n, _, _ in _parse_mjcf_tree(args.mjcf)]
    wrists = [names.index("L_Wrist"), names.index("R_Wrist")]

    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".pt":
            shutil.copy2(f, dst / f.name)

    for f in sorted(src.glob("*.pt")):
        t = torch.load(f, map_location="cpu").clone()
        old = t[:, I_CONTACT_OBJ].numpy().astype(int)
        new, d = relabel(t, wrists, args.threshold)
        t[:, I_CONTACT_OBJ] = torch.tensor(new, dtype=t.dtype)
        torch.save(t, dst / f.name)
        changed = int((old != new).sum())
        print(f"{f.name}: {changed}/{len(old)} frames relabeled "
              f"(threshold {args.threshold}m)")
        print(f"  old: {spans(old)}")
        print(f"  new: {spans(new)}")
    print(f"done -> {dst} (flags only; positions untouched)")


if __name__ == "__main__":
    main()

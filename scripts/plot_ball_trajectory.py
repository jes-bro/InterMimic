#!/usr/bin/env python3
"""Ball-trajectory figure to hold against a CARI4D render: what does the
export's pose_abs ball actually do, and (optionally) the installed clip's?

One row per source (bundle / clip), two panels each:
  left  -- all three coordinates vs frame (the timing/height story)
  right -- the arc: the two highest-variance axes as a path, frames annotated
           every 10 so a video moment can be matched to a point on the curve

The two rows are in DIFFERENT world frames (the conversion rotates/translates),
so compare shapes and timing, never raw coordinates.

  PYTHONPATH=/simurgh2/projects/ret-hoi/CARI4D python3 scripts/plot_ball_trajectory.py \
      --bundle .../Date03_Sub01_bball_dribble.pth \
      --clip InterAct/behave_cari4d_optj3d/sub100_bball_000.pt \
      --out renders/ball_traj_optj3d.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_ball(args):
    rows = []
    if args.bundle:
        b = torch.load(args.bundle, map_location="cpu", weights_only=False)
        pr = b["pr"] if "pr" in b else b
        rows.append(("bundle pose_abs", np.asarray(pr["pose_abs"])[:, :3, 3].astype(float)))
    if args.clip:
        c = torch.load(args.clip, map_location="cpu")
        rows.append(("installed clip obj_pos", c[:, 318:321].double().numpy()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="CARI4D export .pth")
    ap.add_argument("--clip", help="installed 591-channel .pt")
    ap.add_argument("--out", required=True, help="output .png")
    args = ap.parse_args()
    rows = load_ball(args)
    if not rows:
        raise SystemExit("pass --bundle and/or --clip")

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5 * len(rows)), squeeze=False)
    for r, (tag, P) in enumerate(rows):
        T = len(P)
        ax = axes[r][0]
        for i, lbl in enumerate("xyz"):
            ax.plot(range(T), P[:, i], label=lbl)
        ax.set_title(f"{tag}: coordinates vs frame ({T} frames)")
        ax.set_xlabel("frame"); ax.set_ylabel("m"); ax.legend(); ax.grid(alpha=0.3)

        # arc panel: two highest-variance axes
        var_order = np.argsort(P.var(0))[::-1][:2]
        a, b_ = sorted(var_order)
        ax = axes[r][1]
        ax.plot(P[:, a], P[:, b_], "-", lw=1.5)
        for f in range(0, T, 10):
            ax.annotate(str(f), (P[f, a], P[f, b_]), fontsize=7)
        ax.scatter([P[0, a]], [P[0, b_]], marker="o", label="start")
        ax.scatter([P[-1, a]], [P[-1, b_]], marker="x", label="end")
        ax.set_title(f"{tag}: arc in axes {'xyz'[a]}-{'xyz'[b_]} (frame numbers annotated)")
        ax.set_xlabel(f"{'xyz'[a]} (m)"); ax.set_ylabel(f"{'xyz'[b_]} (m)")
        ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

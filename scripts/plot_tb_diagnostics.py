#!/usr/bin/env python3
"""Optimizer diagnostics from rl_games TensorBoard logs: last_lr, KL, clip_frac.

Companion to plot_tb_rewards.py (which shows WHAT the reward did); this shows
WHY -- the three signals that explain a slow or unstable run:

  info/last_lr    what the adaptive-LR controller actually set. A run pinned at
                  its kl_threshold gets throttled here (normval_adlr sat at
                  7.6e-6, 7.6x below its own peak -- the mini_epochs=3 arm exists
                  because of this plot).
  info/kl         where policy updates sit relative to kl_threshold.
  info/clip_frac  fraction of the batch hitting the PPO clip. Healthy ~0.1-0.2;
                  ~0.5 means later inner epochs are mostly being clipped away.

X-axis is the logged-point index, NOT frames: these scalars are logged once per
epoch, but their TB step field is the frame counter, which RESTARTS on every
resume -- merging segments on it interleaves garbage. Segments are instead
concatenated in wall-clock order, so the x-axis reads as "epochs, in order",
exact enough for diagnosing trends. (plot_tb_rewards rebuilds true frames via
each reward point's paired iter tag; these tags have no such pair.)

  python3 scripts/plot_tb_diagnostics.py \
      --run "normval+adlr=~/Downloads/tb_aug04/smplx_teacher_src2_xf_aug_normval_adlr" \
      --out ~/Downloads/diag.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fixed-order palette as the other plot scripts -- one design system.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"

PANELS = [
    ("info/last_lr",   "learning rate (adaptive controller output)", True),
    ("info/kl",        "policy KL per update",                       True),
    ("info/clip_frac", "PPO clip fraction (healthy ~0.1-0.2)",       False),
]


def load_tag(run_dir, tag):
    """Concatenate a tag across a run's event files in wall-clock order.
    Empty/aborted segment files are skipped; a run with NO segment carrying the
    tag returns an empty array (plotted as absent, reported on stderr)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    segs = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(run_dir),
                                           "events.out.tfevents*"))):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags()["scalars"]:
            continue
        ev = acc.Scalars(tag)
        if ev:
            segs.append((ev[0].wall_time, np.array([e.value for e in ev])))
    segs.sort(key=lambda s: s[0])
    return np.concatenate([v for _, v in segs]) if segs else np.array([])


def ema(v, a=0.97):
    if len(v) == 0:
        return v
    out, m = np.empty_like(v), v[0]
    for i, x in enumerate(v):
        m = a * m + (1 - a) * x
        out[i] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"FATAL: --run wants LABEL=DIR, got {spec!r}")
        runs.append((label, d))
    if len(runs) > len(SERIES):
        raise SystemExit(f"FATAL: {len(runs)} runs, {len(SERIES)} palette slots -- "
                         f"cycling hues would make two runs share a colour")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 3.1 * len(PANELS)),
                             facecolor=SURFACE, sharex=False)
    for ax, (tag, title, logy) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for i, (label, d) in enumerate(runs):
            v = load_tag(d, tag)
            if len(v) == 0:
                print(f"  [diag] {label}: no {tag} in any segment -- omitted from "
                      f"that panel (constant-LR runs log it flat, but absence "
                      f"usually means an aborted-only sync)", file=sys.stderr)
                continue
            ax.plot(ema(v), color=SERIES[i], lw=1.6, label=label, zorder=3)
            ax.plot(v, color=SERIES[i], lw=0.6, alpha=0.18, zorder=2)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10.5, weight="bold", color=INK, loc="left")
        ax.grid(axis="y", color="#e3e1dc", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[-1].set_xlabel("logged points (~epochs, segments in wall-clock order)",
                        color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
                   loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print(f"[diag] wrote {a.out}")


if __name__ == "__main__":
    main()

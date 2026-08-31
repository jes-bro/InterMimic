#!/usr/bin/env python3
"""Reward vs EPOCH, read straight from the slurm .out logs.

plot_tb_rewards.py plots against environment frames and wall time, which is the
fair axis for sample efficiency but not what you want when asking "how far in is
this run". The training loop prints

    epoch_num:17250 mean_rewards:[1.67] fps step: ...

every epoch, so the logs already hold the curve.

Runs are grouped by the name slurm embeds in the filename --
teacher-<run>-<jobid>.out -- so a run resubmitted across several jobs is one
line, not several. Segments are concatenated and sorted by epoch.

TWO THINGS THAT MAKE THESE CURVES LIE, both handled:

  Warm-started runs inherit the teacher's epoch counter. The bball arms open
  around 13,005 rather than 0, so plotting raw epoch numbers puts a fresh run
  13k epochs "behind" one that has trained for the same time. --relative
  subtracts each run's own first epoch.

  mean_rewards is per-epoch and noisy. The default smoothing is an EWMA; the
  raw trace is drawn faintly behind it so smoothing cannot hide a spike.

    python3 scripts/plot_epoch_rewards.py --glob 'teacher-g2_mlp_*.out' --out g2_mlp.png
    python3 scripts/plot_epoch_rewards.py --glob 'cari4d-bball-r*.out' --relative --out bball.png
    python3 scripts/plot_epoch_rewards.py --run r7=cari4d-bball-r7_geom-*.out \
        --run r8=cari4d-bball-r8_horiz-*.out --out pair.png
"""
import argparse
import collections
import glob as globmod
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

EPOCH_RE = re.compile(r"epoch_num:(\d+)\s+mean_rewards:\[([-\d.eE+]+)\]")
# slurm names logs <prefix>-<run>-<jobid>.out; the jobid is the last dash field.
NAME_RE = re.compile(r"^(?:teacher-|cari4d-bball-)?(.+)-(\d+)\.out$")


def run_name(path):
    m = NAME_RE.match(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def read_log(path):
    """(epochs, rewards) from one log. Empty arrays if it never trained."""
    ep, rw = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                ep.append(int(m.group(1)))
                rw.append(float(m.group(2)))
    return np.asarray(ep), np.asarray(rw)


def ewma(y, alpha):
    """Exponentially weighted mean, alpha = smoothing in [0, 1)."""
    if alpha <= 0 or len(y) < 2:
        return y
    out = np.empty_like(y, dtype=np.float64)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", action="append", default=[],
                   help="shell glob over .out files; runs are grouped by name. "
                        "Repeatable.")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=GLOB",
                   help="explicit label for a glob, overriding the filename. "
                        "Repeatable.")
    p.add_argument("--relative", action="store_true",
                   help="plot epochs since each run's OWN first epoch. Needed "
                        "when comparing warm-started runs (which inherit the "
                        "teacher's counter, ~13,005 for the bball arms) against "
                        "fresh ones (which start at 0).")
    p.add_argument("--smoothing", type=float, default=0.95,
                   help="EWMA factor in [0,1); 0 disables (default 0.95)")
    p.add_argument("--min-epochs", type=int, default=50,
                   help="skip runs with fewer logged epochs than this")
    p.add_argument("--title", default="Reward vs epoch")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if not a.glob and not a.run:
        raise SystemExit("pass --glob or --run")

    groups = collections.OrderedDict()
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run wants LABEL=GLOB, got {spec!r}")
        label, pat = spec.split("=", 1)
        groups.setdefault(label, []).extend(sorted(globmod.glob(pat)))
    for pat in a.glob:
        for f in sorted(globmod.glob(pat)):
            groups.setdefault(run_name(f), []).append(f)

    if not groups:
        raise SystemExit("no files matched")

    series, skipped = [], []
    for label, files in groups.items():
        ep, rw = [], []
        for f in files:
            e, r = read_log(f)
            ep.append(e); rw.append(r)
        ep = np.concatenate(ep) if ep else np.array([])
        rw = np.concatenate(rw) if rw else np.array([])
        if len(ep) < a.min_epochs:
            skipped.append((label, len(ep)))
            continue
        # A resubmitted run's segments overlap and arrive out of order; sort by
        # epoch and keep the LAST value at each, which is the resumed one.
        order = np.argsort(ep, kind="stable")
        ep, rw = ep[order], rw[order]
        _, keep = np.unique(ep[::-1], return_index=True)
        keep = len(ep) - 1 - keep
        ep, rw = ep[np.sort(keep)], rw[np.sort(keep)]
        series.append((label, ep, rw, len(files)))

    if not series:
        raise SystemExit("every run was below --min-epochs; nothing to plot")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    print(f"{'run':<40} {'epochs':>16} {'final':>8} {'best':>8}  jobs")
    print("-" * 82)
    for i, (label, ep, rw, njobs) in enumerate(series):
        x = ep - ep[0] if a.relative else ep
        c = cmap(i % 10)
        ax.plot(x, rw, color=c, alpha=0.15, linewidth=0.8)          # raw, faint
        ax.plot(x, ewma(rw, a.smoothing), color=c, linewidth=1.8, label=label)
        print(f"{label:<40} {ep[0]:>7d}-{ep[-1]:<8d} {rw[-1]:>8.3f} "
              f"{rw.max():>8.3f}  {njobs}")
    if skipped:
        print("\nskipped (too few epochs logged):")
        for label, n in skipped:
            print(f"  {label}: {n}")

    ax.set_xlabel("epochs since this run's first" if a.relative else "epoch_num")
    ax.set_ylabel("mean_rewards")
    ax.set_title(a.title)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    if not a.relative and any(s[1][0] > 1000 for s in series):
        ax.text(0.99, 0.02,
                "note: warm-started runs inherit the teacher's epoch counter; "
                "use --relative to compare fairly",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, style="italic", alpha=0.7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

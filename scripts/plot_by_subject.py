#!/usr/bin/env python3
"""Break out eval performance by TARGET subject (body), across runs.

Complements plot_teacher_evals.py (whose headline is body-GROUP means) and
plot_curriculum_evals.py (whose per-run heatmaps are body x source): here every
figure has the target subject on its own axis so you can ask "who is hard?".

Two CSV families, two figure styles:

  Teacher CSVs (smplx_teacher_*.csv, one source each, from eval_one.sh):
      small multiples -- one panel per target subject, runs as horizontal bars.
      Panels instead of grouped bars because there are >6 runs and the validated
      palette has 6 slots; identity comes from the y labels, not color.
      One figure per metric: teacher_by_subject_<metric>.png
      Default keeps only the LATEST checkpoint per run (logged); pass
      --all-checkpoints to compare a run at several steps (labeled @<step>k).

  Curriculum CSVs (*__full.csv, body x source matrix):
      grouped bars -- x = target subject, one series per run (<=6, fits the
      palette), value = mean over that body's evaluated sources.
      One 3-panel figure: curriculum_by_subject.png

Missing (body, run) cells are drawn as a GAP labeled n/r, never as 0.

  python3 scripts/plot_by_subject.py --in ~/Downloads/latestresultsmorefinishedaug8 \
      --out ~/Downloads/latestresultsmorefinishedaug8/plots_by_subject
"""
import argparse
import csv
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Design tokens (shared verbatim with plot_teacher_evals.py / dataviz palette).
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e1dc"
BAR_HUE = SERIES[0]                       # single hue for the small multiples
GROUP_ORDER = ["In-distribution", "Held-out", "Synthetic"]
BAND_EDGE = {"In-distribution": "#5b6b7a", "Held-out": "#b4524f", "Synthetic": "#6a5fa8"}
HELDOUT = {"sub10", "sub16", "sub13"}     # matches eval_one.sh / plot_teacher_evals.py

METRICS = [
    ("success_rate",      "Success rate (%)",     True),   # higher is better
    ("human_pose_error",  "Human pose error (m)", False),
    ("object_pose_error", "Object pose error (m)", False),
]


def body_group(body):
    n = int(body[3:])
    if n >= 100:
        return "Synthetic"
    return "Held-out" if body in HELDOUT else "In-distribution"


def ordered_bodies(bodies):
    return sorted(bodies, key=lambda b: (GROUP_ORDER.index(body_group(b)), int(b[3:])))


# ------------------------------------------------------------------ loading
def _read(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"FATAL: {path} has no data rows")
    # A pair whose eval CRASHED is written with empty metrics + exit_code=1.
    # Drop it (loudly) so one dead pair doesn't kill the whole figure; it will
    # surface as n/r (or a mean over fewer sources) rather than a fake number.
    kept = []
    for r in rows:
        if any(r[k] == "" for k in ("success_rate", "human_pose_error",
                                    "object_pose_error")):
            print(f"[read] {os.path.basename(path)}: dropping FAILED pair "
                  f"(body={r['body']}, source={r['source']}, exit_code={r['exit_code']})")
            continue
        for k in ("success_rate", "human_pose_error", "object_pose_error"):
            r[k] = float(r[k])
        kept.append(r)
    if not kept:
        raise SystemExit(f"FATAL: {path} has no rows with metrics (all pairs failed?)")
    return kept


def load_teacher(in_dir, all_checkpoints=False):
    """smplx_teacher_*.csv -> {label: {rows, step, source}}.

    Run + step come from the checkpoint column (authoritative), not the file
    name. Same run+step twice (e.g. an __all6 and a partial re-eval) keeps the
    fuller CSV; older checkpoints of a run are dropped unless --all-checkpoints.
    Everything dropped is printed, so a thinner chart is never a silent one.
    """
    by_run_step = {}                       # (run, step) -> (rows, source, path)
    for p in sorted(glob.glob(os.path.join(in_dir, "smplx_teacher_*.csv"))):
        rows = _read(p)
        ckpt = rows[0]["checkpoint"]
        run = ckpt.split("/")[1].replace("smplx_teacher_", "")
        step = int(os.path.basename(ckpt).split("_")[-1].split(".")[0])
        key = (run, step)
        if key in by_run_step:
            keep, drop = max(by_run_step[key], (rows, rows[0]["source"], p),
                             key=lambda t: len(t[0])), None
            drop = p if keep[2] != p else by_run_step[key][2]
            print(f"[teacher] {run}@{step}: two CSVs, keeping the fuller "
                  f"({len(keep[0])} rows: {os.path.basename(keep[2])}), "
                  f"dropping {os.path.basename(drop)}")
            by_run_step[key] = keep
        else:
            by_run_step[key] = (rows, rows[0]["source"], p)

    if not all_checkpoints:
        latest = {}
        for (run, step) in by_run_step:
            latest[run] = max(latest.get(run, step), step)
        for (run, step) in sorted(by_run_step):
            if step != latest[run]:
                print(f"[teacher] {run}: dropping step {step:,} "
                      f"(latest is {latest[run]:,}; --all-checkpoints keeps both)")
        by_run_step = {k: v for k, v in by_run_step.items() if k[1] == latest[k[0]]}

    dupes = {run for (run, _) in by_run_step
             if sum(1 for (r, _) in by_run_step if r == run) > 1}
    labels = {key: (f"{key[0]}@{key[1] // 1000}k" if key[0] in dupes else key[0])
              for key in by_run_step}
    # Two steps of a run inside the same 1000-step bucket would share an @<N>k
    # label and one would silently vanish -- fall back to exact-step labels.
    counts = {}
    for lab in labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    for key, lab in labels.items():
        if counts[lab] > 1:
            labels[key] = f"{key[0]}@{key[1]}"
    out = {}
    for key, (rows, source, _p) in sorted(by_run_step.items()):
        out[labels[key]] = {"rows": rows, "step": key[1], "source": source}
    return out


def load_curriculum(in_dir):
    """*__full.csv -> {run: rows} (body x source matrices)."""
    return {os.path.basename(p).replace("__full.csv", ""): _read(p)
            for p in sorted(glob.glob(os.path.join(in_dir, "*__full.csv")))}


def per_body(rows, metric):
    """{body: mean over that body's rows}. Teacher CSVs have one row per body,
    so this is the value itself; curriculum matrices average over sources."""
    vals = {}
    for r in rows:
        vals.setdefault(r["body"], []).append(r[metric])
    return {b: float(np.mean(v)) for b, v in vals.items()}


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def fig_teacher(runs, metric, label, hib, out):
    """One panel per target subject; runs as horizontal bars, single hue."""
    names = list(runs)                                    # fixed order everywhere
    per = {run: per_body(d["rows"], metric) for run, d in runs.items()}
    bodies = ordered_bodies({b for v in per.values() for b in v})

    ncols = 4
    nrows = math.ceil(len(bodies) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(16, 1.4 + nrows * (0.55 + 0.24 * len(names))),
                             facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    # Shared x-scale so panels compare; errors get a common max across all cells.
    xmax = 100.0 if metric == "success_rate" else \
        1.05 * max(v for p in per.values() for v in p.values())

    y = np.arange(len(names))
    for k, body in enumerate(bodies):
        ax = axes[k // ncols][k % ncols]
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        vals = [per[run].get(body, np.nan) for run in names]
        ax.barh(y, [0 if np.isnan(v) else v for v in vals], 0.72, color=BAR_HUE,
                zorder=3, edgecolor=SURFACE, linewidth=0.8)
        for yi, v in zip(y, vals):
            if np.isnan(v):
                ax.text(xmax * 0.02, yi, "n/r", va="center", fontsize=6.5,
                        color=INK3, style="italic")
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(names) - 0.4, -0.6)               # first run on top
        ax.set_yticks(y)
        # Run names once per row of panels (leftmost) -- identity by label.
        ax.set_yticklabels(names if k % ncols == 0 else [], fontsize=7.5)
        g = body_group(body)
        # Short group tags so the rightmost column's title fits inside the figure.
        tag = {"In-distribution": "in-dist", "Held-out": "held-out",
               "Synthetic": "synthetic"}[g]
        ax.set_title(f"{body}  ·  {tag}", fontsize=9.5, weight="bold",
                     color=BAND_EDGE[g], loc="left", pad=3)
    for k in range(len(bodies), nrows * ncols):           # blank spare cells
        axes[k // ncols][k % ncols].axis("off")

    better = "higher is better" if hib else "lower is better"
    fig.suptitle(f"Teacher {label} by target subject", x=0.5, y=0.995,
                 fontsize=15, weight="bold", color=INK)
    # fig.text, NOT an axes annotation: an annotation anchored in figure coords
    # joins the first panel's tight_layout bbox and opens a blank band up top.
    fig.text(0.01, 0.985, f"Latest checkpoint per run unless labeled @step. "
             f"{better.capitalize()}; n/r = body not evaluated for that run.",
             ha="left", va="top", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[teacher] wrote -> {out}")


def fig_curriculum(runs, out):
    """Grouped bars per target subject, one series per run, 3 stacked metrics."""
    names = list(runs)
    if len(names) > len(SERIES):
        # 6 palette slots is a hard limit (no hue cycling); the caller must trim.
        raise SystemExit(f"FATAL: {len(names)} curriculum runs but the palette has "
                         f"{len(SERIES)} slots -- rerun with a subset via --curriculum-runs")
    per = {m: {run: per_body(rows, m) for run, rows in runs.items()}
           for m, _, _ in METRICS}
    bodies = ordered_bodies({b for p in per["success_rate"].values() for b in p})
    x = np.arange(len(bodies))
    w = 0.8 / len(names)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10.5), facecolor=SURFACE)
    for ax, (metric, label, hib) in zip(axes, METRICS):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for i, run in enumerate(names):
            vals = [per[metric][run].get(b, np.nan) for b in bodies]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.88, color=SERIES[i],
                   zorder=3, edgecolor=SURFACE, linewidth=0.9,
                   label=run if ax is axes[0] else None)
            for xi, v in zip(x, vals):
                if np.isnan(v):
                    ax.text(xi + i * w - 0.4 + w / 2, 0, "n/r", rotation=90,
                            ha="center", va="bottom", fontsize=6.5, color=INK3,
                            style="italic")
        ax.set_ylabel(label, color=INK2, fontsize=10)
        if metric == "success_rate":
            ax.set_ylim(0, 100)
        ax.set_title("higher is better" if hib else "lower is better",
                     fontsize=9, color=INK3, loc="right", pad=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bodies, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
                   loc="lower right", bbox_to_anchor=(1, 1.06))
    fig.suptitle("Curriculum runs by target subject (mean over evaluated sources)",
                 x=0.5, y=0.985, fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[curriculum] wrote -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="keep every checkpoint of a run, labeled @<step>k "
                         "(default: latest per run)")
    ap.add_argument("--curriculum-runs", nargs="*", default=None,
                    help="subset of *__full.csv run names (palette caps at 6)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    teachers = load_teacher(args.in_dir, args.all_checkpoints)
    if teachers:
        for metric, label, hib in METRICS:
            fig_teacher(teachers, metric, label, hib,
                        os.path.join(args.out, f"teacher_by_subject_{metric}.png"))
    else:
        print("[teacher] no smplx_teacher_*.csv found -- skipping")

    curriculum = load_curriculum(args.in_dir)
    if args.curriculum_runs is not None:
        missing = [r for r in args.curriculum_runs if r not in curriculum]
        if missing:
            raise SystemExit(f"FATAL: --curriculum-runs not found: {missing} "
                             f"(have: {sorted(curriculum)})")
        curriculum = {r: curriculum[r] for r in args.curriculum_runs}
    if curriculum:
        fig_curriculum(curriculum, os.path.join(args.out, "curriculum_by_subject.png"))
    else:
        print("[curriculum] no *__full.csv found -- skipping")


if __name__ == "__main__":
    main()

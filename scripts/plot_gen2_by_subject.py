#!/usr/bin/env python3
"""Gen-2 eval results: one panel per fold, vertical bars per OMOMO body.

Each panel is one fold; x is the evaluated body (all real OMOMO subjects except
sub4); each cluster holds one bar per training config (plain/ret x stock/nvadlr).
The bodies that fold HELD OUT are shaded, so in-distribution and generalization
are read off the same axis.

WHY THE FOLD IS A PANEL AND NOT A LABEL. The two folds hold out different
bodies -- f0 tests on {10,13,16}, f1 on {5,7,12} -- and they are mirror images:
each fold's held-out trio is the other's training trio. Putting the fold in the
panel title rather than in floating text over a shared axis makes that
unambiguous and removes a whole class of mislabelling.

sub4 IS ABSENT ON PURPOSE. Its MJCF crashes the simulator, so it has never been
evaluated for any run; a bar at zero would read as "failed the task".

Reads the eval CSVs written by slurm_eval_curriculum.sh / scripts/eval_one.sh.
Run and fold come from the checkpoint column, not the filename, so a renamed
file still works.

    # MLP, all bodies, both folds
    python3 scripts/plot_gen2_by_subject.py --in ~/Downloads/eval_results \
        --include 'g2_mlp_*' --out gen2_mlp.png

    # transformer
    python3 scripts/plot_gen2_by_subject.py --in ~/Downloads/eval_results \
        --include 'g2_xf_*' --out gen2_xf.png

    python3 scripts/plot_gen2_by_subject.py --in DIR --metric success_rate --out sr.png
"""
import argparse
import csv
import fnmatch
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.patches as mpatches     # noqa: E402
import numpy as np                        # noqa: E402

METRICS = ["success_rate", "human_pose_error", "object_pose_error"]
NICE = {"success_rate": "success rate (%)",
        "human_pose_error": "human pose error (m)",
        "object_pose_error": "object pose error (m)"}
# Higher is better for success; lower for the errors. Stated on the axis so a
# tall error bar is never read as a good result.
BETTER = {"success_rate": "higher is better",
          "human_pose_error": "lower is better",
          "object_pose_error": "lower is better"}

# The gen-2 fold design, read off the training configs
# (isaacgym/src/intermimic/data/cfg/omomo_teacher_g2_*__f{0,1}.yaml, key
# env.subjectBodies). Hardcoded rather than inferred: a CSV that happens to
# cover only three bodies is indistinguishable from a fold definition, and
# guessing wrong would mislabel generalization as in-distribution.
FOLD_HELDOUT = {"f0": ("sub10", "sub13", "sub16"),
                "f1": ("sub5", "sub7", "sub12")}


def subnum(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else -1


def read_rows(path):
    """Usable rows from one eval CSV; [] if it holds nothing we can plot.

    A pair whose eval crashed is written with empty metrics and exit_code=1.
    Those are dropped rather than counted, so a dead pair shows as a missing
    bar instead of a zero -- a zero here would read as 'failed the task'.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if any(r.get(m, "") == "" for m in METRICS):
            continue
        for m in METRICS:
            r[m] = float(r[m])
        out.append(r)
    return out


def run_and_fold(ckpt):
    """(config, fold, run_dir) from a checkpoint path.

    e.g. 'collab/jm/checkpointsjm/smplx_teacher_g2_mlp_ret_stock__f1/nn/mimic.pth'
      -> ('mlp_ret_stock', 'f1', 'smplx_teacher_g2_mlp_ret_stock__f1')
    The arch stays in the config label so MLP and transformer runs can share a
    figure without silently collapsing onto each other.
    """
    run = next((x for x in ckpt.split("/") if x.startswith("smplx_")), None)
    if run is None:
        return None, None, None
    short = run.replace("smplx_teacher_", "")
    m = re.search(r"__(f\d)$", short)
    fold = m.group(1) if m else None
    cfg = re.sub(r"__f\d$", "", short)
    cfg = re.sub(r"^g2_", "", cfg)                   # 'g2_mlp_ret_stock' -> 'mlp_ret_stock'
    return cfg, fold, run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--include", default=None, metavar="GLOB",
                   help="only read CSVs whose filename matches, e.g. 'g2_mlp_*'")
    p.add_argument("--metric", choices=METRICS, default=None,
                   help="one metric instead of all three rows of panels")
    p.add_argument("--synthetic", action="store_true",
                   help="also plot the synthetic augmentation bodies (sub100+); "
                        "by default only real OMOMO subjects are shown")
    p.add_argument("--title", default="Gen-2: per-body evaluation by fold")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # (fold, body) -> {cfg: {metric: value}};  (cfg, fold) -> {checkpoint paths}
    data = defaultdict(dict)
    ckpts = defaultdict(set)
    configs, folds = [], []
    n_syn_dropped = 0

    paths = sorted(glob.glob(os.path.join(a.in_dir, "*.csv")))
    if a.include:
        paths = [q for q in paths if fnmatch.fnmatch(os.path.basename(q), a.include)]
    for path in paths:
        rows = read_rows(path)
        if not rows or "checkpoint" not in rows[0]:
            continue
        cfg, fold, run = run_and_fold(rows[0]["checkpoint"])
        if cfg is None or fold is None:
            print(f"[skip] {os.path.basename(path)}: cannot read config/fold "
                  f"from {rows[0]['checkpoint']!r}")
            continue
        if fold not in FOLD_HELDOUT:
            raise SystemExit(
                f"{os.path.basename(path)}: fold {fold!r} is not in the known "
                f"fold design {sorted(FOLD_HELDOUT)}. Add its held-out bodies to "
                f"FOLD_HELDOUT before plotting -- guessing would mislabel "
                f"generalization as in-distribution.")
        if cfg not in configs:
            configs.append(cfg)
        if fold not in folds:
            folds.append(fold)
        ckpts[(cfg, fold)].add(rows[0]["checkpoint"])
        for r in rows:
            body = r["body"]
            if subnum(body) >= 100 and not a.synthetic:
                n_syn_dropped += 1
                continue
            data[(fold, body)][cfg] = {m: r[m] for m in METRICS}

    if not data:
        raise SystemExit(f"no usable eval CSVs in {a.in_dir}"
                         + (f" matching {a.include!r}" if a.include else ""))

    # Two CSVs for the same run at DIFFERENT checkpoints silently overwrite each
    # other -- whichever sorts last wins, which is how a 54.6k-epoch result got
    # replaced by a 27k one. Refuse, and say which files to narrow away.
    mixed = {k: v for k, v in ckpts.items() if len(v) > 1}
    if mixed:
        lines = []
        for (cfg, fold), v in sorted(mixed.items()):
            lines.append(f"  {cfg} {fold}:")
            lines += [f"      {c}" for c in sorted(v)]
        raise SystemExit(
            "more than one checkpoint per run matched --include, so the bars "
            "would mix epochs:\n" + "\n".join(lines) +
            "\n\nNarrow --include to one checkpoint per run.")

    if n_syn_dropped:
        print(f"[note] dropped {n_syn_dropped} synthetic-body rows (sub100+); "
              f"pass --synthetic to include them")

    configs.sort()
    folds.sort()
    bodies = sorted({b for (_, b) in data}, key=subnum)

    # Say what is missing rather than letting a gap pass for a result.
    missing = [(f, b, c) for f in folds for b in bodies for c in configs
               if c not in data.get((f, b), {})]
    if missing:
        print(f"[note] {len(missing)} (fold, body, config) cells have no eval "
              f"and are drawn as gaps:")
        for f, b, c in missing[:12]:
            print(f"        {f} {b} {c}")
        if len(missing) > 12:
            print(f"        ... and {len(missing) - 12} more")

    metrics = [a.metric] if a.metric else METRICS
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11})
    fig, axes = plt.subplots(len(metrics), len(folds),
                             figsize=(max(7.0, 0.52 * len(bodies)) * len(folds),
                                      2.9 * len(metrics)),
                             sharey="row", squeeze=False)
    cmap = plt.get_cmap("tab10")
    width = 0.8 / len(configs)
    xs = np.arange(len(bodies))

    for row, metric in enumerate(metrics):
        for col, fold in enumerate(folds):
            ax = axes[row][col]
            held = set(FOLD_HELDOUT[fold])
            # Shade the held-out bodies FIRST so the bars sit on top of it.
            for j, b in enumerate(bodies):
                if b in held:
                    ax.axvspan(j - 0.5, j + 0.5, color="0.88", zorder=0)
            for i, cfg in enumerate(configs):
                vals = [data.get((fold, b), {}).get(cfg, {}).get(metric, np.nan)
                        for b in bodies]
                ax.bar(xs + (i - (len(configs) - 1) / 2) * width, vals,
                       width * 0.92, label=cfg if (row == 0 and col == 0) else None,
                       color=cmap(i % 10), zorder=2)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.6, len(bodies) - 0.4)
            ax.set_xticks(xs)
            if row == len(metrics) - 1:
                ax.set_xticklabels([b.replace("sub", "") for b in bodies], fontsize=9)
                ax.set_xlabel("OMOMO subject  (shaded = held out for this fold)")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"{NICE[metric]}\n({BETTER[metric]})", fontsize=10)
            if row == 0:
                ax.set_title(f"fold {fold[-1]}   "
                             f"(held out: {', '.join(sorted(held, key=subnum))})",
                             fontsize=12, weight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(mpatches.Patch(color="0.88", label="held out (never trained)"))
    labels.append("held out (never trained)")
    fig.legend(handles, labels, fontsize=9, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(a.out, dpi=170)

    # The number the figure exists to show: in-distribution mean vs held-out
    # mean, per fold per config. A config that generalizes has a small gap.
    m0 = metrics[0]
    print(f"\n=== {NICE[m0]} ({BETTER[m0]}) ===")
    print(f"{'fold':<6} {'config':<18} {'in-dist':>9} {'held-out':>9} {'gap':>9}")
    print("-" * 54)
    for fold in folds:
        held = set(FOLD_HELDOUT[fold])
        for cfg in configs:
            ind = [data[(fold, b)][cfg][m0] for b in bodies
                   if b not in held and cfg in data.get((fold, b), {})]
            out = [data[(fold, b)][cfg][m0] for b in bodies
                   if b in held and cfg in data.get((fold, b), {})]
            f = lambda v: f"{np.mean(v):9.2f}" if v else f"{'--':>9}"
            gap = (f"{np.mean(ind) - np.mean(out):9.2f}" if ind and out else f"{'--':>9}")
            print(f"{fold:<6} {cfg:<18} {f(ind)} {f(out)} {gap}")

    print(f"\nper-body {m0}:")
    print(f"{'fold':<6} {'body':<7} {'held':<5} " + " ".join(f"{c:>18}" for c in configs))
    for fold in folds:
        held = set(FOLD_HELDOUT[fold])
        for b in bodies:
            vals = " ".join(
                f"{data.get((fold, b), {}).get(c, {}).get(m0, float('nan')):>18.2f}"
                for c in configs)
            print(f"{fold:<6} {b:<7} {'yes' if b in held else '':<5} {vals}")

    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

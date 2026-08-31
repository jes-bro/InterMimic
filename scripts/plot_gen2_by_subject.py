#!/usr/bin/env python3
"""Gen-2 eval results: vertical bars clustered by held-out subject, folds side by side.

x is the held-out subject; each cluster holds one bar per training config
(plain/ret x stock/nvadlr). The fold-0 subjects and fold-1 subjects form two
separated groups along that axis.

WHY THE FOLDS ARE SEPARATE GROUPS RATHER THAN PAIRED BARS. Each fold holds out
a DIFFERENT three bodies -- f0 tests on {10,13,16}, f1 on {5,7,12} -- so no
subject was ever evaluated under both. A figure pairing "fold 0" against
"fold 1" per subject would have one empty bar in every pair. Grouping is the
honest layout: within a group the four configs are directly comparable, and
across groups they are not, which the gap makes visible.

Reads the eval CSVs written by slurm_eval_curriculum.sh. Run and fold come from
the checkpoint column, not the filename, so a renamed file still works.

    python3 scripts/plot_gen2_by_subject.py --in ~/Downloads/eval_results --out gen2.png
    python3 scripts/plot_gen2_by_subject.py --in DIR --include '*g2_mlp*' --out gen2_mlp.png
    python3 scripts/plot_gen2_by_subject.py --in DIR --metric success_rate --out sr.png
"""
import argparse
import csv
import fnmatch
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

METRICS = ["success_rate", "human_pose_error", "object_pose_error"]
NICE = {"success_rate": "success rate",
        "human_pose_error": "human pose error (m)",
        "object_pose_error": "object pose error (m)"}
# Higher is better for success; lower for the errors. Only used to say so on
# the axis, so nobody reads a tall error bar as a good result.
BETTER = {"success_rate": "higher is better",
          "human_pose_error": "lower is better",
          "object_pose_error": "lower is better"}


def read_rows(path):
    """Usable rows from one eval CSV; [] if it holds nothing we can plot.

    A pair whose eval crashed is written with empty metrics and exit_code=1.
    Those are dropped rather than counted, so a dead pair shows as a missing
    bar instead of a zero -- a zero here would read as 'failed the task'.
    """
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if any(r.get(m, "") == "" for m in METRICS):
            continue
        for m in METRICS:
            r[m] = float(r[m])
        out.append(r)
    return out


def run_and_fold(ckpt):
    """(config, fold) from a checkpoint path, e.g. ('ret_stock', 'f1')."""
    run = next((x for x in ckpt.split("/") if x.startswith("smplx_")), None)
    if run is None:
        return None, None
    run = run.replace("smplx_teacher_", "")
    m = re.search(r"__(f\d)$", run)
    fold = m.group(1) if m else None
    cfg = re.sub(r"__f\d$", "", run)
    cfg = re.sub(r"^g2_(mlp|xf)_", "", cfg)          # keep the config, drop arch
    return cfg, fold


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--include", default=None, metavar="GLOB",
                   help="only read CSVs whose filename matches, e.g. '*g2_mlp*'")
    p.add_argument("--metric", choices=METRICS, default=None,
                   help="one metric instead of all three panels")
    p.add_argument("--title", default="Gen-2: held-out subjects by fold")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # (subject, fold) -> {config: value}
    data, configs, folds = {}, [], []
    for path in sorted(glob.glob(os.path.join(a.in_dir, "*.csv"))):
        if path.endswith("__full.csv"):
            continue
        if a.include and not fnmatch.fnmatch(os.path.basename(path), a.include):
            continue
        rows = read_rows(path)
        if not rows or "checkpoint" not in rows[0]:
            continue
        cfg, fold = run_and_fold(rows[0]["checkpoint"])
        if cfg is None or fold is None:
            print(f"[skip] {os.path.basename(path)}: cannot read config/fold "
                  f"from {rows[0]['checkpoint']!r}")
            continue
        if cfg not in configs:
            configs.append(cfg)
        if fold not in folds:
            folds.append(fold)
        for r in rows:
            data.setdefault((r["body"], fold), {})[cfg] = {m: r[m] for m in METRICS}

    if not data:
        raise SystemExit(f"no usable eval CSVs in {a.in_dir}"
                         + (f" matching {a.include!r}" if a.include else ""))

    configs.sort()
    folds.sort()
    # x order: all of fold 0's subjects, then all of fold 1's. Subjects are
    # sorted numerically (sub5 before sub12) rather than as strings.
    def subnum(s):
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 0
    columns = []
    for f in folds:
        subs = sorted({b for (b, ff) in data if ff == f}, key=subnum)
        columns.extend((b, f) for b in subs)

    metrics = [a.metric] if a.metric else METRICS
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11})
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 1.5 * len(columns)),
                                                       3.2 * len(metrics)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    cmap = plt.get_cmap("tab10")
    width = 0.8 / len(configs)
    xs = np.arange(len(columns))

    for ax, metric in zip(axes, metrics):
        for i, cfg in enumerate(configs):
            vals = [data[c].get(cfg, {}).get(metric, np.nan) for c in columns]
            ax.bar(xs + (i - (len(configs) - 1) / 2) * width, vals, width * 0.92,
                   label=cfg, color=cmap(i % 10))
        ax.set_ylabel(f"{NICE[metric]}\n({BETTER[metric]})", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        # Separate the folds with a line and a label, so nobody compares a bar
        # in one group against a bar in the other without noticing.
        n_first = sum(1 for c in columns if c[1] == folds[0])
        if len(folds) > 1:
            ax.axvline(n_first - 0.5, color="0.4", linewidth=1, linestyle="--")
        if ax is axes[0]:
            for f in folds:
                idx = [j for j, c in enumerate(columns) if c[1] == f]
                ax.text(np.mean(idx), ax.get_ylim()[1], f"fold {f[-1]}",
                        ha="center", va="bottom", fontsize=11, weight="bold")

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([c[0] for c in columns])
    axes[-1].set_xlabel("held-out subject")
    axes[0].legend(fontsize=9, ncol=len(configs), loc="upper left",
                   bbox_to_anchor=(0, -0.02), framealpha=0.9)
    fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=170)

    print(f"{'subject':<10} {'fold':<6} " + " ".join(f"{c:>16}" for c in configs))
    print("-" * (18 + 17 * len(configs)))
    for c in columns:
        vals = " ".join(f"{data[c].get(cfg, {}).get(metrics[0], float('nan')):>16.3f}"
                        for cfg in configs)
        print(f"{c[0]:<10} {c[1]:<6} {vals}")
    print(f"\nwrote {a.out}  ({metrics[0]} shown in the table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

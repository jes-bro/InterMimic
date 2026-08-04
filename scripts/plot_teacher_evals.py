#!/usr/bin/env python3
"""Visualize source-teacher eval CSVs: in-dist vs held-out vs synthetic bodies.

Reads the `indist+heldout+syn` CSVs emitted by scripts/eval_one.sh and produces
a table + grouped bars + heatmaps, with the three body groups kept visually
distinct (that split is the whole point of the experiment).

Body groups come from eval_one.sh:104 -- HELDOUT="sub10 sub16 sub13", synthetic
is sub100+, everything else is in-distribution. sub4 is absent (known crasher).

  python3 scripts/plot_teacher_evals.py --out ~/teacher_eval_figs
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# --- Design tokens (validated categorical palette; see dataviz/references/palette.md).
# Slot order is the CVD-safety mechanism -- worst adjacent dE 47.2, well over the 12 floor.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948"]                     # blue, aqua, yellow, green, violet, red
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
# Sequential blue ramp (magnitude), light -> dark. One hue, never a rainbow.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)

# Band tints behind each body group, so train/test/synthetic read at a glance.
GROUP_ORDER = ["In-distribution", "Held-out", "Synthetic"]
BAND = {"In-distribution": "#eef2f6", "Held-out": "#fbeceb", "Synthetic": "#f2f0fa"}
BAND_EDGE = {"In-distribution": "#5b6b7a", "Held-out": "#b4524f", "Synthetic": "#6a5fa8"}

HELDOUT = {"sub10", "sub16", "sub13"}   # eval_one.sh:104


def body_group(body):
    n = int(body[3:])
    if n >= 100:
        return "Synthetic"
    return "Held-out" if body in HELDOUT else "In-distribution"


def short_run(exp):
    """checkpoints/smplx_teacher_src2_xf_aug/... -> src2_xf_aug"""
    return exp.replace("smplx_teacher_", "")


def load(paths, exclude=()):
    exclude = set(exclude)
    loaded = []
    for p in paths:
        rows = list(csv.DictReader(open(p)))
        if not rows:
            raise SystemExit(f"FATAL: {p} has no data rows")
        ckpt = rows[0]["checkpoint"]
        exp = short_run(ckpt.split("/")[1])
        step = int(os.path.basename(ckpt).split("_")[-1].split(".")[0])
        # Drop excluded bodies (e.g. sub13, whose synthetic near-duplicate sub121
        # contaminates its held-out eval) BEFORE any grouping/means are computed.
        rows = [r for r in rows if r["body"] not in exclude]
        if not rows:
            raise SystemExit(f"FATAL: {p} has no rows left after excluding {sorted(exclude)}")
        for r in rows:
            r["group"] = body_group(r["body"])
            r["identity"] = r["is_identity"].strip().lower() == "true"
            for k in ("avg_steps", "human_pose_error", "object_pose_error", "success_rate"):
                r[k] = float(r[k])
        loaded.append((exp, {"rows": rows, "step": step, "source": rows[0]["source"]}))

    # Two passes, because keying on the experiment name alone would make a second
    # CSV from the SAME run silently REPLACE the first -- and comparing one run at
    # two checkpoints (a matched-epoch read) is a normal thing to want. Only the
    # names that actually repeat get an @<step> suffix, so the common case keeps
    # its short label.
    dupes = {e for e, _ in loaded if sum(1 for x, _ in loaded if x == e) > 1}
    runs = {}
    for exp, d in loaded:
        key = f"{exp}@{d['step'] // 1000}k" if exp in dupes else exp
        if key in runs:
            raise SystemExit(f"FATAL: two CSVs map to the same key {key!r} "
                             f"(same run AND same checkpoint passed twice?)")
        runs[key] = d
    return runs


def all_bodies(runs):
    """Union of bodies across runs, group-ordered. Runs do NOT always cover the
    same set -- a later eval may be submitted with BODIES="..." over a subset --
    and taking the first run's set would silently drop columns another run has."""
    seen = []
    for r in runs.values():
        for b in ordered_bodies(r["rows"]):
            if b not in seen:
                seen.append(b)
    return ordered_bodies([{"body": b} for b in seen])


def ordered_bodies(rows):
    """Bodies sorted by group (in-dist, held-out, synthetic) then numerically."""
    return sorted({r["body"] for r in rows},
                  key=lambda b: (GROUP_ORDER.index(body_group(b)), int(b[3:])))


def group_means(runs, metric):
    """{run: {group: mean}} -- the headline read."""
    return {
        run: {g: float(np.mean([r[metric] for r in d["rows"] if r["group"] == g]))
              for g in GROUP_ORDER}
        for run, d in runs.items()
    }


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color="#e3e1dc", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def draw_bands(ax, bodies, y0=0, y1=1, label=True):
    """Shade the x-range of each body group + label it. This is what makes
    train vs test legible without the reader cross-referencing a legend."""
    for g in GROUP_ORDER:
        idx = [i for i, b in enumerate(bodies) if body_group(b) == g]
        if not idx:
            continue
        lo, hi = min(idx) - 0.5, max(idx) + 0.5
        ax.axvspan(lo, hi, color=BAND[g], zorder=0)
        if label:
            ax.text((lo + hi) / 2, y1, g, ha="center", va="bottom",
                    fontsize=9.5, weight="bold", color=BAND_EDGE[g],
                    transform=ax.get_xaxis_transform())


# ----------------------------------------------------------------- figure 1
def fig_success_bars(runs, out):
    # Runs do NOT always cover the same bodies -- a later eval may be submitted
    # with BODIES="..." over a subset. Take the union so no run's data is dropped,
    # and leave a visible GAP where a run has no row for a body (np.nan), rather
    # than plotting a zero, which would read as "scored 0%" instead of "not run".
    bodies = all_bodies(runs)
    names = list(runs)
    x = np.arange(len(bodies))
    w = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(15, 6.2), facecolor=SURFACE)
    style_ax(ax)
    draw_bands(ax, bodies, y1=1.005)

    if len(names) > len(SERIES):
        print(f"  [plot] WARNING: {len(names)} runs but only {len(SERIES)} palette "
              f"slots -- colours repeat after {len(SERIES)}; read the legend order, "
              f"not the colour alone", file=sys.stderr)
    missing = {}
    for i, run in enumerate(names):
        lut = {r["body"]: r for r in runs[run]["rows"]}
        vals = [lut[b]["success_rate"] if b in lut else np.nan for b in bodies]
        gaps = [b for b in bodies if b not in lut]
        if gaps:
            missing[run] = gaps
        # 2px surface gap between adjacent bars; 4px rounded data-end at the top.
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.88, label=run, color=SERIES[i % len(SERIES)],
               zorder=3, linewidth=0.9, edgecolor=SURFACE)

    for run, gaps in missing.items():
        print(f"  [plot] NOTE {run}: no eval row for {len(gaps)} body(ies) "
              f"({', '.join(gaps)}) -- drawn as a GAP, not as 0%", file=sys.stderr)

    ax.set_xticks(x)
    ax.set_xticklabels(bodies, rotation=45, ha="right")
    ax.set_ylabel("Success rate (%)", color=INK2, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(bodies) - 0.5)

    handles = [Patch(facecolor=SERIES[i % len(SERIES)], label=f"{r}  (step {runs[r]['step']:,}, "
                                                f"source {runs[r]['source']})")
               for i, r in enumerate(names)]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2,
              loc="lower left", bbox_to_anchor=(0, -0.34), ncol=2)

    fig.suptitle("Teacher success rate per body", x=0.5, y=0.99, fontsize=15,
                 weight="bold", color=INK)
    ax.set_title("Held-out bodies were never trained on. Higher is better.",
                 fontsize=10, color=INK2, loc="center", pad=26)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def fig_heatmaps(runs, out):
    bodies = all_bodies(runs)
    names = list(runs)
    # (metric, title, higher_is_better) -- one hue each; darker always = "more".
    panels = [("success_rate", "Success rate (%) — higher is better", True),
              ("human_pose_error", "Human pose error — lower is better", False),
              ("object_pose_error", "Object pose error — lower is better", False)]

    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 9.2), facecolor=SURFACE)
    for ax, (metric, title, hib) in zip(axes, panels):
        # np.nan where a run has no row for that body -> imshow leaves the cell
        # blank instead of colouring it as if it were a real (bad) score.
        M = np.array([[ (lambda lut: lut[b][metric] if b in lut else np.nan)(
                            {r["body"]: r for r in runs[run]["rows"]})
                       for b in bodies] for run in names], dtype=float)
        # Darker = better, regardless of metric direction, so the eye reads one way.
        norm = M if hib else -M
        im = ax.imshow(norm, cmap=SEQ, aspect="auto")
        ax.set_xticks(range(len(bodies)))
        ax.set_xticklabels(bodies, rotation=45, ha="right", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=11, weight="bold", color=INK, loc="left", pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

        lo, hi = np.nanmin(norm), np.nanmax(norm)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):          # body not evaluated for this run
                    ax.text(j, i, "n/r", ha="center", va="center", fontsize=7,
                            color=INK3, style="italic")
                    continue
                frac = (norm[i, j] - lo) / (hi - lo + 1e-9)
                txt = f"{M[i, j]:.0f}" if metric == "success_rate" else f"{M[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                        color="#ffffff" if frac > 0.55 else INK)
        # Divider between body groups.
        for g in GROUP_ORDER[:-1]:
            idx = [k for k, b in enumerate(bodies) if body_group(b) == g]
            if idx:
                ax.axvline(max(idx) + 0.5, color=INK, lw=2.2)
        if ax is axes[0]:
            for g in GROUP_ORDER:
                idx = [k for k, b in enumerate(bodies) if body_group(b) == g]
                if idx:
                    ax.text((min(idx) + max(idx)) / 2, -0.85, g, ha="center",
                            fontsize=10, weight="bold", color=BAND_EDGE[g])

    fig.suptitle("Per-body metrics across the three teacher runs", x=0.5, y=0.995,
                 fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def fig_summary(runs, out):
    names = list(runs)
    sr = group_means(runs, "success_rate")
    hp = group_means(runs, "human_pose_error")
    op = group_means(runs, "object_pose_error")

    # The table is 3 rows per run + header, so its height must scale with the run
    # count -- a fixed figsize made a 4-run table collide with its own title.
    n_rows = 3 * len(names) + 1
    table_h = 0.42 * n_rows
    fig = plt.figure(figsize=(15, 5.0 + table_h), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, table_h], hspace=0.55)

    # -- top: group-mean success, the headline comparison
    ax = fig.add_subplot(gs[0])
    style_ax(ax)
    x = np.arange(len(GROUP_ORDER))
    w = 0.8 / len(names)
    for i, run in enumerate(names):
        vals = [sr[run][g] for g in GROUP_ORDER]
        pos = x + i * w - 0.4 + w / 2
        ax.bar(pos, vals, w * 0.88, color=SERIES[i % len(SERIES)], zorder=3,
               edgecolor=SURFACE, linewidth=1.2, label=run)
        for px, v in zip(pos, vals):          # direct labels (relief rule)
            ax.text(px, v + 1.2, f"{v:.1f}", ha="center", fontsize=9.5,
                    weight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({sum(1 for r in runs[names[0]]['rows'] if r['group']==g)} bodies)"
                        for g in GROUP_ORDER], fontsize=10.5, color=INK)
    for t, g in zip(ax.get_xticklabels(), GROUP_ORDER):
        t.set_color(BAND_EDGE[g])
    ax.set_ylabel("Mean success rate (%)", color=INK2, fontsize=10)
    ax.set_ylim(0, 100)
    # Legend lives ABOVE the axes, opposite the title, so it can never collide with
    # the direct value labels on the tallest bars.
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=3,
              loc="lower right", bbox_to_anchor=(1, 1.01))
    ax.set_title("Mean success rate by body group", fontsize=13, weight="bold",
                 color=INK, loc="left")

    # -- bottom: the table view (required relief for the low-contrast slots)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    cols = ["Run", "Source", "Step", "Group", "Success %", "Human err", "Object err"]
    cells, colors = [], []
    for i, run in enumerate(names):
        for g in GROUP_ORDER:
            cells.append([run, runs[run]["source"], f"{runs[run]['step']:,}", g,
                          f"{sr[run][g]:.1f}", f"{hp[run][g]:.4f}", f"{op[run][g]:.4f}"])
            colors.append(["#ffffff"] * 3 + [BAND[g]] + ["#ffffff"] * 3)
    tbl = ax2.table(cellText=cells, colLabels=cols, cellLoc="center",
                    loc="center", cellColours=colors,
                    colColours=["#eceae4"] * len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcdad4")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK)
            if c == 0:                        # run name carries its series color
                cell.get_text().set_color(SERIES[((r - 1) // len(GROUP_ORDER)) % len(SERIES)])
                cell.get_text().set_weight("bold")
            if c == 3:
                cell.get_text().set_color(BAND_EDGE[cells[r - 1][3]])
                cell.get_text().set_weight("bold")
    ax2.set_title("Group means (table view)", fontsize=13, weight="bold",
                  color=INK, loc="left", pad=14)

    fig.suptitle("Source-teacher evals: in-distribution vs held-out vs synthetic",
                 x=0.5, y=0.985, fontsize=15, weight="bold", color=INK)
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=sorted(
        glob.glob(os.path.expanduser("~/smplxteacher*result/*.csv"))))
    ap.add_argument("--out", default=os.path.expanduser("~/teacher_eval_figs"))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="bodies to drop from every figure, e.g. --exclude sub13. "
                         "Output filenames get a _no_<bodies> suffix so existing "
                         "figures are never overwritten.")
    a = ap.parse_args()

    if not a.csv:
        raise SystemExit("FATAL: no CSVs matched. Pass --csv explicitly.")
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.csv, exclude=a.exclude)
    # Suffix added to every output name when bodies are excluded -> new files, the
    # baseline figures on disk stay untouched.
    suffix = ("_no_" + "_".join(sorted(a.exclude))) if a.exclude else ""
    print(f"[plot] {len(runs)} runs: {', '.join(runs)}")
    if a.exclude:
        print(f"[plot] excluding bodies: {sorted(a.exclude)}  (suffix '{suffix}')")
    for run, d in runs.items():
        counts = {g: sum(1 for r in d["rows"] if r["group"] == g) for g in GROUP_ORDER}
        print(f"[plot]   {run}: source={d['source']} step={d['step']:,} bodies={counts}")

    fig_summary(runs, f"{a.out}/1_summary{suffix}.png")
    fig_success_bars(runs, f"{a.out}/2_success_by_body{suffix}.png")
    fig_heatmaps(runs, f"{a.out}/3_heatmaps{suffix}.png")
    print(f"[plot] wrote 3 figures -> {a.out}  (names end in '{suffix}')")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One readable figure: success rate per body, a few arms, grouped bars.

Why this exists alongside plot_teacher_evals.py: that script plots the UNION of
every body any run covers, which with mixed 6-body and 21-body evals produces a
21-column chart that is mostly gaps, and with >6 runs it recycles hues so two
arms share a colour. This one is deliberately narrow -- at most 5 arms, only the
bodies they ALL cover -- so every bar is comparable to the bar beside it.

PER-RUN BODY EXCLUSION (--drop RUN:BODY, repeatable) exists for one real case:
sub13 is not a valid held-out body for src2_aug, which trained on the synthetic
sub121 (0.34 away from sub13). It IS valid for every other arm. Dropping sub13
globally would throw away good data from four arms to protect against one; this
drops it only where it is contaminated, and annotates the gap so the omission is
visible in the figure rather than buried in a caption.

  python3 scripts/plot_arm_bars.py --csv <files...> \
      --drop src2_aug:sub13 --out ~/Downloads/arms.png
"""
import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Same fixed-order categorical set as plot_teacher_evals.py -- one design system,
# not two. Order is the CVD-safety mechanism, so slots are assigned by position
# and never cycled: >5 arms is an error here, not a wrapped palette.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
BAND = {"in-dist": "#eef2f6", "held-out": "#fbeceb"}
BAND_INK = {"in-dist": "#5b6b7a", "held-out": "#b4524f"}
HELDOUT = {"sub10", "sub13", "sub16"}


def group(b):
    return "held-out" if b in HELDOUT else "in-dist"


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("success_rate")]
    if not rows:
        return None
    ck = rows[0]["checkpoint"]
    run = ck.split("/")[1].replace("smplx_teacher_", "")
    step = int(os.path.basename(ck).split("_")[-1].split(".")[0])
    return run, step, {r["body"]: float(r["success_rate"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[], metavar="RUN:BODY",
                    help="drop one body for one run (contaminated), e.g. src2_aug:sub13")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Success rate per body")
    a = ap.parse_args()

    drops = set()
    for d in a.drop:
        if ":" not in d:
            raise SystemExit(f"FATAL: --drop wants RUN:BODY, got {d!r}")
        drops.add(tuple(d.split(":", 1)))

    runs = []
    for p in sorted(a.csv):
        got = load(p)
        if got is None:
            print(f"  skip (no rows): {p}", file=sys.stderr); continue
        run, step, vals = got
        for r, b in drops:
            if r == run and b in vals:
                del vals[b]
                print(f"  [plot] dropped {b} for {run} (--drop)", file=sys.stderr)
        runs.append((f"{run}  @{step/1000:.1f}k", vals))

    if len(runs) > len(SERIES):
        raise SystemExit(
            f"FATAL: {len(runs)} runs but {len(SERIES)} fixed palette slots. "
            f"Cycling hues would make two arms share a colour. Plot fewer arms, "
            f"or split into small multiples.")

    # Only bodies EVERY run has a number for, ignoring --drop'd cells (those are
    # holes in an otherwise-shared column, drawn as gaps).
    dropped_bodies = {b for _, b in drops}
    common = set.intersection(*[set(v) | dropped_bodies for _, v in runs])
    bodies = sorted(common, key=lambda b: (group(b) == "held-out", int(b[3:])))
    if not bodies:
        raise SystemExit("FATAL: these runs share no bodies -- nothing comparable to plot")

    x = np.arange(len(bodies))
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(1.55 * len(bodies) + 3.2, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for g in ("in-dist", "held-out"):
        idx = [i for i, b in enumerate(bodies) if group(b) == g]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color=BAND[g], zorder=0)
            ax.text((min(idx) + max(idx)) / 2, 1.005, g, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, weight="bold", color=BAND_INK[g])

    for i, (label, vals) in enumerate(runs):
        pos = x + i * w - 0.4 + w / 2
        v = [vals.get(b, np.nan) for b in bodies]
        ax.bar(pos, v, w * 0.86, color=SERIES[i], zorder=3,
               linewidth=0.9, edgecolor=SURFACE)     # 2px surface gap between bars
        for xi, vi in zip(pos, v):                    # direct labels: few enough bars
            if not np.isnan(vi):
                ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
        for xi, b in zip(pos, bodies):                # name the holes, don't hide them
            if b not in vals:
                ax.text(xi, 2, "excl.", ha="center", va="bottom", fontsize=6.5,
                        color=INK3, rotation=90, style="italic")

    ax.set_xticks(x); ax.set_xticklabels(bodies, fontsize=10, color=INK2)
    ax.set_ylabel("Success rate (%)", color=INK2, fontsize=10)
    ax.set_ylim(0, 104); ax.set_xlim(-0.5, len(bodies) - 0.5)
    ax.grid(axis="y", color="#e3e1dc", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)

    ax.legend(handles=[Patch(facecolor=SERIES[i], label=l) for i, (l, _) in enumerate(runs)],
              frameon=False, fontsize=9, labelcolor=INK2, loc="lower left",
              bbox_to_anchor=(0, -0.30), ncol=min(3, len(runs)))
    fig.suptitle(a.title, x=0.5, y=0.985, fontsize=14, weight="bold", color=INK)
    ax.set_title("Held-out bodies were never trained on. Higher is better. "
                 "'excl.' = body dropped for that run (contaminated).",
                 fontsize=9, color=INK2, pad=22)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    print(f"[plot] wrote {a.out}  ({len(runs)} arms x {len(bodies)} bodies)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figures for: are the source motions / target bodies retargeted onto sub9?

Renders three PNGs into staged_work/body_identity/:

  1_identity_matrix.png  -- 17 subjects x 17 candidate bodies. Each row is one
                            subject's reference; the boxed cell is the body it
                            actually matches. A sub9 answer would be a vertical
                            stripe; an own-body answer is a diagonal.
  2_own_vs_sub9.png      -- per subject, fit to its own body vs fit to sub9's.
  3_control.png          -- the same read on upstream's OMOMO_retarget, where the
                            answer IS sub9 -- proof the measurement can detect it.

    python3 scripts/plot_body_identity.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_identity import (  # noqa: E402
    FULL_OMOMO_NEW, REPO, REPO_OMOMO_RETARGET,
    available_mjcf_subjects, identity_table,
)

OUT = os.path.join(REPO, "staged_work/body_identity")

# --- design tokens (validated palette; see dataviz references/palette.md) -----
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8981"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"     # categorical slots 1 & 2
# sequential: one hue, light -> dark (blue ramp steps 100..700)
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": INK_MUTED,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _finish(fig, path):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_identity_matrix(table, cands):
    subs = list(table)
    M = np.array([[table[s]["errors"][c] for c in cands] for s in subs])

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    im = ax.imshow(M, cmap=SEQ, norm=LogNorm(vmin=max(M.min(), 1e-2), vmax=M.max()),
                   aspect="auto")

    # box the winner in each row -- the body that reference is actually on
    for r, s in enumerate(subs):
        c = cands.index(table[s]["best"])
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                               edgecolor=SERIES_2, lw=2.2, zorder=3))

    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(cands, rotation=45, ha="right")
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(subs)
    ax.set_xlabel("candidate body (MJCF)")
    ax.set_ylabel("subject whose reference motion we measured")
    ax.set_title("Every subject's reference matches its OWN body\n"
                 "bone-length error, mm (log scale) — boxed = best match",
                 loc="left", color=INK, fontsize=12, pad=12)

    # mark the sub9 column: the "everything was retargeted to sub9" hypothesis.
    # The tick label carries the emphasis, so nothing has to sit in the margin.
    j = cands.index("sub9")
    ax.axvline(j, color=SERIES_2, lw=1.0, ls=(0, (3, 3)), zorder=2, alpha=0.7)
    ax.get_xticklabels()[j].set_color(SERIES_2)
    ax.get_xticklabels()[j].set_fontweight("bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("mean bone-length error (mm)", color=INK_2)
    cb.outline.set_edgecolor(INK_MUTED)
    ax.grid(False)
    fig.text(0.01, -0.06,
             "sub9 (dashed) is the body upstream's canonical omomo.xml describes. If the "
             "motions had been retargeted onto it,\nevery box would sit in that column. "
             "They sit on the diagonal instead — each subject on its own body.",
             fontsize=9, color=INK_2, va="top")
    _finish(fig, os.path.join(OUT, "1_identity_matrix.png"))


def fig_own_vs_sub9(table):
    """sub9 is kept in: for that one subject the two bars are the same measurement,
    which is what "its own body IS sub9's body" looks like -- a useful sanity anchor
    rather than an omission the reader has to take on trust."""
    subs = list(table)
    own = [table[s]["errors"][s] for s in subs]
    to9 = [table[s]["errors"]["sub9"] for s in subs]
    x = np.arange(len(subs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    ax.bar(x - w / 2 - 0.01, own, w, color=SERIES_1, label="fit to its OWN body")
    ax.bar(x + w / 2 + 0.01, to9, w, color=SERIES_2, label="fit to sub9's body")

    j = subs.index("sub9")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=0)
    ax.get_xticklabels()[j].set_color(INK_MUTED)
    ax.set_ylabel("mean bone-length error (mm, log)")

    # headline stats over the 16 subjects where the comparison is between two
    # different bodies -- sub9's self-comparison would only dilute it
    others = [i for i, s in enumerate(subs) if s != "sub9"]
    med_own = np.median([own[i] for i in others])
    med_9 = np.median([to9[i] for i in others])
    ax.set_title(
        f"Each subject's reference fits its own skeleton {med_9/med_own:.0f}x better "
        "than sub9's\n"
        f"medians over the 16 non-sub9 subjects: own body {med_own:.3f} mm  ·  "
        f"sub9's body {med_9:.1f} mm",
        loc="left", color=INK, fontsize=12, pad=26)
    ax.legend(frameon=False, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.005),
              fontsize=9.5)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    # The note goes in the headroom above every bar -- the only region of a
    # log-scale bar chart that is reliably empty.
    ax.text(0.5, 0.965, "sub9's own body IS sub9's body, so its two bars are the "
            "same measurement (greyed tick)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK_2)
    _finish(fig, os.path.join(OUT, "2_own_vs_sub9.png"))


def fig_control(train, control):
    """Same measurement, two datasets: it says 'own body' for one and 'sub9' for the other."""
    shared = sorted(set(train) & set(control), key=lambda s: int(s[3:]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    panels = [("OMOMO_new  (what we train on)", train),
              ("OMOMO_retarget  (upstream teacher-corrected)", control)]

    for ax, (title, tbl) in zip(axes, panels):
        x = np.arange(len(shared))
        own = [tbl[s]["errors"][s] for s in shared]
        to9 = [tbl[s]["errors"]["sub9"] for s in shared]
        ax.bar(x - 0.2, own, 0.38, color=SERIES_1, label="fit to its OWN body")
        ax.bar(x + 0.2, to9, 0.38, color=SERIES_2, label="fit to sub9's body")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(shared)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", color=INK_MUTED, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for xi, s in zip(x, shared):
            verdict = "own body" if tbl[s]["best"] == s else f"{tbl[s]['best']}"
            ax.annotate(f"verdict: {verdict}", (xi, ax.get_ylim()[1]), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)

    axes[0].set_ylabel("mean bone-length error (mm, log)")
    fig.suptitle("Control: the same measurement DOES detect a sub9 retarget when there is one",
                 x=0.01, ha="left", fontsize=12, color=INK)
    # One figure-level legend, clear of the bars in either panel.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncols=2, loc="upper left",
               bbox_to_anchor=(0.01, 0.945), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, os.path.join(OUT, "3_control.png"))


if __name__ == "__main__":
    cands = available_mjcf_subjects(real_only=True)
    train = identity_table(FULL_OMOMO_NEW, cands)
    if not train:
        sys.exit(f"ERROR: no clips under {FULL_OMOMO_NEW} -- need the full 17-subject copy")
    fig_identity_matrix(train, cands)
    fig_own_vs_sub9(train)

    control = identity_table(REPO_OMOMO_RETARGET, cands)
    if control:
        fig_control(train, control)
    else:
        print(f"SKIP fig 3: no clips under {REPO_OMOMO_RETARGET}")

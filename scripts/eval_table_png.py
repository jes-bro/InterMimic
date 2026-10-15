#!/usr/bin/env python3
"""Render scripts/summarize_evals.py's numbers as a PNG table.

Runs summarize_evals.py (the ONE place the pooled-over-clips numbers are
computed; nothing is recomputed here) once per metric on the given CSVs, parses
its rows, and lays them out as a table image: one block per metric, one row per
run, columns in-dist / held-out / synthetic / each held-out body. Notes the
summariser prints (crashed rows dropped) are carried into the figure so a number
never travels without its caveat.

    python3 scripts/eval_table_png.py ~/Downloads/eval_resultsoct15/*chairs*.csv \\
        --heldout sub4 sub10 sub13 sub16 --out ~/Downloads/chairs_27000.png

Text only, text ink only: a table is not a chart, so no series colour is needed
and none is used (the dataviz rule: values wear text tokens).
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("success_rate", "success rate (%)"), ("human_pose_error", "human pose error (m)"),
           ("object_pose_error", "object pose error (m)")]
ROW = re.compile(r"^(\S+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+((?:sub\d+=[\d.]+\s*)+)(\[.*\])?\s*$")


def summarize(csvs, metric, heldout):
    """[(run, epoch, in, held, syn, {body: value}, note)] from summarize_evals.py."""
    cmd = [sys.executable, os.path.join(REPO, "scripts", "summarize_evals.py"), *csvs, "--metric", metric]
    if heldout:
        cmd += ["--heldout", *heldout]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        bodies = dict(kv.split("=") for kv in m.group(6).split())
        rows.append((m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), bodies, (m.group(7) or "").strip("[] ")))
    if not rows:
        raise SystemExit(f"no rows parsed from summarize_evals.py --metric {metric}:\n{out}")
    return rows


def write_tex(path, blocks, bodies, caption, tex_label, heldout):
    """A booktabs table: one block per metric, one row per run; the best value per
    column within a metric block is bold (higher is better for success, lower for
    errors). Runs are named by their variant tags; arm names are kept in the caption."""
    tex_name = {"chairs_xf_nvadlr_nopose": "plain (control)", "chairs_xf_nvadlr_nopose_msexp": "motion-scale reward"}
    cols = ["In-dist.", "Held-out", "Synth."] + [b.replace("sub", "sub") for b in bodies]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append("  \\begin{tabular}{l l " + "r" * len(cols) + "}")
    lines.append("    \\toprule")
    lines.append("    & & \\multicolumn{3}{c}{pooled} & \\multicolumn{%d}{c}{held-out bodies} \\\\" % len(bodies))
    lines.append("    \\cmidrule(lr){3-5} \\cmidrule(lr){6-%d}" % (5 + len(bodies)))
    lines.append("    Metric & Teacher & " + " & ".join(c.replace("sub", "sub\\,") for c in cols) + " \\\\")
    for bi, (label, rows) in enumerate(blocks):
        lines.append("    \\midrule")
        higher_better = "success" in label
        vals = [[r[2], r[3], r[4]] + [r[5].get(b, "") for b in bodies] for r in rows]
        best = []
        for ci in range(len(cols)):
            col = [float(v[ci]) for v in vals if v[ci] != ""]
            best.append((max if higher_better else min)(col) if col else None)
        notes = [r[6] for r in rows]
        for ri, r in enumerate(rows):
            cells = []
            for ci, v in enumerate(vals[ri]):
                if v == "":
                    cells.append("--")
                elif best[ci] is not None and float(v) == best[ci]:
                    cells.append("\\textbf{%s}" % v)
                else:
                    cells.append(v)
            name = tex_name.get(short(r[0]), short(r[0]).replace("_", "\\_"))
            metric_cell = ("\\multirow{%d}{*}{%s}" % (len(rows), label.replace("%", "\\%"))) if ri == 0 else ""
            star = "$^\\dagger$" if notes[ri] else ""
            lines.append("    %s & %s%s & %s \\\\" % (metric_cell, name, star, " & ".join(cells)))
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    epoch = blocks[0][1][0][1]
    held = " ".join(heldout) if heldout else "default split"
    any_note = sorted({r[6] for _, rows in blocks for r in rows if r[6]})
    cap = caption or "Evaluation at epoch %s." % epoch
    cap += " Success rate pooled over clips (every clip counts once); pose errors in metres, lower is better. Held-out bodies: %s. Bold: best per column within a metric." % held.replace("sub", "sub\\,")
    if any_note:
        cap += " $^\\dagger$ " + "; ".join(any_note).replace("row(s)", "pair(s)") + "."
    lines.append("  \\caption{%s}" % cap)
    lines.append("  \\label{%s}" % tex_label)
    lines.append("\\end{table}")
    lines.append("% requires \\usepackage{booktabs,multirow}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def short(run):
    """Arm name -> what distinguishes it (the trailing variant tags)."""
    return run.replace("gen4_omomo_geoall_", "").replace("__f0", "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--heldout", nargs="*", default=None, help="passed through to summarize_evals.py")
    ap.add_argument("--out", default=None, help="PNG path")
    ap.add_argument("--tex", default=None, help="also/instead write a booktabs LaTeX table here")
    ap.add_argument("--title", default=None)
    ap.add_argument("--label", default="tab:eval", help="LaTeX \\label for --tex")
    a = ap.parse_args(argv)
    if not a.out and not a.tex:
        raise SystemExit("give --out <png> and/or --tex <file.tex>")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = [(label, summarize(a.csvs, metric, a.heldout)) for metric, label in METRICS]
    bodies = sorted({b for _, rows in blocks for r in rows for b in r[5]}, key=lambda s: int(s[3:]))
    if a.tex:
        write_tex(a.tex, blocks, bodies, a.title, a.label, a.heldout)
        print(f"wrote {a.tex}")
    if not a.out:
        return 0
    header = ["metric", "run", "epoch", "in-dist", "held-out", "synthetic"] + bodies + ["note"]
    cells = []
    for label, rows in blocks:
        for i, (run, ep, ind, held, syn, per, note) in enumerate(rows):
            cells.append([label if i == 0 else "", short(run), ep, ind, held, syn]
                         + [per.get(b, "") for b in bodies] + [note])

    ink, muted, rule, head_bg = "#1f2328", "#6b7280", "#d0d4da", "#f3f4f6"
    n_rows, n_cols = len(cells), len(header)
    widths = [0.16, 0.26, 0.06, 0.07, 0.07, 0.08] + [0.06] * len(bodies) + [0.24]
    fig_w = 16.0
    fig_h = 0.42 * (n_rows + 2) + (0.5 if a.title else 0.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, colWidths=widths, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(rule)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(ink)
        if r == 0:
            cell.set_facecolor(head_bg)
            cell.get_text().set_weight("bold")
        elif c == n_cols - 1:
            cell.get_text().set_color(muted)
            cell.get_text().set_fontsize(8)
        if c >= 2 and r > 0:
            cell.get_text().set_ha("right")
            cell.set_text_props(ha="right")
    if a.title:
        ax.set_title(a.title, color=ink, fontsize=12, loc="left", pad=8)
    fig.text(0.01, 0.01, "numbers: scripts/summarize_evals.py, success pooled over clips (every clip once); "
             f"held-out = {' '.join(a.heldout) if a.heldout else 'default'}", color=muted, fontsize=8)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}  ({n_rows} rows x {n_cols} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

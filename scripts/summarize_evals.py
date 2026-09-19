#!/usr/bin/env python3
"""Print the group-mean summary table for teacher eval CSVs -- the same
computation behind the tables Claude shows in-session (and behind
plot_teacher_evals.py's summary figure): one success_rate per body from the
CSV, then an UNWEIGHTED mean over bodies per group (every body counts equally,
regardless of clip count). Crashed rows (empty metrics) are dropped, never
averaged as zero.

Groups: held-out = the run's test trio (auto-detected from a __f0/__f1 name,
else the historical {sub10,sub13,sub16}; override with --heldout);
in-dist = other real bodies (id < 100); synthetic = id >= 100.

  python3 scripts/summarize_evals.py ~/Downloads/latestresultsmorefinishedaug11/smplx_teacher_g2_*.csv
  python3 scripts/summarize_evals.py --heldout sub5 sub7 sub12 <f1 csvs...>
"""
import argparse
import csv
import os

import numpy as np

FOLD_TRIOS = {"__f0": {"sub10", "sub13", "sub16"},
              "__f1": {"sub5", "sub7", "sub12"}}


def heldout_for(path, override):
    if override:
        return set(override)
    for tag, trio in FOLD_TRIOS.items():
        if tag in os.path.basename(path):
            return trio
    return FOLD_TRIOS["__f0"]          # historical default split


def per_body_means(rows, metric):
    """{body: mean of `metric` over that body's non-crashed rows}.

    A CSV holds one row per (body, source) pair, so a body appears once per
    source. The previous version built {body: value} straight from the rows,
    which kept only the LAST source's row per body and silently threw the rest
    away -- a bball7 CSV (3 bodies x 7 sources) was being reported from one
    clip per body. Average over sources first; the group means over bodies
    stay unweighted on top of that.
    """
    acc = {}
    for r in rows:
        if r[metric] == "":
            continue                    # crashed pair: excluded, never zero
        acc.setdefault(r["body"], []).append(float(r[metric]))
    return {b: float(np.mean(v)) for b, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--heldout", nargs="*", default=None,
                    help="override the held-out trio (default: from __fN in the "
                         "filename, else sub10/sub13/sub16)")
    ap.add_argument("--metric", default="success_rate",
                    choices=["success_rate", "human_pose_error", "object_pose_error"])
    args = ap.parse_args()

    m = args.metric
    print(f"metric: {m} (unweighted mean over bodies per group)")
    print(f"{'run':34s} {'epoch':>7s} {'in-dist':>8s} {'held-out':>9s} {'syn':>6s}"
          f"  {'held-out bodies':s}")
    for p in args.csvs:
        rows = list(csv.DictReader(open(p)))
        dropped = [r for r in rows if r[m] == ""]
        d = per_body_means(rows, m)
        ck = rows[0]["checkpoint"]
        # Find the experiment directory ANYWHERE in the path. This used to take
        # component [1], which assumes checkpoints/<exp>/nn/... -- true for our
        # runs, but the fold-1 arms live at collab/jm/checkpointsjm/<exp>/nn/...
        # so every one of them was labelled "jm" and they were indistinguishable
        # from each other in the table.
        parts = ck.split("/")
        exp = next((s for s in parts if s.startswith("smplx_teacher_")), None)
        run = exp.replace("smplx_teacher_", "") if exp else os.path.basename(p).split("__")[0]
        # A numbered snapshot is mimic_<epoch>.pth; a ROLLING checkpoint is just
        # mimic.pth, which carries no epoch. The fold-1 arms came from a
        # collaborator that way, so this must not crash on them -- and it must not
        # invent an epoch either, because "unknown training amount" is exactly the
        # caveat those rows need to carry into any comparison.
        stem = os.path.basename(ck).split(".")[0]
        tail = stem.split("_")[-1]
        step = int(tail) if tail.isdigit() else None
        held = heldout_for(p, args.heldout)
        missing = held - set(d)
        if missing:
            print(f"{run:34s}  WARNING: held-out bodies {sorted(missing)} not in "
                  f"this CSV -- wrong --heldout for this run?")
        groups = {
            "ind": [v for b, v in d.items() if int(b[3:]) < 100 and b not in held],
            "held": [d[b] for b in sorted(held) if b in d],
            "syn": [v for b, v in d.items() if int(b[3:]) >= 100],
        }
        g = {k: (float(np.mean(v)) if v else float("nan")) for k, v in groups.items()}
        per_held = "  ".join(f"{b}={d[b]:.1f}" for b in sorted(held) if b in d)
        note = f"  [{len(dropped)} crashed row(s) dropped]" if dropped else ""
        step_s = f"{step:,}" if step is not None else "rolling"
        print(f"{run:34s} {step_s:>7s} {g['ind']:8.1f} {g['held']:9.1f} {g['syn']:6.1f}"
              f"  {per_held}{note}")


if __name__ == "__main__":
    main()

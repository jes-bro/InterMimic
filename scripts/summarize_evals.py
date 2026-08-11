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
        d = {r["body"]: float(r[m]) for r in rows if r[m] != ""}
        ck = rows[0]["checkpoint"]
        run = ck.split("/")[1].replace("smplx_teacher_", "")
        step = int(os.path.basename(ck).split("_")[-1].split(".")[0])
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
        print(f"{run:34s} {step:7,} {g['ind']:8.1f} {g['held']:9.1f} {g['syn']:6.1f}"
              f"  {per_held}{note}")


if __name__ == "__main__":
    main()

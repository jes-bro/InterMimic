#!/usr/bin/env python3
"""Print the summary table for teacher/student eval CSVs, POOLED OVER CLIPS.

Every clip counts once: a group's success rate is (successes in that group) /
(clips attempted in that group), and a pose error is the clip-weighted mean.
That is InterMimic's own definition -- its task computes
success_count / num_motions for the clips a run loaded (intermimic.py
print_final_eval_summary) -- so these numbers are directly comparable to the
baseline's, which is the whole reason for pooling.

WHAT THIS REPLACED, AND WHY. Until 2026-09-20 the table took one value per body
and then an unweighted mean over bodies. Two problems. (1) The per-body value
was originally whichever source's row came LAST in the file, so a body x source
CSV reported one source per body; fixed on 09-19 by averaging over sources,
which made every SOURCE count equally instead of every clip -- a source with 1
clip then carried the weight of one with 14. (2) Neither weighting matched the
baseline's. Pooling fixes both at once, and here it is also body-balanced
anyway: every body in these CSVs runs the same clip set, so weighting clips
equally weights bodies equally too.

Crashed rows (empty metric) are dropped from BOTH numerator and denominator --
their outcome is unknown, so counting their clips as failures would be a guess
-- and the count of dropped rows is printed on the row, never silently.

Groups: held-out = the run's test trio (auto-detected from a __f0/__f1 name,
else the historical {sub10,sub13,sub16}; override with --heldout);
in-dist = other real bodies (id < 100); synthetic = id >= 100.

  python3 scripts/summarize_evals.py ~/Downloads/eval_results/smplx_teacher_g3_*.csv
  python3 scripts/summarize_evals.py --metric object_pose_error <csvs...>
  python3 scripts/summarize_evals.py --heldout sub5 sub7 sub12 <f1 csvs...>
"""
import argparse
import csv
import os

FOLD_TRIOS = {"__f0": {"sub10", "sub13", "sub16"},
              "__f1": {"sub5", "sub7", "sub12"}}

# success_rate is a percentage already; the pose errors are metres and need the
# decimals. One format for every column killed the --metric flag in practice:
# every pose error printed as 0.2 or 0.3.
FMT = {"success_rate": ("{:.1f}", 8), "human_pose_error": ("{:.4f}", 9),
       "object_pose_error": ("{:.4f}", 9)}


def heldout_for(path, override):
    if override:
        return set(override)
    for tag, trio in FOLD_TRIOS.items():
        if tag in os.path.basename(path):
            return trio
    return FOLD_TRIOS["__f0"]          # historical default split


def per_body_totals(rows, metric):
    """{body: (numerator, n_clips)}, poolable by summing both across bodies.

    For success_rate the numerator is the success COUNT scaled by 100, so the
    ratio is an exact percentage rather than an average of per-row percentages
    (the CSV's success_rate column is already 0-100). For a pose error each
    row's value is that pair's mean over its clips, so the numerator is
    value * n_clips and the ratio is the clip-weighted mean.
    """
    acc = {}
    for r in rows:
        if r[metric] == "" or not r.get("success_total"):
            continue                    # crashed pair: out of numerator AND denominator
        n = int(r["success_total"])
        num = 100.0 * int(r["success_count"]) if metric == "success_rate" \
            else float(r[metric]) * n
        a = acc.setdefault(r["body"], [0.0, 0])
        a[0] += num
        a[1] += n
    return {b: (num, n) for b, (num, n) in acc.items() if n > 0}


def pool(totals, bodies):
    """(value, clips) over `bodies`, or (nan, 0) if none of them are present."""
    num = sum(totals[b][0] for b in bodies if b in totals)
    den = sum(totals[b][1] for b in bodies if b in totals)
    return (num / den if den else float("nan")), den


def run_label(rows, path):
    """The experiment the CHECKPOINT belongs to, found anywhere in its path.

    Not component [1]: the fold-1 arms live at collab/jm/checkpointsjm/<exp>/nn/,
    which labelled every one of them "jm". Falls back to the filename stem.
    """
    ck = rows[0]["checkpoint"]
    for pref in ("smplx_teacher_", "smplx_student_"):
        exp = next((s for s in ck.split("/") if s.startswith(pref)), None)
        if exp:
            return exp[len(pref):]
    return os.path.basename(path).split("__")[0]


def run_epoch(rows):
    """The checkpoint's epoch, or None for a rolling mimic.pth -- which must not
    be invented, because "unknown training amount" is the caveat such a row
    carries into any matched-epoch comparison."""
    stem = os.path.basename(rows[0]["checkpoint"]).split(".")[0]
    tail = stem.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--heldout", nargs="*", default=None,
                    help="override the held-out trio (default: from __fN in the "
                         "filename, else sub10/sub13/sub16)")
    ap.add_argument("--metric", default="success_rate", choices=sorted(FMT))
    ap.add_argument("--counts", action="store_true",
                    help="also print successes/clips behind each group (success_rate only)")
    args = ap.parse_args(argv)

    m = args.metric
    fmt, width = FMT[m]
    print(f"metric: {m} (POOLED OVER CLIPS: every clip counts once, as InterMimic reports it)")
    print(f"{'run':34s} {'epoch':>7s} {'in-dist':>{width}s} {'held-out':>{width}s} "
          f"{'syn':>{width}s}  {'held-out bodies':s}")
    for p in args.csvs:
        rows = list(csv.DictReader(open(p)))
        if not rows:
            print(f"{os.path.basename(p):34s}  WARNING: empty CSV")
            continue
        dropped = [r for r in rows if r[m] == ""]
        totals = per_body_totals(rows, m)
        held = heldout_for(p, args.heldout)
        run = run_label(rows, p)
        missing = held - set(totals)
        if missing:
            print(f"{run:34s}  WARNING: held-out bodies {sorted(missing)} not in "
                  f"this CSV -- wrong --heldout for this run?")
        real = {b for b in totals if int(b[3:]) < 100}
        groups = {"ind": sorted(real - held),
                  "held": sorted(held & set(totals)),
                  "syn": sorted(b for b in totals if int(b[3:]) >= 100)}
        vals = {k: pool(totals, bs) for k, bs in groups.items()}
        per_held = "  ".join(f"{b}={fmt.format(totals[b][0] / totals[b][1])}"
                             for b in groups["held"])
        bits = []
        if dropped:
            bits.append(f"{len(dropped)} crashed row(s) dropped")
        if args.counts and m == "success_rate":
            bits.append(" ".join(f"{k} {int(vals[k][0] * vals[k][1] / 100 + 0.5)}/{vals[k][1]}"
                                 for k in ("ind", "held", "syn") if vals[k][1]))
        note = f"  [{'; '.join(bits)}]" if bits else ""
        step = run_epoch(rows)
        step_s = f"{step:,}" if step is not None else "rolling"
        cells = "".join(f" {fmt.format(vals[k][0]):>{width}s}" for k in ("ind", "held", "syn"))
        print(f"{run:34s} {step_s:>7s}{cells}  {per_held}{note}")


if __name__ == "__main__":
    main()

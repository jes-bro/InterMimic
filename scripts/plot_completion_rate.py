#!/usr/bin/env python3
"""Completion rate vs EPOCH, read from the TERMINATION REASONS tables in slurm logs.

Companion to plot_epoch_rewards.py: same log files, same run grouping, same epoch
axis, so the two plots line up. The tables only exist when the job ran with
TERM_REASON=1 (every g3 launcher sets it). The task prints one every
TERM_REASON_EVERY sim steps (2000 in the g3 launchers):

    TERMINATION REASONS  (sim step 4000)
        body  episodes         completed              fell  ...
        sub1     12731     315 (  2.5%)       1 (  0.0%)  ...

WHY THIS SCRIPT DIFFERENCES THE TABLES (--mode window, the default).
The counts in each table are CUMULATIVE since the job started -- the task never
resets them (intermimic.py _accumulate_term_reasons). So the percentage printed
in the log is a running average over the whole job, and it lags further behind
the policy the longer the job runs: at epoch 20k a jump in completion barely
moves it. Window mode subtracts the previous table from the current one, so each
point is the rate over just the episodes that ended since the last table.
--mode cumulative reproduces the printed number, for checking against the log.

Rate = (sum over bodies of <cause> count) / (sum over bodies of episodes ended).
'completed' is exclusive (ended with no terminal cause), so it is a clean
fraction. WHAT IT MEASURES: an episode ends without a terminal cause when it
reaches the clip's end OR has run rolloutLength steps from its start
(humanoid.py compute_humanoid_reset). With rolloutLength 50 and Hybrid init that
is mostly "survived a 50-step window from a random start", NOT "tracked the whole
clip" -- the log's own "survived the clip" wording overstates it. Clip-level
success comes from the eval runs (stateInit Start), not from this curve. The failure columns (fell, ig_diverge, ...) overlap -- one reset can
trip several -- so those can sum past 100% across causes.

A resubmitted run is several logs (one per slurm job). Each job's counters start
again at zero, so windows are computed within each log and then the logs are
joined on the epoch axis. A table printed before the first epoch line has no
epoch to sit at; it is not plotted but still anchors the next window's delta.

    python3 scripts/plot_completion_rate.py --glob 'teacher-g3_bball7_geoall__f0-*.out' --out bball7_completion.png
    python3 scripts/plot_completion_rate.py --glob 'teacher-g3_bball*.out' --out bball_completion.png
    python3 scripts/plot_completion_rate.py --glob 'teacher-g3_bball7*.out' --cause ig_diverge --out bball7_igdiv.png
"""
import argparse
import collections
import glob as globmod
import os
import re
import sys

import numpy as np

# Reuse the reward plotter's conventions so both plots name runs and read epochs
# identically. Importing it also switches matplotlib to the headless Agg backend.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Only names that exist in the COMMITTED plot_epoch_rewards.py: importing
# anything newer breaks this script on a clone that has not got those edits.
from plot_epoch_rewards import EPOCH_RE, ewma, run_name  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402


def break_gaps(x, *ys, factor=6.0):
    """Insert NaN where x jumps, so the line is drawn with a hole, not across it.

    A resubmitted run can be missing a stretch of logs; joining the segments
    either side draws a straight line that reads as progress through a region
    with no data. A gap is a step more than `factor` times the median step.
    Returns ((x, *ys) padded, [(from, to), ...]).
    """
    x = np.asarray(x, dtype=np.float64)
    ys = [np.asarray(y, dtype=np.float64) for y in ys]
    if len(x) < 3 or factor <= 0:
        return (x, *ys), []
    d = np.diff(x)
    med = np.median(d[d > 0]) if np.any(d > 0) else 0.0
    idx = np.flatnonzero(d > factor * med) if med > 0 else np.array([], dtype=int)
    if len(idx) == 0:
        return (x, *ys), []
    gaps = [(float(x[i]), float(x[i + 1])) for i in idx]
    at = idx + 1            # positions in the ORIGINAL array; np.insert takes all at once
    return (np.insert(x, at, np.nan), *[np.insert(y, at, np.nan) for y in ys]), gaps

# "TERMINATION REASONS  (sim step 4000)" opens a table.
BLOCK_RE = re.compile(r"^TERMINATION REASONS\s+\(sim step (\d+)\)")
# A body with nothing ended yet prints no cells, just this note.
EMPTY_ROW_RE = re.compile(r"^\s*(\S+)\s+0\s+\(no episodes ended yet\)\s*$")
# "    sub1     12731     315 (  2.5%)   1 (  0.0%) ..." -> body, episodes, cells.
ROW_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(.*)$")
# One cell is "<count> (<pct>%)"; only the count is kept, the pct is recomputed.
CELL_RE = re.compile(r"(\d+)\s+\(\s*[\d.]+%\)")


def parse_log(path):
    """Read every TERMINATION REASONS table in one log.

    Returns (tables, n_unfinished). Each table is a dict with the sim step, the
    epoch of the last epoch_num line before it (None if there was none yet), the
    column labels from its header, and per-body episodes and cause counts.
    n_unfinished is 1 when the log ends inside a table (the job was killed while
    printing it); that partial table is left out rather than half-counted.
    """
    tables, cur, last_epoch = [], None, None
    with open(path, errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            m = EPOCH_RE.search(line)
            if m:
                last_epoch = int(m.group(1))
                continue
            m = BLOCK_RE.match(line)
            if m:
                cur = {"step": int(m.group(1)), "epoch": last_epoch, "labels": None,
                       "episodes": {}, "counts": {}, "line": lineno}
                continue
            if cur is None:
                continue
            if not line.endswith("\n"):
                continue            # half-written final line of a killed job
            s = line.strip()
            if s and set(s) == {"="}:
                # The closing rule. (The opening rule comes before the title
                # line, when no table is open, so it never lands here.)
                if cur["labels"] is None:
                    raise ValueError(f"{path}:{cur['line']}: table closed before its header row")
                tables.append(cur)
                cur = None
                continue
            if cur["labels"] is None:
                words = s.split()
                if words[:2] == ["body", "episodes"]:
                    cur["labels"] = words[2:]
                continue            # the two explanation lines above the header
            m = EMPTY_ROW_RE.match(line)
            if m:
                cur["episodes"][m.group(1)] = 0
                cur["counts"][m.group(1)] = [0] * len(cur["labels"])
                continue
            m = ROW_RE.match(line)
            if not m:
                continue
            cells = [int(c) for c in CELL_RE.findall(m.group(3))]
            if not cells:
                continue            # some other print interleaved into the table
            if len(cells) != len(cur["labels"]):
                raise ValueError(
                    f"{path}:{lineno}: row for {m.group(1)!r} has {len(cells)} cells "
                    f"but the header names {len(cur['labels'])} columns {cur['labels']}")
            cur["episodes"][m.group(1)] = int(m.group(2))
            cur["counts"][m.group(1)] = cells
    return tables, (0 if cur is None else 1)


def rate_series(tables, cause="completed", mode="window"):
    """Tables from ONE log -> (epochs, rate %, episodes per point, n_no_epoch).

    window:     rate over the episodes that ended since the previous table.
    cumulative: rate over everything since the job started (the logged number).
    A step that does not increase means the counters restarted, so the next
    window is measured from zero. Points with no episodes ended are skipped.
    """
    ep, rate, n = [], [], []
    no_epoch = 0
    prev_e = prev_c = 0
    prev_step = None
    for t in tables:
        ci = t["labels"].index(cause)
        tot_e = sum(t["episodes"].values())
        tot_c = sum(row[ci] for row in t["counts"].values())
        if prev_step is not None and t["step"] <= prev_step:
            prev_e = prev_c = 0
        prev_step = t["step"]
        if mode == "window":
            d_e, d_c = tot_e - prev_e, tot_c - prev_c
        else:
            d_e, d_c = tot_e, tot_c
        prev_e, prev_c = tot_e, tot_c
        if t["epoch"] is None:
            no_epoch += 1
            continue
        if d_e <= 0:
            continue
        ep.append(t["epoch"])
        rate.append(100.0 * d_c / d_e)
        n.append(d_e)
    return (np.asarray(ep, dtype=np.int64), np.asarray(rate, dtype=np.float64),
            np.asarray(n, dtype=np.int64), no_epoch)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", action="append", default=[],
                   help="shell glob over .out files; runs are grouped by name. Repeatable.")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=GLOB",
                   help="explicit label for a glob, overriding the filename. Repeatable.")
    p.add_argument("--cause", default="completed",
                   help="which column to plot: completed (default), fell, nan_obs, "
                        "ig_diverge, contact_diverge")
    p.add_argument("--mode", choices=["window", "cumulative"], default="window",
                   help="window (default): rate since the previous table. "
                        "cumulative: rate since job start, as printed in the log.")
    p.add_argument("--smoothing", type=float, default=0.0,
                   help="EWMA factor in [0,1); 0 (default) plots the windows as-is. "
                        "When on, the raw trace is drawn faintly behind.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="clip every series at this epoch, for matched-epoch reads")
    p.add_argument("--gap-factor", type=float, default=6.0,
                   help="break the line where the epoch step exceeds this many times "
                        "the median step (default 6; 0 disables)")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True, help="output PNG")
    a = p.parse_args(argv)

    if not a.glob and not a.run:
        raise SystemExit("pass --glob or --run")

    groups = collections.OrderedDict()
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run wants LABEL=GLOB, got {spec!r}")
        label, pat = spec.split("=", 1)
        groups.setdefault(label, []).extend(sorted(globmod.glob(pat)))
    for pat in a.glob:
        for f in sorted(globmod.glob(pat)):
            groups.setdefault(run_name(f), []).append(f)
    if not groups:
        raise SystemExit("no files matched")

    series = []
    for label, files in groups.items():
        eps, rates, ns = [], [], []
        n_tables = n_unfinished = n_no_epoch = 0
        for f in files:
            tables, unfinished = parse_log(f)
            n_tables += len(tables)
            n_unfinished += unfinished
            if not tables:
                continue
            if a.cause not in tables[0]["labels"]:
                raise SystemExit(f"{f}: --cause {a.cause!r} is not a column in this log; "
                                 f"it has {tables[0]['labels']}")
            e, r, n, ne = rate_series(tables, a.cause, a.mode)
            eps.append(e); rates.append(r); ns.append(n)
            n_no_epoch += ne
        if n_tables == 0:
            print(f"  SKIP {label}: no TERMINATION REASONS tables in {len(files)} log(s) "
                  f"-- was the job run with TERM_REASON=1?", flush=True)
            continue
        if n_unfinished:
            print(f"  {label}: left out {n_unfinished} table(s) cut off at the end of a log",
                  flush=True)
        if n_no_epoch:
            print(f"  {label}: {n_no_epoch} table(s) printed before the first epoch line; "
                  f"not plotted, still used as the baseline for the next window", flush=True)
        ep = np.concatenate(eps) if eps else np.array([], dtype=np.int64)
        rt = np.concatenate(rates) if rates else np.array([])
        nn = np.concatenate(ns) if ns else np.array([], dtype=np.int64)
        # Resubmitted jobs can overlap in epoch; sort and keep the LAST value at
        # each epoch, which comes from the later (resumed) log.
        order = np.argsort(ep, kind="stable")
        ep, rt, nn = ep[order], rt[order], nn[order]
        _, keep = np.unique(ep[::-1], return_index=True)
        keep = np.sort(len(ep) - 1 - keep)
        ep, rt, nn = ep[keep], rt[keep], nn[keep]
        if a.max_epochs is not None:
            m = ep <= a.max_epochs
            ep, rt, nn = ep[m], rt[m], nn[m]
        if len(ep) == 0:
            print(f"  SKIP {label}: no plottable points", flush=True)
            continue
        series.append((label, ep, rt, nn, len(files)))

    if not series:
        raise SystemExit("nothing to plot: no TERMINATION REASONS tables in any matched log. "
                         "They are only printed when the job ran with TERM_REASON=1.")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    print(f"{'run':<40} {'epochs':>16} {'final':>8} {'best':>8} {'eps/pt':>8}  logs")
    print("-" * 90)
    for i, (label, ep, rt, nn, nfiles) in enumerate(series):
        color = cmap(i % 10)
        # Smooth BEFORE inserting gap NaNs, or one NaN would poison the EWMA.
        sm = ewma(rt, a.smoothing) if a.smoothing > 0 else rt
        (xg, rg, sg), gaps = break_gaps(ep, rt, sm, factor=a.gap_factor)
        if a.smoothing > 0:
            ax.plot(xg, rg, color=color, alpha=0.25, lw=0.8)
        ax.plot(xg, sg, color=color, lw=1.8, label=label)
        for g0, g1 in gaps:
            print(f"  {label}: no tables between epoch {g0:.0f} and {g1:.0f}; line broken there",
                  flush=True)
        print(f"{label:<40} {int(ep[0]):>7d}-{int(ep[-1]):<8d} {rt[-1]:>7.1f}% "
              f"{rt.max():>7.1f}% {int(np.median(nn)):>8d}  x{nfiles}")

    what = "since previous table" if a.mode == "window" else "cumulative since job start"
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{a.cause} (% of episodes ended, {what})")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(a.title or f"{a.cause} rate vs epoch")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

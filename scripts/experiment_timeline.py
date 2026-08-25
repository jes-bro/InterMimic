#!/usr/bin/env python3
"""When did each experiment start and end -- across ALL of its slurm jobs?

An experiment is not a job. Walltime resubmits mean one experiment name spans
several job ids, and `sacct` only knows about jobs. This stitches the two
together:

  1. scan slurm .out logs for the line every bball launcher prints:
         [<tag>] host=<host> job=<id> -> checkpoints/<EXPERIMENT>/nn/
     which ties a job id to an experiment name from the job's OWN output --
     more reliable than guessing from job names, which are abbreviated tags
     (bball-r2_warm) that do not match experiment names (smplx_cari4d_bball_r2_warm).
  2. ask sacct about exactly those job ids
  3. aggregate per experiment: first start, last end, wall span, compute time,
     job count, and the final state

NO SILENT FALLBACKS. A job sacct has forgotten is reported as such, and its
times come from the log file's mtime clearly labelled APPROX -- never silently
mixed in with real accounting data. An experiment with no logs at all is
reported as NO LOGS, not omitted.

Run on the CLUSTER (needs sacct), from the repo root:

    python3 scripts/experiment_timeline.py
    python3 scripts/experiment_timeline.py --experiments-file my_list.txt
    python3 scripts/experiment_timeline.py --log-dir . --csv timeline.csv
    python3 scripts/experiment_timeline.py --since 2026-07-01
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

# "[bball-r2_warm] host=simurgh2 job=17021158 -> checkpoints/smplx_..._r2_warm/nn/"
JOB_LINE = re.compile(r"job=(\d+)\s*->\s*checkpoints/(\S+?)/nn/")
# fallback for logs that only name the experiment (run.py's own banner)
EXP_LINE = re.compile(r"experiment=(\S+)\s")
JOBID_FROM_NAME = re.compile(r"-(\d+)\.out$")
# "epoch_num:14304 mean_rewards:[0.29] fps step: 20167.8 fps total: 16551.5"
PROGRESS = re.compile(r"epoch_num:(\d+)\s+mean_rewards:\[([-\d.eE]+)\]"
                      r"(?:.*?fps total:\s*([\d.]+))?")
# "[warm-start] Successfully restored from ...; resuming at epoch 12970"
WARMSTART = re.compile(r"\[warm-start\].*resuming at epoch (\d+)")


def parse_progress(path, tail_n=50):
    """Last epoch, reward (last value and mean of the last tail_n), fps, and the
    warm-start epoch this job began from.

    epoch_num is NOT epochs-trained: resume_from restores the checkpoint's epoch
    counter (intermimic_agent.py:186), so a run warm-started from a teacher
    begins at the TEACHER's epoch. The warm-start line is what lets us subtract
    it off instead of reporting an inflated number.
    """
    epochs, rewards, fps, warm = [], [], [], None
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = PROGRESS.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    rewards.append(float(m.group(2)))
                    if m.group(3):
                        fps.append(float(m.group(3)))
                    continue
                if warm is None:
                    w = WARMSTART.search(line)
                    if w:
                        warm = int(w.group(1))
    except OSError:
        return None
    if not epochs:
        return None
    tail = rewards[-tail_n:]
    return {
        "last_epoch": max(epochs),
        "reward_last": rewards[-1],
        "reward_tail": sum(tail) / len(tail),
        "tail_n": len(tail),
        "fps_total": (sum(fps[-tail_n:]) / len(fps[-tail_n:])) if fps else None,
        "warm_epoch": warm,
    }

SACCT_FMT = ["JobID", "JobName", "Start", "End", "Elapsed", "State", "ExitCode"]


def scan_logs(log_dir):
    """{experiment: {jobid: logpath}} from the launcher's own output."""
    found = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
        exp = jobid = None
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = JOB_LINE.search(line)
                    if m:
                        jobid, exp = m.group(1), m.group(2)
                        break
                    m = EXP_LINE.search(line)
                    if m and exp is None:
                        exp = m.group(1)
        except OSError as e:
            print(f"[warn] unreadable log {path}: {e}", file=sys.stderr)
            continue
        if exp is None:
            continue                      # not an experiment log
        if jobid is None:                 # recover the id from the filename
            m = JOBID_FROM_NAME.search(os.path.basename(path))
            if not m:
                print(f"[warn] {path}: experiment {exp} but no job id anywhere",
                      file=sys.stderr)
                continue
            jobid = m.group(1)
        found.setdefault(exp, {})[jobid] = path
    return found


def query_sacct(jobids, since=None):
    """{jobid: {field: value}} for jobs sacct still remembers."""
    if not jobids:
        return {}
    cmd = ["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(jobids)),
           "--format=" + ",".join(SACCT_FMT)]
    if since:
        cmd += ["-S", since]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("[warn] sacct not found -- are you on the cluster? "
              "Falling back to log mtimes (APPROX) for everything.", file=sys.stderr)
        return {}
    if out.returncode != 0:
        print(f"[warn] sacct failed (rc={out.returncode}): {out.stderr.strip()}",
              file=sys.stderr)
        return {}
    return parse_sacct(out.stdout)


def parse_sacct(text):
    rows = {}
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < len(SACCT_FMT):
            continue
        rec = dict(zip(SACCT_FMT, parts))
        rows[rec["JobID"].split(".")[0]] = rec
    return rows


def parse_time(s):
    """sacct timestamps are 2026-08-24T14:01:33; Unknown/None mean still running."""
    if not s or s in ("Unknown", "None", "N/A"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_elapsed(s):
    """sacct Elapsed is [DD-]HH:MM:SS -> seconds."""
    if not s:
        return 0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    try:
        h, m, sec = (int(x) for x in s.split(":"))
    except ValueError:
        return 0
    return days * 86400 + h * 3600 + m * 60 + sec


def humanise(seconds):
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def summarise(exp, jobs, sacct):
    """One row per experiment, stitched across its jobs."""
    starts, ends, compute, states, approx = [], [], 0, [], False
    prog_by_job, ordered = {}, sorted(jobs.items(), key=lambda kv: int(kv[0]))
    for jobid, logpath in ordered:
        pr = parse_progress(logpath)
        if pr:
            prog_by_job[jobid] = pr
    for jobid, logpath in ordered:
        rec = sacct.get(jobid)
        if rec:
            st, en = parse_time(rec["Start"]), parse_time(rec["End"])
            compute += parse_elapsed(rec["Elapsed"])
            states.append(f"{jobid}:{rec['State'].split()[0]}")
        else:
            # sacct has forgotten this job (aged out, or wrong cluster). Say so
            # rather than dropping it or pretending the mtime is accounting data.
            st = en = None
            try:
                en = datetime.fromtimestamp(os.path.getmtime(logpath))
            except OSError:
                pass
            states.append(f"{jobid}:NO-SACCT")
            approx = True
        if st:
            starts.append(st)
        if en:
            ends.append(en)
    first = min(starts) if starts else None
    last = max(ends) if ends else None
    span = (last - first).total_seconds() if (first and last) else None

    latest_job = max(jobs, key=int) if jobs else "-"
    # epoch_num is inflated by the warm start: subtract the epoch the FIRST job
    # resumed at. A fresh run has no warm-start line and starts at 0.
    firstjob = ordered[0][0] if ordered else None
    baseline = (prog_by_job.get(firstjob, {}) or {}).get("warm_epoch") or 0
    finals = [p["last_epoch"] for p in prog_by_job.values()]
    last_epoch = max(finals) if finals else None
    latest_prog = prog_by_job.get(latest_job)
    return {
        "experiment": exp,
        "jobs": len(jobs),
        "latest_job": latest_job,
        "epochs": (last_epoch - baseline) if last_epoch is not None else "-",
        "epoch_num": last_epoch if last_epoch is not None else "-",
        "warm_from": baseline if baseline else "",
        "reward": (f"{latest_prog['reward_tail']:.3f}" if latest_prog else "-"),
        "fps": (f"{latest_prog['fps_total']:.0f}"
                if latest_prog and latest_prog["fps_total"] else "-"),
        "first_start": first.strftime("%Y-%m-%d %H:%M") if first else "-",
        "last_end": last.strftime("%Y-%m-%d %H:%M") if last else "(running?)",
        "wall_span": humanise(span) if span is not None else "-",
        "compute": humanise(compute) if compute else "-",
        "approx": "YES" if approx else "",
        "job_states": " ".join(states),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=".",
                    help="where the slurm .out files live (default: repo root)")
    ap.add_argument("--experiments-file",
                    help="file with one experiment name per line; names not found "
                         "in any log are reported as NO LOGS rather than skipped")
    ap.add_argument("--filter", default="bball",
                    help="only report experiments whose name contains this "
                         "(default: bball; pass '' for all)")
    ap.add_argument("--since", help="sacct -S date, e.g. 2026-07-01, to widen the "
                                    "accounting window for old jobs")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args()

    found = scan_logs(args.log_dir)
    wanted = None
    if args.experiments_file:
        with open(args.experiments_file) as fh:
            wanted = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    names = wanted if wanted else sorted(
        e for e in found if args.filter in e)

    all_ids = {j for e in names if e in found for j in found[e]}
    sacct = query_sacct(all_ids, args.since)

    rows, missing = [], []
    for exp in names:
        if exp not in found:
            missing.append(exp)
            continue
        rows.append(summarise(exp, found[exp], sacct))
    rows.sort(key=lambda r: (r["first_start"] == "-", r["first_start"]))

    cols = [("experiment", 40), ("jobs", 4), ("latest_job", 10),
            ("epochs", 8), ("warm_from", 9), ("reward", 7), ("fps", 7),
            ("first_start", 17), ("last_end", 17), ("wall_span", 9),
            ("compute", 8), ("approx", 6)]
    print()
    print("  ".join(h.upper().ljust(w) for h, w in cols))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        print("  ".join(str(r[h])[:w].ljust(w) for h, w in cols))
    print()
    print("wall_span = first job's start to last job's end (INCLUDES queue gaps "
          "and idle time between resubmits)")
    print("compute   = sum of the jobs' Elapsed (actual GPU time)")
    print("approx=YES means sacct had no record for >=1 job; its end time is the "
          "log file mtime, NOT accounting data")
    print("epochs    = epochs ACTUALLY TRAINED = final epoch_num - warm_from.")
    print("warm_from = epoch the first job resumed at. resume_from restores the")
    print("            checkpoint's epoch counter, so an arm warm-started from the")
    print("            sub2 teacher begins at the TEACHER's epoch, not 0. Blank =")
    print("            fresh start. Reading raw epoch_num overstates training by")
    print("            this much.")
    print("reward    = mean of the last 50 mean_rewards lines in the LATEST job")
    print("            (single-value reward is noisy; r3 swings 0.25-0.35).")
    print("fps       = mean 'fps total' over the same window.")

    if missing:
        print()
        print("NO LOGS FOUND for these (typo, never launched, or logs elsewhere):")
        for m in missing:
            print(f"  {m}")

    unreported = sorted(e for e in found
                        if args.filter in e and e not in set(names))
    if unreported and wanted:
        print()
        print("Present in logs but NOT in your list:")
        for u in unreported:
            print(f"  {u}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                               else ["experiment"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

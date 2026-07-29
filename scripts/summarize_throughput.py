#!/usr/bin/env python3
"""Turn throughput-probe logs into one comparison table.

Reads any number of probe logs and reports, per probe: median fps step / fps
total, peak GPU occupancy, where the motion tensors lived, and how it compares to
the reference. Sorted by fps total, best first.

Warm-up epochs are DROPPED (default 5): job 16390586 reads 7071 at epoch 5 and
7390 at epoch 6 before settling, so early epochs flatter whichever probe you read
first. Median, not mean, so one contended epoch cannot move the result.

A probe that crashed (OOM is the expected way for a high --num-envs probe to die)
is reported as FAILED with its reason rather than silently omitted -- a missing
row reads as "not run" when it actually means "this configuration does not fit".

Usage:
    python3 scripts/summarize_throughput.py probe-*.out
    python3 scripts/summarize_throughput.py sweep_12345/*.log --warmup 8
"""

import argparse
import os
import re
import statistics
import sys

# epoch_num:5 mean_rewards:[0.16] fps step: 7071.1 fps total: 5423.6
RE_EPOCH = re.compile(
    r"epoch_num:\s*(\d+).*?fps step:\s*([\d.]+)\s+fps total:\s*([\d.]+)")
# [mem] step 201: torch 20.42G | GPU used 43.5/44G
RE_MEM_STEP = re.compile(r"\[mem\] step \d+: torch ([\d.]+)G \| GPU used ([\d.]+)/(\d+)G")
# [mem] motion tensors: ... = 7.87G on GPU     (or 'on CPU (streamed per step)')
RE_MEM_MOTION = re.compile(r"\[mem\] motion tensors:.*?= ([\d.]+)G (on \w+)")
RE_TAG = re.compile(r"\[probe\] tag=(\S+)")
RE_HOST = re.compile(r"\[probe\].*host=(\S+)")
RE_NUMENVS = re.compile(r"^num_envs:\s*(\d+)", re.M)
# Failure signatures worth naming rather than reporting as "no data".
FAILURES = [
    ("CUDA out of memory", "OOM"),
    ("RuntimeError: CUDA error: out of memory", "OOM"),
    ("Segmentation fault", "SEGFAULT"),
    ("PxgCudaDeviceMemoryAllocator", "PHYSX OOM"),
]


def parse(path, warmup=5):
    """Return a dict summarising one probe log."""
    text = open(path, errors="replace").read()
    tag = RE_TAG.search(text)
    host = RE_HOST.search(text)
    envs = RE_NUMENVS.search(text)
    motion = RE_MEM_MOTION.search(text)

    epochs = [(int(e), float(s), float(t)) for e, s, t in RE_EPOCH.findall(text)]
    kept = [(s, t) for e, s, t in epochs if e > warmup]

    mem = RE_MEM_STEP.findall(text)
    peak_used = max((float(u) for _, u, _ in mem), default=None)
    total_mem = float(mem[0][2]) if mem else None

    failure = None
    for needle, label in FAILURES:
        if needle in text:
            failure = label
            break
    if not kept and not failure:
        failure = "NO EPOCHS" if not epochs else f"ONLY {len(epochs)} EPOCHS (all warmup)"

    return {
        "file": os.path.basename(path),
        "tag": tag.group(1) if tag else os.path.basename(path),
        "host": host.group(1) if host else "?",
        "num_envs": int(envs.group(1)) if envs else None,
        "fps_step": statistics.median(s for s, _ in kept) if kept else None,
        "fps_total": statistics.median(t for _, t in kept) if kept else None,
        "n_epochs": len(kept),
        "peak_used": peak_used,
        "total_mem": total_mem,
        "motion_size": float(motion.group(1)) if motion else None,
        "motion_where": motion.group(2) if motion else "?",
        "failure": failure,
    }


def render(rows, ref_step=None, ref_total=None):
    """Format the comparison table. Returns a list of lines."""
    ok = [r for r in rows if not r["failure"]]
    bad = [r for r in rows if r["failure"]]
    ok.sort(key=lambda r: r["fps_total"], reverse=True)

    out = []
    hdr = (f"{'probe':<28} {'envs':>6} {'fps step':>10} {'fps total':>10} "
           f"{'vs ref':>8} {'peak GPU':>11} {'motion':>16}")
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in ok:
        delta = ""
        if ref_total:
            pct = 100.0 * (r["fps_total"] - ref_total) / ref_total
            delta = f"{pct:+.1f}%"
        gpu = (f"{r['peak_used']:.1f}/{r['total_mem']:.0f}G"
               if r["peak_used"] is not None else "?")
        motion = (f"{r['motion_size']:.2f}G {r['motion_where'].replace('on ', '')}"
                  if r["motion_size"] is not None else "?")
        out.append(f"{r['tag']:<28} {r['num_envs'] or '?':>6} {r['fps_step']:>10.1f} "
                   f"{r['fps_total']:>10.1f} {delta:>8} {gpu:>11} {motion:>16}")
    for r in bad:
        out.append(f"{r['tag']:<28} {r['num_envs'] or '?':>6} {'--':>10} {'--':>10} "
                   f"{'--':>8} {'--':>11} {('FAILED: ' + r['failure']):>16}")

    if ref_total and ref_step:
        out.append("")
        out.append(f"reference (job 16390586): {ref_step:.0f} fps step, "
                   f"{ref_total:.0f} fps total, 43.7/44G")
    hosts = {r["host"] for r in rows if r["host"] != "?"}
    if len(hosts) > 1:
        out.append(f"WARNING: probes ran on DIFFERENT hosts {sorted(hosts)} -- fps is "
                   f"contention-sensitive, so these are not directly comparable.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", help="probe log file(s)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="epochs to drop before measuring (default 5)")
    ap.add_argument("--tag", default=None,
                    help="override the tag when summarising a single log")
    ap.add_argument("--ref-step", type=float, default=7250.0,
                    help="reference fps step (default: job 16390586's median)")
    ap.add_argument("--ref-total", type=float, default=5450.0,
                    help="reference fps total (default: job 16390586's median)")
    args = ap.parse_args(argv)

    rows = []
    for path in args.logs:
        if not os.path.isfile(path):
            print(f"SKIP: no such log {path}", file=sys.stderr)
            continue
        r = parse(path, warmup=args.warmup)
        if args.tag and len(args.logs) == 1:
            r["tag"] = args.tag
        rows.append(r)
    if not rows:
        print("no logs read", file=sys.stderr)
        return 2

    print("\n".join(render(rows, args.ref_step, args.ref_total)))
    return 0 if any(not r["failure"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

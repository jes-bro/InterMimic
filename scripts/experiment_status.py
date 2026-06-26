#!/usr/bin/env python3
"""Status board for curriculum experiments -- auto-discovers every run.

For each curriculum_work/<run>/ it reads the LATEST generated env config (so it
reports what ACTUALLY ran, not what you meant to run), cross-references squeue,
and the substage logs, then prints one row per run:

  features  : network (mlp/xf), betas (gendered/neutral/neutral_aug), balance,
              body-norm, #synthetic bodies
  progress  : current substage + latest epoch seen in the logs
  status    : RUNNING <jobid> (time) | STOPPED  (-> probably needs a resume)

Run from repo root:  python scripts/experiment_status.py
"""
import glob
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "curriculum_work"


def running_jobs():
    """run-name -> (jobid, state, time) for our c-<run> jobs in squeue."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    jobs = {}
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) == 4 and p[1].startswith("c-"):
            jobs[p[1][2:]] = (p[0], p[2], p[3])
    return jobs


def features(env_cfg_text):
    t = env_cfg_text
    def grab(pat, d="?"):
        m = re.search(pat, t, re.M)
        return m.group(1) if m else d
    betas = grab(r"betas_file:\s*(\S+)")
    betas = "neutral_aug" if "aug" in betas else ("neutral" if "neutral" in betas
                                                  else "gendered" if "omomo_betas" in betas else betas)
    net = "xf" if re.search(r"useTransformerObs:\s*true", t) else "mlp"
    bnorm = "ON" if re.search(r"bodyNormalizedReward:\s*true", t) else "-"
    bal = "uniform" if "uniform pair sampling" in t else "invexp"
    sb = grab(r"subjectBodies:\s*(\[[^\]]*\])", "[]")
    n_syn = len(set(re.findall(r"sub1[0-3]\d\b", sb)))      # sub100..sub139
    return net, betas, bal, bnorm, n_syn


def latest_epoch(run_dir):
    logs = glob.glob(str(run_dir / "substage_s*.log"))
    if not logs:
        return "?", "?"
    newest = max(logs, key=os.path.getmtime)
    stage = re.search(r"substage_s(\w+)\.log", os.path.basename(newest))
    stage = stage.group(1) if stage else "?"
    ep = "?"
    try:
        # last epoch_num: in the file (read tail only)
        with open(newest, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8000))
            for m in re.finditer(r"epoch_num:(\d+)", f.read().decode(errors="ignore")):
                ep = m.group(1)
    except Exception:
        pass
    return stage, ep


def main():
    jobs = running_jobs()                       # run -> (jobid, state, time), incl. PENDING
    # Discover from BOTH the on-disk work dirs AND squeue, so queued / just-started
    # runs (no config written yet) still show up -- that was the missing-run bug.
    cfg_dir = {}
    if WORK.is_dir():
        for d in sorted(WORK.glob("*")):
            if d.is_dir() and glob.glob(str(d / "cfgs" / "env_s*.yaml")):
                cfg_dir[d.name] = d
    rows = []
    for name in sorted(set(cfg_dir) | set(jobs)):
        d = cfg_dir.get(name)
        if d is not None:
            cfgs = sorted(glob.glob(str(d / "cfgs" / "env_s*.yaml")))
            net, betas, bal, bnorm, n_syn = features(Path(cfgs[-1]).read_text())
            stage, ep = latest_epoch(d)
        else:                                   # in squeue but hasn't written configs yet
            net = betas = bal = bnorm = "?"
            n_syn, stage, ep = 0, "(starting)", "-"
        if name in jobs:
            jid, state, tm = jobs[name]
            status = f"{state} {jid} ({tm})"     # RUNNING or PENDING
        else:
            status = "STOPPED -> resume?"
        rows.append((name, net, betas, bal, bnorm, str(n_syn) if n_syn else "-",
                     stage, ep, status))
    if not rows:
        print(f"no runs found (looked in {WORK} and squeue) -- run from the repo root on the cluster")
        return

    hdr = ("RUN", "NET", "BETAS", "BAL", "BNORM", "SYN", "STAGE", "EPOCH", "STATUS")
    w = [max(len(r[i]) for r in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join(f"{{:<{x}}}" for x in w)
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * x for x in w]))
    # running, then pending, then stopped
    rank = lambda s: 0 if s.startswith("RUNNING") else 1 if s.startswith("PENDING") else 2
    for r in sorted(rows, key=lambda r: (rank(r[-1]), r[0])):
        print(fmt.format(*r))
    n_run = sum(1 for r in rows if r[-1].startswith("RUNNING"))
    n_pend = sum(1 for r in rows if r[-1].startswith("PENDING"))
    print(f"\n{len(rows)} runs | {n_run} running | {n_pend} pending | "
          f"{len(rows) - n_run - n_pend} stopped")


if __name__ == "__main__":
    main()

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
    if not WORK.is_dir():
        print(f"no {WORK} -- run from the repo root on the cluster")
        return
    jobs = running_jobs()
    rows = []
    for d in sorted(WORK.glob("*")):
        if not d.is_dir():
            continue
        cfgs = sorted(glob.glob(str(d / "cfgs" / "env_s*.yaml")))
        if not cfgs:
            continue
        net, betas, bal, bnorm, n_syn = features(Path(cfgs[-1]).read_text())
        stage, ep = latest_epoch(d)
        if d.name in jobs:
            jid, state, tm = jobs[d.name]
            status = f"RUNNING {jid} ({tm})"
        else:
            status = "STOPPED -> resume?"
        rows.append((d.name, net, betas, bal, bnorm, str(n_syn) if n_syn else "-",
                     stage, ep, status))

    hdr = ("RUN", "NET", "BETAS", "BAL", "BNORM", "SYN", "STAGE", "EPOCH", "STATUS")
    w = [max(len(r[i]) for r in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join(f"{{:<{x}}}" for x in w)
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * x for x in w]))
    # running first, then stopped
    for r in sorted(rows, key=lambda r: (not r[-1].startswith("RUNNING"), r[0])):
        print(fmt.format(*r))
    n_run = sum(1 for r in rows if r[-1].startswith("RUNNING"))
    print(f"\n{len(rows)} runs | {n_run} running | {len(rows) - n_run} stopped")


if __name__ == "__main__":
    main()

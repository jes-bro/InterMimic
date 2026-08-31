#!/usr/bin/env python3
"""Eval a checkpoint on every (target_body, source_subject) combination.

Unlike eval_per_subject.py (which fixes body=source for identity-pair eval),
this iterates over the cartesian product of --bodies × --sources. That
isolates cross-body retargeting from body-control: same body across many
source subjects shows whether the policy can retarget; same source across
many bodies shows whether the policy generalizes body shape.

Example:
    python scripts/eval_per_pair.py \\
        --checkpoint checkpoints/smplx_multibody_stage2/nn/mimic.pth \\
        --bodies  sub2 sub10 sub3 sub17 sub9 sub1 sub5 \\
        --sources sub2 sub10 \\
        --output-csv eval_pair_matrix.csv \\
        --num-envs 1024
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

RESUME_METRICS = ["avg_steps", "human_pose_error", "object_pose_error",
                  "success_rate"]


def load_resumable(csv_path, checkpoint):
    """{(body, source): row} for the pairs in csv_path that already SUCCEEDED.

    A pair counts only when exit_code is 0 and its metrics are actually
    present -- a job that lands on a bad GPU writes a full CSV of exit_code=1
    rows with empty metrics, and resuming over that would preserve the hole.
    Timeouts and crashes are likewise retried rather than kept.

    The rows must belong to the SAME checkpoint. Reusing another checkpoint's
    numbers would produce one CSV silently describing two different policies,
    which no downstream reader could detect, so that is a hard error.
    """
    if not Path(csv_path).exists():
        return {}
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = {}
    for r in rows:
        if r.get("exit_code") != "0":
            continue
        if any(not r.get(m) for m in RESUME_METRICS):
            continue
        if r.get("checkpoint") != str(checkpoint):
            raise SystemExit(
                f"ERROR: {csv_path} holds results for a different checkpoint.\n"
                f"       in the file: {r.get('checkpoint')}\n"
                f"       requested  : {checkpoint}\n"
                f"       Resuming would mix two policies into one CSV. Delete "
                f"the file, or point --output-csv somewhere else.")
        keep[(r["body"], r["source"])] = r
    return keep


METRIC_PATTERNS = {
    "avg_steps":         re.compile(r"Average Execution Steps:\s+([0-9.]+)"),
    "human_pose_error":  re.compile(r"Average Human Pose Error:\s+([0-9.]+)"),
    "object_pose_error": re.compile(r"Average Object Pose Error:\s+([0-9.]+)"),
    "success_rate":      re.compile(r"Success Rate:\s+([0-9.]+)%\s*\(([0-9]+)/([0-9]+)\)"),
}


def make_temp_yaml(base_yaml_path, body, source, all_objects=False, betas_file=None):
    """Copy base_yaml and patch subjectBodies=[body], dataSub=[source].
    If all_objects, also drop any dataObjects restriction so the eval covers
    the subject's FULL object set (the base test config carries a student-eval
    leftover `dataObjects: ['largetable','woodchair']` that otherwise filters
    every clip to 2 objects -- and empties out subjects that lack them).
    Returns path to the temp file."""
    base_text = Path(base_yaml_path).read_text()
    new_text = re.sub(
        r"^(\s*dataSub:).*$",
        rf"\1 ['{source}']",
        base_text, flags=re.MULTILINE,
    )
    new_text = re.sub(
        r"^(\s*subjectBodies:).*$",
        rf"\1 ['{body}']",
        new_text, flags=re.MULTILINE,
    )
    if all_objects:
        # [] => env treats it as "no restriction" and loads all objects.
        new_text = re.sub(
            r"^(\s*dataObjects:).*$",
            r"\1 []",
            new_text, flags=re.MULTILINE,
        )
    if betas_file:
        # Match the checkpoint's training betas (gendered vs neutral vs neutral_aug);
        # the betas occupy part of the obs, so a mismatch silently corrupts eval.
        new_text = re.sub(
            r"^(\s*betas_file:).*$",
            rf"\1 {betas_file}",
            new_text, flags=re.MULTILINE,
        )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_b{body}_s{source}.yaml", delete=False
    )
    tmp.write(new_text)
    tmp.close()
    return tmp.name


def parse_metrics(stdout):
    out = {}
    for name, pat in METRIC_PATTERNS.items():
        m = pat.search(stdout)
        if m is None:
            return None
        if name == "success_rate":
            out["success_rate"] = float(m.group(1))
            out["success_count"] = int(m.group(2))
            out["success_total"] = int(m.group(3))
        else:
            out[name] = float(m.group(1))
    return out


def run_eval(body, source, base_yaml, train_yaml, checkpoint, num_envs, repo_root, timeout_sec, all_objects=False, betas_file=None):
    tmp_yaml = make_temp_yaml(base_yaml, body, source, all_objects=all_objects, betas_file=betas_file)
    cmd = [
        "python", "-u", "-m", "intermimic.run",
        "--task", "InterMimic",
        "--cfg_env", tmp_yaml,
        "--cfg_train", train_yaml,
        "--test",
        "--headless",
        "--checkpoint", str(checkpoint),
        "--num_envs", str(num_envs),
    ]
    tag = f"[body={body},source={source}]"
    print(f"\n{tag} running (timeout={timeout_sec}s)")
    env = {"PYTHONPATH": f"{repo_root}/isaacgym/src:{repo_root}"}
    env = {**os.environ, **env}

    p = subprocess.Popen(
        cmd, cwd=repo_root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = p.communicate(timeout=timeout_sec)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"{tag} TIMEOUT after {timeout_sec}s; killing process group")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            stdout, stderr = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            stdout, stderr = p.communicate()
        rc = -1
        time.sleep(5)

    print(f"{tag} return code: {rc}{' (killed)' if timed_out else ''}")
    metrics = parse_metrics(stdout)
    if metrics is None:
        print(f"{tag} WARNING: could not parse EVALUATION METRICS")
        print(f"{tag} stdout tail:\n{stdout[-1500:]}")
        if stderr:
            print(f"{tag} stderr tail:\n{stderr[-1500:]}")
    else:
        print(f"{tag} metrics: {metrics}")
    Path(tmp_yaml).unlink(missing_ok=True)
    return metrics, rc, timed_out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--bodies", nargs="+", required=True,
                   help="Target subject bodies (subjectBodies in env yaml)")
    p.add_argument("--sources", nargs="+", required=True,
                   help="Source subject motions (dataSub in env yaml)")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument(
        "--base-yaml",
        default="isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml",
    )
    p.add_argument(
        "--train-yaml",
        default="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml",
    )
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--timeout-per-pair", type=int, default=900)
    p.add_argument("--all-objects", action="store_true",
                   help="drop the base config's dataObjects restriction so each "
                        "pair is evaluated on the subject's FULL object set (the "
                        "test config's ['largetable','woodchair'] is a student-eval "
                        "leftover that filters most subjects to empty).")
    p.add_argument("--betas-file", default=None,
                   help="override betas_file in the base yaml to match the checkpoint's "
                        "training betas, e.g. scripts/omomo_betas_neutral.npz. Required "
                        "when the base test config's betas differ from what was trained.")
    p.add_argument("--resume", action="store_true",
                   help="reuse the pairs that already succeeded in --output-csv "
                        "and only run the missing ones. A pair counts as done "
                        "only if exit_code is 0 AND its metrics are present, so "
                        "failures and timeouts are retried.")
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = p.parse_args()

    fields = ["body", "source", "is_identity",
              "avg_steps", "human_pose_error", "object_pose_error",
              "success_rate", "success_count", "success_total",
              "exit_code", "timed_out", "checkpoint"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --resume: reuse the pairs that already succeeded. A 16-body eval is ~80
    # minutes, so a job that hits its walltime three quarters of the way through
    # should not throw away the twelve pairs it paid for.
    done = {}
    if args.resume:
        done = load_resumable(args.output_csv, args.checkpoint)
        if done:
            print(f"[resume] reusing {len(done)} completed pairs from "
                  f"{args.output_csv}")

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for body in args.bodies:
            for source in args.sources:
                if (body, source) in done:
                    writer.writerow(done[(body, source)])
                    f.flush()
                    print(f"[resume] [body={body},source={source}] already done")
                    continue
                metrics, rc, timed_out = run_eval(
                    body, source, args.base_yaml, args.train_yaml,
                    args.checkpoint, args.num_envs, args.repo_root,
                    args.timeout_per_pair, all_objects=args.all_objects,
                    betas_file=args.betas_file,
                )
                row = {
                    "body": body,
                    "source": source,
                    "is_identity": body == source,
                    "exit_code": rc,
                    "timed_out": timed_out,
                    "checkpoint": str(args.checkpoint),
                }
                if metrics is not None:
                    row.update(metrics)
                writer.writerow(row)
                f.flush()

    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()

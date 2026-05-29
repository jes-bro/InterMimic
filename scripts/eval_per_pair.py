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

METRIC_PATTERNS = {
    "avg_steps":         re.compile(r"Average Execution Steps:\s+([0-9.]+)"),
    "human_pose_error":  re.compile(r"Average Human Pose Error:\s+([0-9.]+)"),
    "object_pose_error": re.compile(r"Average Object Pose Error:\s+([0-9.]+)"),
    "success_rate":      re.compile(r"Success Rate:\s+([0-9.]+)%\s*\(([0-9]+)/([0-9]+)\)"),
}


def make_temp_yaml(base_yaml_path, body, source):
    """Copy base_yaml and patch subjectBodies=[body], dataSub=[source].
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


def run_eval(body, source, base_yaml, train_yaml, checkpoint, num_envs, repo_root, timeout_sec):
    tmp_yaml = make_temp_yaml(base_yaml, body, source)
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
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = p.parse_args()

    fields = ["body", "source", "is_identity",
              "avg_steps", "human_pose_error", "object_pose_error",
              "success_rate", "success_count", "success_total",
              "exit_code", "timed_out"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for body in args.bodies:
            for source in args.sources:
                metrics, rc, timed_out = run_eval(
                    body, source, args.base_yaml, args.train_yaml,
                    args.checkpoint, args.num_envs, args.repo_root,
                    args.timeout_per_pair,
                )
                row = {
                    "body": body,
                    "source": source,
                    "is_identity": body == source,
                    "exit_code": rc,
                    "timed_out": timed_out,
                }
                if metrics is not None:
                    row.update(metrics)
                writer.writerow(row)
                f.flush()

    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()

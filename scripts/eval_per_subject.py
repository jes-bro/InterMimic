#!/usr/bin/env python3
"""Eval a checkpoint on each subject separately and dump metrics to a CSV.

For each subject in --subjects, this script generates a single-subject test
yaml on the fly, runs `intermimic.run --test --headless`, parses the
`EVALUATION METRICS` block from stdout, and appends a row to --output-csv.

Example:
    python scripts/eval_per_subject.py \\
        --checkpoint checkpoints/nn/smplx_multibody/last_smplx_multibody_ep_2400_rew_61.5.pth \\
        --subjects sub10 sub17 sub9 sub2 sub3 sub1 sub5 \\
        --output-csv eval_per_subject.csv \\
        --num-envs 1024
"""

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path


# Regexes for parsing the eval metrics block printed at end of eval
METRIC_PATTERNS = {
    "avg_steps":         re.compile(r"Average Execution Steps:\s+([0-9.]+)"),
    "human_pose_error":  re.compile(r"Average Human Pose Error:\s+([0-9.]+)"),
    "object_pose_error": re.compile(r"Average Object Pose Error:\s+([0-9.]+)"),
    "success_rate":      re.compile(r"Success Rate:\s+([0-9.]+)%\s*\(([0-9]+)/([0-9]+)\)"),
}


def make_temp_yaml(base_yaml_path, subject_id):
    """Write a temp yaml that's a copy of base_yaml with subjectBodies and
    dataSub both set to [subject_id]. Returns the temp file path."""
    base_text = Path(base_yaml_path).read_text()

    # Both lines have specific patterns we can swap. Use single-subject
    # list value for both.
    single = f"['{subject_id}']"
    new_text = re.sub(
        r"^(\s*dataSub:).*$",
        rf"\1 {single}",
        base_text, flags=re.MULTILINE,
    )
    new_text = re.sub(
        r"^(\s*subjectBodies:).*$",
        rf"\1 {single}",
        new_text, flags=re.MULTILINE,
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{subject_id}.yaml", delete=False
    )
    tmp.write(new_text)
    tmp.close()
    return tmp.name


def parse_metrics(stdout):
    """Extract the EVALUATION METRICS block from stdout. Returns a dict
    keyed by metric name, or None if the block wasn't found."""
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


def run_eval(subject_id, base_yaml, train_yaml, checkpoint, num_envs, repo_root):
    """Invoke intermimic.run --test for one subject, return parsed metrics."""
    tmp_yaml = make_temp_yaml(base_yaml, subject_id)
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
    print(f"\n[{subject_id}] running: {' '.join(cmd)}")
    env = {"PYTHONPATH": f"{repo_root}/isaacgym/src:{repo_root}"}
    import os
    env = {**os.environ, **env}
    res = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    print(f"[{subject_id}] return code: {res.returncode}")

    metrics = parse_metrics(res.stdout)
    if metrics is None:
        print(f"[{subject_id}] WARNING: could not parse EVALUATION METRICS block")
        print(f"[{subject_id}] stdout tail:\n{res.stdout[-2000:]}")
        if res.stderr:
            print(f"[{subject_id}] stderr tail:\n{res.stderr[-2000:]}")
    else:
        print(f"[{subject_id}] metrics: {metrics}")

    Path(tmp_yaml).unlink(missing_ok=True)
    return metrics, res.returncode


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--subjects", nargs="+", required=True,
                   help="Subjects to evaluate, e.g. sub10 sub17 sub9 sub2 sub3 sub1 sub5")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument(
        "--base-yaml",
        default="isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml",
        help="Template test yaml (subjectBodies + dataSub get overwritten per subject)",
    )
    p.add_argument(
        "--train-yaml",
        default="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml",
        help="RL games train yaml (only network arch matters during eval)",
    )
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="InterMimic repo root (sets cwd + PYTHONPATH)",
    )
    args = p.parse_args()

    fields = ["subject", "avg_steps", "human_pose_error",
              "object_pose_error", "success_rate",
              "success_count", "success_total", "exit_code"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sub in args.subjects:
            metrics, rc = run_eval(
                sub, args.base_yaml, args.train_yaml,
                args.checkpoint, args.num_envs, args.repo_root,
            )
            row = {"subject": sub, "exit_code": rc}
            if metrics is not None:
                row.update(metrics)
            writer.writerow(row)
            f.flush()

    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()

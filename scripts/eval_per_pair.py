#!/usr/bin/env python3
"""Eval a checkpoint on every (target_body, source_subject) combination.

Unlike eval_per_subject.py (which fixes body=source for identity-pair eval),
this iterates over the cartesian product of --bodies × --sources. That
isolates cross-body retargeting from body-control: same body across many
source subjects shows whether the policy can retarget; same source across
many bodies shows whether the policy generalizes body shape.

--env-yaml is the arm's OWN eval config and is passed through untouched; the
two keys that vary across the sweep, subjectBodies and dataSub, are overridden
on the run's command line (--subject_bodies / --data_sub, config.py). Nothing is
copied or rewritten, so the environment a checkpoint is scored in is exactly the
committed, reviewed one -- there is no template that can quietly contribute a
default the arm never trained with.

Example:
    python scripts/eval_per_pair.py \\
        --checkpoint checkpoints/smplx_teacher_g3_bball__f0/nn/mimic_0005.pth \\
        --env-yaml   isaacgym/src/intermimic/data/cfg/omomo_eval_g3_bball__f0.yaml \\
        --train-yaml isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_bball__f0.yaml \\
        --bodies  sub2 sub10 sub16 \\
        --sources sub100 \\
        --output-csv eval_pair_matrix.csv \\
        --num-envs 1024
"""

import argparse
import csv
import os
import re
import signal
import subprocess
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


def _last_metrics_block(stdout):
    """The one block of stdout the metrics should be read from.

    The env prints an 'EVALUATION METRICS:' block EVERY time a per-clip running
    best improves (intermimic.py:1699), then one 'FINAL EVALUATION SUMMARY' at the
    end (print_final_eval_summary). So stdout holds many blocks, ascending.

    This used to be a plain pat.search() over the whole of stdout, which returns
    the FIRST match -- i.e. the earliest, most pessimistic snapshot, taken the
    moment every clip had merely been attempted once. Every CSV written that way
    understates the policy. Prefer the final summary; fall back to the last
    periodic block when the run was killed before printing it (a timeout).

    Returning ONE block also keeps the four metrics mutually consistent: read
    independently they could otherwise come from different points in the rollout.
    """
    marker = "FINAL EVALUATION SUMMARY"
    idx = stdout.rfind(marker)
    if idx != -1:
        return stdout[idx:]
    idx = stdout.rfind("EVALUATION METRICS:")
    return stdout[idx:] if idx != -1 else stdout


def parse_metrics(stdout):
    block = _last_metrics_block(stdout)
    out = {}
    for name, pat in METRIC_PATTERNS.items():
        m = pat.search(block)
        if m is None:
            return None
        if name == "success_rate":
            out["success_rate"] = float(m.group(1))
            out["success_count"] = int(m.group(2))
            out["success_total"] = int(m.group(3))
        else:
            out[name] = float(m.group(1))
    return out


def run_eval(body, source, env_yaml, train_yaml, checkpoint, num_envs, repo_root, timeout_sec):
    # The arm's OWN committed eval config is passed through untouched; only the
    # two keys that vary across the sweep are overridden, on the command line.
    # Nothing is copied, rewritten or written to a temp file, so the environment
    # the policy is scored in is byte-for-byte the reviewed one.
    cmd = [
        "python", "-u", "-m", "intermimic.run",
        "--task", "InterMimic",
        "--cfg_env", str(env_yaml),
        "--cfg_train", train_yaml,
        "--test",
        "--headless",
        "--checkpoint", str(checkpoint),
        "--subject_bodies", body,
        "--data_sub", source,
    ]
    # num_envs is NOT passed unless explicitly asked for. It is an optimistic-bias
    # knob -- success is the best attempt per CLIP (a running max indexed by seq_id,
    # over a clip-count denominator, intermimic.py:1685-1703), so more envs can only
    # raise the success rate and lower the pose errors. It therefore belongs in the
    # committed eval config, where it is reviewable and identical across arms, not
    # in a caller's default that silently overrides it.
    if num_envs:
        cmd += ["--num_envs", str(num_envs)]
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
    # REQUIRED, with no default. The default used to be the old shared template (omomo_test_multibody.yaml),
    # a chunk-1 smoke-test config that silently supplied its own retargeting (none),
    # betas, reset gating, obs horizons and PhysX buffer to whatever checkpoint it
    # was handed. Pass the arm's OWN eval config (cfg/omomo_eval_<arm>.yaml).
    p.add_argument("--env-yaml", required=True, type=Path,
                   help="the arm's own eval config, e.g. "
                        "isaacgym/src/intermimic/data/cfg/omomo_eval_g3_bball__f0.yaml. "
                        "Passed to --cfg_env untouched; only subjectBodies and dataSub "
                        "are overridden per pair, on the command line.")
    p.add_argument("--train-yaml", required=True,
                   help="the arm's rl_games train config (it carries the network arch)")
    p.add_argument("--num-envs", type=int, default=0,
                   help="override the eval config's numEnvs. Default 0 = DON'T, so "
                        "the committed config decides. This is a scoring-budget knob "
                        "(success = best attempt per clip), so overriding it for one "
                        "arm and not another invalidates the comparison.")
    p.add_argument("--timeout-per-pair", type=int, default=900)
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
                    body, source, args.env_yaml, args.train_yaml,
                    args.checkpoint, args.num_envs, args.repo_root,
                    args.timeout_per_pair,
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

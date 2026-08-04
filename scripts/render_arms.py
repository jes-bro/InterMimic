#!/usr/bin/env python3
"""Render policy rollouts for several arms on ONE pinned reference clip.

Purpose: success_rate is a number; this is what it looks like. Given arms that
score e.g. 65% and 8% on the same body at the same epoch, a video of each on the
SAME clip shows whether the difference is falling over, dropping the object,
drifting late, or never acquiring it at all.

WHAT IS HELD FIXED ACROSS ARMS
  * the reference clip -- dataObjects=[OBJECT] + maxClipsPerObject=1 selects the
    first clip of that (source, object) bucket, and intermimic.py:210 sorts by
    file path before capping, so every arm gets the same file. Verified, not
    assumed: each run prints the clip it loaded and this script cross-checks them
    and FAILS if they differ.
  * body, source, state init (Start), and the base test yaml.
  * The base yaml is the arch-matched TEST config (no retargetedMotionDir), so
    the retarget arms are rendered against the same references as everyone else
    -- the same yardstick the evals use.

WHAT VARIES (deliberately)
  * --attempts N runs N envs at once. Each env is an independent attempt at the
    same clip, so one video shows the spread. This matters because success_rate
    is best-of-~385 attempts: a single rollout is one draw and may be atypical in
    either direction. Watching N side by side is the honest visual analogue.

Arch/betas/checkpoint resolution is delegated to eval_one.sh in EMIT mode -- the
same single implementation the eval path uses. Re-deriving it here is exactly how
a mismatched betas file silently corrupts 32 obs dims and still renders something
plausible-looking.

Usage (needs a GPU; see slurm_render_arms.sh for the queued form):
    python3 scripts/render_arms.py \\
        --runs src2_xf_aug_retarget src2_xf_aug_retarget_cpumotion \\
        --body sub16 --source sub2 --object largetable \\
        --attempts 4 --out-dir render_out

A run may pin a checkpoint as run@path, exactly as slurm_eval_multi.sh takes it.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def emit_plan(run, repo_root):
    """Resolve one run via eval_one.sh EMIT -> dict of KEY=VALUE."""
    spec, _, pinned = run.partition("@")
    cmd = ["sh", "scripts/eval_one.sh", spec] + ([pinned] if pinned else [])
    env = dict(os.environ, EMIT="1")
    p = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"FATAL: could not resolve run {run!r}.\n"
                         f"  Try: DRY=1 sh scripts/eval_one.sh {spec}\n{p.stderr}")
    plan = {}
    for line in p.stdout.splitlines():
        m = re.match(r"^(\w+)='(.*)'$", line.strip())
        if m:
            plan[m.group(1)] = m.group(2)
    for k in ("CHECKPOINT", "BETAS_FILE", "BASE_YAML", "TRAIN_YAML"):
        if k not in plan:
            raise SystemExit(f"FATAL: eval_one.sh EMIT gave no {k} for {run!r}")
    return plan


def make_render_yaml(base_yaml, body, source, obj, attempts, betas_file):
    """Patch the arch-matched TEST yaml into a single-clip render config.

    Regex-on-text rather than yaml round-trip, matching eval_per_pair.py's
    make_temp_yaml -- the configs carry load-bearing comments and a round-trip
    would strip them, making the temp file useless for debugging.
    """
    text = Path(base_yaml).read_text()
    # (key, new value). A trailing `# comment` on the replaced line is preserved --
    # those comments explain WHY a value is what it is, and the temp file is what
    # you read when a render looks wrong.
    subs = [
        ("dataSub",           f"['{source}']"),
        ("subjectBodies",     f"['{body}']"),
        ("dataObjects",       f"['{obj}']"),
        ("numEnvs",           str(attempts)),
        ("maxClipsPerObject", "1"),
    ]
    for key, val in subs:
        pat = rf"^(\s*{key}:)[^#\n]*(\s*#.*)?$"
        rep = rf"\1 {val}\2"
        text, n = re.subn(pat, rep, text, flags=re.MULTILINE)
        if n == 0:
            # dataObjects/maxClipsPerObject may be absent from the base config;
            # append rather than silently skip -- a missing pin would render a
            # DIFFERENT clip per arm and the comparison would be meaningless.
            text, n2 = re.subn(r"^(env:\s*)$", rf"\1\n  {key}: {val}", text,
                               count=1, flags=re.MULTILINE)
            if n2 == 0:
                raise SystemExit(
                    f"FATAL: {base_yaml} has neither a '{key}:' line nor an 'env:' "
                    f"block to add one to. Refusing to render -- without {key} the "
                    f"arms would not be pinned to the same clip.")
    if betas_file and betas_file != "none":
        text, n = re.subn(r"^(\s*betas_file:).*$", rf"\1 {betas_file}", text,
                          flags=re.MULTILINE)
        if n == 0:
            raise SystemExit(f"FATAL: no betas_file line in {base_yaml} to override")
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="render_")
    os.close(fd)
    Path(path).write_text(text)
    return path


CLIP_RE = re.compile(r"(InterAct/\S*?(sub\d+_\w+_\d+)\.pt)")


def render_one(run, plan, args, repo_root):
    cfg = make_render_yaml(plan["BASE_YAML"], args.body, args.source,
                           args.object, args.attempts, plan["BETAS_FILE"])
    label = f"{run.partition('@')[0]}__{args.body}"
    out_mp4 = Path(args.out_dir) / f"{label}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{repo_root}/isaacgym/src:{repo_root}:" + env.get("PYTHONPATH", ""),
        "RECORD_VIDEO": str(out_mp4),
        "MAX_VIDEO_FRAMES": str(args.frames),
        "RECORD_VIDEO_WIDE": "1" if args.attempts > 1 else "0",
    })
    cmd = ["python", "-u", "-m", "intermimic.run",
           "--task", "InterMimic",
           "--cfg_env", cfg,
           "--cfg_train", plan["TRAIN_YAML"],
           "--test", "--headless",
           "--checkpoint", plan["CHECKPOINT"],
           "--num_envs", str(args.attempts)]
    print(f"\n=== {run} ===\n  ckpt : {plan['CHECKPOINT']}\n"
          f"  betas: {plan['BETAS_FILE']}\n  base : {plan['BASE_YAML']}\n"
          f"  cmd  : {' '.join(shlex.quote(c) for c in cmd)}", flush=True)

    p = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-4000:])
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        print(f"  !! {run} exited {p.returncode}", file=sys.stderr)
    clips = sorted(set(m.group(2) for m in CLIP_RE.finditer(p.stdout)))
    print(f"  clip(s) loaded: {clips or '<none detected in stdout>'}")
    return {"run": run, "rc": p.returncode, "clips": clips, "mp4": str(out_mp4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run ids as eval_one.sh takes them; run@ckpt to pin")
    ap.add_argument("--body", required=True)
    ap.add_argument("--source", default="sub2")
    ap.add_argument("--object", required=True,
                    help="pins the clip together with maxClipsPerObject=1")
    ap.add_argument("--attempts", type=int, default=4,
                    help="parallel envs = simultaneous attempts at the same clip")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--out-dir", default="render_out")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    results = [render_one(r, emit_plan(r, args.repo_root), args, args.repo_root)
               for r in args.runs]

    print("\n================ RENDER SUMMARY ================")
    for r in results:
        print(f"  {'ok ' if r['rc'] == 0 else 'FAIL'}  {r['run']:44} -> {r['mp4']}")

    # The comparison is only valid if every arm rendered the SAME reference clip.
    seen = {tuple(r["clips"]) for r in results if r["clips"]}
    if len(seen) > 1:
        print("\n  !! CLIPS DIFFER ACROSS ARMS -- these videos are NOT comparable:",
              file=sys.stderr)
        for r in results:
            print(f"     {r['run']:44} {r['clips']}", file=sys.stderr)
        sys.exit(3)
    elif seen:
        print(f"  all arms rendered the same clip: {sorted(seen)[0]}")
    else:
        print("  WARNING: could not detect the clip from stdout; verify manually "
              "before comparing", file=sys.stderr)
    if any(r["rc"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

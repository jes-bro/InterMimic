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


def make_render_yaml(base_yaml, body, source, obj, attempts, betas_file,
                     motion_dir=None, playdataset=False):
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
    if motion_dir:
        text, n = re.subn(r"^(\s*motion_file:).*$", rf"\1 {motion_dir}", text,
                          flags=re.MULTILINE)
        if n == 0:
            raise SystemExit(f"FATAL: no motion_file line in {base_yaml} to repoint")
    if playdataset:
        text, n = re.subn(r"^(\s*playdataset:).*$", r"\1 True", text, flags=re.MULTILINE)
        if n == 0:
            text = re.sub(r"^(env:\s*)$", r"\1\n  playdataset: True", text,
                          count=1, flags=re.MULTILINE)
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


def pin_clip_dir(base_yaml, source, obj, clip, repo_root):
    """Return (motion_dir, clip_name) exposing exactly ONE clip.

    There is no config key that selects the Nth clip -- the loader filters by
    dataSub/dataObjects, sorts by path, then caps with maxClipsPerObject, so
    maxClipsPerObject=1 always yields clip _000. For sub2/largetable that is the
    SHORTEST of 17 (153 frames vs up to 260), which makes for a poor qualitative
    comparison: five seconds is not enough to tell arms apart.

    So: build a temp dir holding a symlink to the chosen clip and point
    motion_file at it. Filenames are preserved, which matters -- the loader parses
    sub<src>_<obj>_<idx>.pt to recover subject and object.
    """
    import glob as _glob
    mf = re.search(r"^\s*motion_file:\s*(\S+)", Path(base_yaml).read_text(), re.MULTILINE)
    if not mf:
        raise SystemExit(f"FATAL: no motion_file in {base_yaml}; cannot pin a clip")
    src_dir = os.path.join(repo_root, mf.group(1))
    clips = sorted(_glob.glob(os.path.join(src_dir, f"{source}_{obj}_*.pt")))
    if not clips:
        raise SystemExit(f"FATAL: no {source}_{obj}_*.pt under {src_dir}")
    names = [os.path.basename(c) for c in clips]
    listing = "\n".join(f"    {n}" for n in names)
    if not clip:                                   # default: first after sorting
        chosen = clips[0]
    else:
        want = clip if clip.endswith(".pt") else clip + ".pt"
        if want not in names:
            raise SystemExit(
                f"FATAL: clip {want!r} not found for {source}/{obj}. "
                f"{len(clips)} available:\n{listing}\n"
                f"  (select by NAME, not position -- the file set differs between "
                f"machines, so an index is not portable)")
        chosen = clips[names.index(want)]
    d = tempfile.mkdtemp(prefix="clip_")
    os.symlink(chosen, os.path.join(d, os.path.basename(chosen)))
    return d, os.path.basename(chosen)


def render_one(run, plan, args, repo_root, motion_dir=None, reference=False):
    # numEnvs=1, NOT numEnvs=attempts. The recording camera is attached to a
    # single env (intermimic_players.py:97, task.envs[_record_env_idx]), so extra
    # envs are simply off-camera -- and the "wide" preset that was meant to show
    # them just parks the camera at (15,15,12), making everything unviewably
    # small. Attempts come SEQUENTIALLY instead: each episode is a fresh attempt,
    # so recording attempts*episodeLength frames captures that many, all close up.
    cfg = make_render_yaml(plan["BASE_YAML"], args.body, args.source,
                           args.object, 1, plan["BETAS_FILE"],
                           motion_dir=motion_dir, playdataset=reference)
    # Filename carries every knob that changes what the video SHOWS: body,
    # attempts, and the checkpoint epoch. Two renders of the same arm that differ
    # in any of those are different experiments and must not share a path --
    # relying on the caller to pass a distinct --out-dir is how one silently
    # overwrites the other.
    step = re.search(r"mimic_0*(\d+)\.pth", plan["CHECKPOINT"])
    stamp = f"ep{int(step.group(1))}" if step else "eplatest"
    clip = ("__" + args.clip.replace(".pt", "")) if args.clip else ""
    label = (f"REFERENCE__{args.body}{clip}" if reference else
             f"{run.partition('@')[0]}__{args.body}__{stamp}{clip}__x{args.attempts}")
    out_mp4 = Path(args.out_dir) / f"{label}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists() and not args.overwrite:
        raise SystemExit(
            f"FATAL: {out_mp4} already exists. Refusing to overwrite a render.\n"
            f"  Pass --overwrite if replacing it is what you want, or use a "
            f"different --out-dir.")

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{repo_root}/isaacgym/src:{repo_root}:" + env.get("PYTHONPATH", ""),
        "RECORD_VIDEO": str(out_mp4),
        "MAX_VIDEO_FRAMES": str(args.frames if args.frames else
                                args.attempts * args.episode_len),
        "RECORD_VIDEO_WIDE": "0",          # never: it is a far-away single-env view
        "RECORD_VIDEO_CAM_POS": args.cam_pos,
        "RECORD_VIDEO_CAM_TARGET": args.cam_target,
    })
    cmd = ["python", "-u", "-m", "intermimic.run",
           "--task", "InterMimic",
           "--cfg_env", cfg,
           "--cfg_train", plan["TRAIN_YAML"],
           "--test", "--headless",
           "--num_envs", "1"]
    if not reference:                      # play_dataset ignores the policy
        cmd[-2:-2] = ["--checkpoint", plan["CHECKPOINT"]]
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
                    help="SEQUENTIAL episodes to record; each is one attempt at "
                         "the pinned clip")
    ap.add_argument("--frames", type=int, default=0,
                    help="max recorded frames; 0 = attempts * --episode-len")
    ap.add_argument("--episode-len", type=int, default=300,
                    help="frames per attempt; must match episodeLength in the cfg")
    ap.add_argument("--cam-pos", default="2.5,2.5,1.8",
                    help="camera position x,y,z. Closer = bigger subject. The "
                         "old default (3,3,2.5) already read small; the previous "
                         "'wide' preset (15,15,12) was unviewable.")
    ap.add_argument("--cam-target", default="0,0,0.9",
                    help="look-at point x,y,z; ~0.9 is torso height")
    ap.add_argument("--out-dir", default="render_out")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--clip", default="",
                    help="clip to pin, BY NAME e.g. sub2_largetable_017 (.pt "
                         "optional). Empty = first after sorting, which is what "
                         "maxClipsPerObject=1 implicitly picks -- for sub2/largetable "
                         "that is the SHORTEST of 17 (153 frames vs up to 260), a poor "
                         "qualitative comparison. An unknown name prints the list. "
                         "By name and not by index because the file set differs "
                         "between machines: index 15 is _017 on one and _016 on the "
                         "cluster, which silently renders a different clip.")
    ap.add_argument("--reference", action="store_true",
                    help="ALSO render the ground-truth mocap replay of the pinned "
                         "clip on the same body (playdataset), so 'what the policy "
                         "is imitating' can be watched beside what it did")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing mp4 (default: refuse)")
    ap.add_argument("--allow-mixed-epochs", action="store_true",
                    help="render arms at DIFFERENT checkpoints anyway (see below)")
    args = ap.parse_args()

    plans = {r: emit_plan(r, args.repo_root) for r in args.runs}

    # Arms train at different speeds, so their LATEST checkpoints sit at different
    # epochs -- rendering those compares training length, not the arms. Refuse by
    # default rather than emit a caveat nobody carries into the figure.
    steps = {r: int(re.search(r"mimic_0*(\d+)\.pth", p["CHECKPOINT"]).group(1))
             if re.search(r"mimic_0*(\d+)\.pth", p["CHECKPOINT"]) else None
             for r, p in plans.items()}
    distinct = {s for s in steps.values() if s is not None}
    if len(distinct) > 1 and not args.allow_mixed_epochs:
        lo = min(distinct)
        print("FATAL: arms resolve to DIFFERENT epochs -- this would compare "
              "training length, not the arms:", file=sys.stderr)
        for r, s in steps.items():
            print(f"    {r:44} epoch {s}", file=sys.stderr)
        print(f"\n  Pin them all to a common epoch (lowest here is {lo}), e.g.:",
              file=sys.stderr)
        for r in args.runs:
            exp = Path(plans[r]["CHECKPOINT"]).parents[1].name
            print(f"    {r.partition('@')[0]}@checkpoints/{exp}/nn/mimic_{lo:08d}.pth",
                  file=sys.stderr)
        print("\n  Or pass --allow-mixed-epochs if the mismatch is the point.",
              file=sys.stderr)
        sys.exit(4)
    if distinct:
        print(f"[render] all arms pinned at epoch {sorted(distinct)[0]}"
              if len(distinct) == 1 else
              f"[render] MIXED epochs {sorted(distinct)} (--allow-mixed-epochs)")

    motion_dir, clip_name = pin_clip_dir(
        plans[args.runs[0]]["BASE_YAML"], args.source, args.object,
        args.clip, args.repo_root)
    print(f"[render] pinned clip {clip_name} -> {motion_dir}")

    results = [render_one(r, plans[r], args, args.repo_root, motion_dir)
               for r in args.runs]
    if args.reference:
        results.append(render_one(args.runs[0], plans[args.runs[0]], args,
                                  args.repo_root, motion_dir, reference=True))

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

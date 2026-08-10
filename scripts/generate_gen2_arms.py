#!/usr/bin/env python3
"""Generate the GEN-2 teacher grid: 8 methods x 2 folds = 16 teachers,
plus a 4-cell buffer-multiplier satellite (see cells()) = 20 total.

Method axes (each carried by exactly one knob set; verified independent by
tests/test_generate_gen2_arms.py):
  arch    : mlp (numObs 3230, network 'intermimic')
          | xf  (numObs 6524, useTransformerObs, network 'intermimic_transformer')
  refs    : plain (original references)
          | ret   (retargetedMotionDir=InterAct/OMOMO_retarget_contact_src2
                   + cpuMotionData ON -- ~7.9G body-major refs can't live in VRAM;
                   an infrastructure necessity, not a swept knob)
  recipe  : stock  (normalize_value False, constant LR 2e-5)
          | nvadlr (normalize_value True, adaptive exact-KL LR, kl_threshold 0.06)

Fold axis (env subjectBodies only):
  f0: test/held-out {sub10, sub13, sub16}   f1: test {sub5, sub7, sub12}
Training bodies per fold = (17 reals - sub4(broken MJCF) - test trio)
  + a SHARED synthetic roster: synthetics that clear the leak floor against
  EVERY fold's test trio. The floor is COMPUTED = smallest real-real subject
  distance (2.106): no training body may sit closer to a test body than two
  real humans ever are. Sharing the roster makes every fold train on identical
  synthetics and equal-size lists (13 real + 30 syn = 43), so fold differences
  isolate the real-body swap. See shared_synthetics() for the tradeoff note.

Env count: UNIFORM 2048 for all 16 cells (user decision 2026-08-09, option a):
envs x horizon = batch, so mixed counts would confound the refs axis with
batch size. NUM_ENVS=<n> at sbatch time overrides per submission.

Usage (from repo root):
    python3 scripts/generate_gen2_arms.py [--dry-run]
Submit (MLP cells first per plan):  sbatch slurm_teacher_g2_mlp_*__f0.sh  etc.
Eval:  HELDOUT="<fold's test trio>" sh scripts/eval_one.sh g2_<arm>__f<k>
"""
import argparse
import os
import re

import numpy as np

from generate_kfold_cfgs import BETAS, beta_dist, real_human_floor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

FOLDS = {0: ["sub10", "sub13", "sub16"], 1: ["sub5", "sub7", "sub12"]}
BROKEN = "sub4"

ENV_BASE = {
    ("mlp", "plain"): "omomo_teacher_src2_mlp_lowbuf.yaml",
    ("mlp", "ret"):   "omomo_teacher_src2_mlp_retarget_lowbuf.yaml",
    ("xf", "plain"):  "omomo_teacher_src2_xf_aug_lowbuf.yaml",
    ("xf", "ret"):    "omomo_teacher_src2_xf_aug_retarget_nvadlr_lowbuf.yaml",
}
TRAIN_BASE = {
    ("mlp", "stock"):  "train/rlg/omomo_teacher_src2_mlp_retarget.yaml",
    ("mlp", "nvadlr"): "train/rlg/omomo_teacher_src2_mlp_retarget_nvadlr.yaml",
    ("xf", "stock"):   "train/rlg/omomo_teacher_src2_xf_aug.yaml",
    ("xf", "nvadlr"):  "train/rlg/omomo_teacher_src2_xf_aug_normval_adlr_lowbuf.yaml",
}


def cells():
    """16 main cells (multiplier 12) + the 4-cell buffer satellite (multiplier
    20): {mlp,xf} x {plain,ret} at stock recipe, fold0 only -- the minimal set
    that isolates the lowbuf change for both archs and both ref types. Added
    2026-08-09 at Jess's request ('does lowbuf affect performance?')."""
    for arch in ("mlp", "xf"):
        for refs in ("plain", "ret"):
            for recipe in ("stock", "nvadlr"):
                for fold in sorted(FOLDS):
                    yield (f"g2_{arch}_{refs}_{recipe}__f{fold}",
                           arch, refs, recipe, fold, 12)
    for arch in ("mlp", "xf"):
        for refs in ("plain", "ret"):
            yield (f"g2_{arch}_{refs}_stock_buf20__f0", arch, refs, "stock", 0, 20)


def shared_synthetics(betas, floor, log=lambda s: None):
    """SHARED synthetic roster: a synthetic survives only if it clears the leak
    floor against EVERY fold's test trio (an AND across folds). All folds then
    train on identical synthetics + equal-size body lists, so fold-to-fold
    differences can only come from the real-body swap -- the thing folds test.
    Cost (accepted 2026-08-09): each fold gives up synthetics that are harmless
    for it but leaky for another fold; uniform across arms, so a level shift,
    not a confound. Support-distance geometry per test body is printed by
    main() -- it CANNOT be equalized by any roster choice, only measured."""
    keep = []
    for i in range(100, 140):
        s = f"sub{i}"
        worst = min(min(beta_dist(betas, s, t) for t in FOLDS[f]) for f in FOLDS)
        if worst < floor:
            log(f"    SHARED-DROP {s}: {worst:.3f} from some fold's test body (< {floor:.3f})")
        else:
            keep.append(s)
    return keep


def fold_bodies(betas, fold, shared_syns):
    """13 real training bodies for this fold + the shared synthetic roster."""
    test = FOLDS[fold]
    reals = [f"sub{i}" for i in range(1, 18)
             if f"sub{i}" != BROKEN and f"sub{i}" not in test]
    return reals + shared_syns


def set_bodies(env_text, bodies, path):
    """Replace subjectBodies (flow [] or block-list style) with `bodies`."""
    flow = re.search(r"^(\s*)subjectBodies:\s*\[(.*)\]\s*$", env_text, re.M)
    if flow:
        return env_text.replace(
            flow.group(0),
            f"{flow.group(1)}subjectBodies: [{', '.join(repr(b) for b in bodies)}]")
    block = re.search(r"^(\s*)subjectBodies:\s*\n((?:\1- \S+\s*\n)+)", env_text, re.M)
    if not block:
        raise SystemExit(f"FATAL: no subjectBodies (flow or block) in {path}")
    return env_text.replace(
        block.group(0),
        f"{block.group(1)}subjectBodies:\n"
        + "".join(f"{block.group(1)}- {b}\n" for b in bodies))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", default=REPO)
    args = ap.parse_args(argv)

    raw = np.load(BETAS)
    betas = {k: raw[k] for k in raw.files if k.startswith("sub")}
    floor = real_human_floor(betas)
    print(f"[gen2] leak floor = smallest real-real distance = {floor:.3f}")
    shared = shared_synthetics(betas, floor, print)
    print(f"[gen2] shared synthetic roster: {len(shared)} of 40 (identical in every fold)")
    bodies_by_fold = {}
    for fold in sorted(FOLDS):
        bodies_by_fold[fold] = fold_bodies(betas, fold, shared)
        print(f"[gen2] f{fold}: test={FOLDS[fold]} -> {len(bodies_by_fold[fold])} "
              f"train bodies (13 real + {len(shared)} shared syn)")
        # Support geometry: how far each test body sits from its nearest
        # training support. NOT equalizable by design -- printed so fold
        # results are always read next to it. (Historical calibration: sub10
        # is the MOST isolated test body yet the easiest -- support distance
        # is a weak driver of difficulty in past evals.)
        for t in FOLDS[fold]:
            near = sorted((beta_dist(betas, tr, t), tr) for tr in bodies_by_fold[fold])[:2]
            print(f"    f{fold} {t}: nearest support "
                  + ", ".join(f"{tr}@{d:.2f}" for d, tr in near))

    out_cfg = os.path.join(args.out_root, "isaacgym/src/intermimic/data/cfg")
    written = []
    for name, arch, refs, recipe, fold, mult in cells():
        env_path = os.path.join(CFG, ENV_BASE[(arch, refs)])
        train_path = os.path.join(CFG, TRAIN_BASE[(arch, recipe)])
        env = set_bodies(open(env_path).read(), bodies_by_fold[fold], env_path)
        if mult != 12:
            # buffer satellite: the ONE change vs its multiplier-12 twin
            old = "default_buffer_size_multiplier: 12.0"
            if env.count(old) != 1:
                raise SystemExit(f"FATAL: {env.count(old)} multiplier lines in {env_path}")
            env = env.replace(old, f"default_buffer_size_multiplier: {mult}.0")
        train, n = re.subn(r"full_experiment_name:\s*\S+",
                           f"full_experiment_name: smplx_teacher_{name}",
                           open(train_path).read())
        if n != 1:
            raise SystemExit(f"FATAL: {n} full_experiment_name lines in {train_path}")
        if args.dry_run:
            print(f"[gen2] (dry) {name}")
            continue
        header = (f"# GENERATED by scripts/generate_gen2_arms.py -- gen-2 grid cell {name}.\n"
                  f"# arch={arch} refs={refs} recipe={recipe} fold=f{fold} "
                  f"buffer_mult={mult} (test={FOLDS[fold]}).\n"
                  f"# Bodies = 13 reals + synthetics beyond the {floor:.3f} leak floor.\n")
        ret_guard = RET_GUARD if refs == "ret" else PLAIN_GUARD
        slurm = SLURM_TMPL.format(name=name, arch=arch, refs=refs, recipe=recipe,
                                  fold=fold, mult=mult, test=" ".join(FOLDS[fold]),
                                  ret_guard=ret_guard)
        for path, content in [
                (os.path.join(out_cfg, f"omomo_teacher_{name}.yaml"), header + env),
                (os.path.join(out_cfg, f"train/rlg/omomo_teacher_{name}.yaml"), header + train),
                (os.path.join(args.out_root, f"slurm_teacher_{name}.sh"), slurm)]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            written.append(path)
    if not args.dry_run:
        print(f"[gen2] wrote {len(written)} files ({len(written) // 3} cells)")
    return written


RET_GUARD = """\
# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\\s*cpuMotionData:\\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi"""
PLAIN_GUARD = """\
# Plain arm: original references; retargeting must NOT be configured.
if grep -qE '^\\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: plain arm has retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi"""

SLURM_TMPL = """#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-{name}"
#SBATCH --output=teacher-{name}-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# GEN-2 grid cell: arch={arch} refs={refs} recipe={recipe} fold=f{fold}
# buffer_mult={mult} (test bodies: {test}). Generated by scripts/generate_gen2_arms.py.
# 24h walltime for fast iteration -- resubmit to auto-resume.
# Eval when done:  HELDOUT="{test}" sh scripts/eval_one.sh {name}

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export PYTHONPATH="isaacgym/src:.${{PYTHONPATH:+:$PYTHONPATH}}"

# Reward diagnostics (print-only; none change training).
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

# UNIFORM env count across ALL 16 cells (batch = envs*horizon must not differ
# between compared arms). Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${{NUM_ENVS:-2048}}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{name}.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{name}.yaml

{ret_guard}

# Buffer guard: the cfg must carry exactly this cell's multiplier ({mult}.0).
if ! grep -qE '^\s*default_buffer_size_multiplier:\s*{mult}\.0' "$CFG_ENV"; then
    echo "[teacher] ERROR: buffer multiplier in $CFG_ENV is not {mult}.0" >&2; exit 1
fi

# Fold guard: this cell's TEST bodies must not be in its training list.
# Parsed exactly (yaml), not grepped -- sub1 vs sub10 substring traps.
for b in {test}; do
    if python3 -c "import yaml,sys; sys.exit(0 if '$b' in yaml.safe_load(open('$CFG_ENV'))['env']['subjectBodies'] else 1)"; then
        echo "[teacher] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2; exit 1
    fi
done

echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[teacher] GEN2 {name}: arch={arch} refs={refs} recipe={recipe} fold=f{fold} buf={mult}.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_{name}/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{{print $2}}')
CKPT="checkpoints/${{EXP}}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${{EXP}}_resume_${{SLURM_JOB_ID}}.yaml"
    sed "s|resume_from: 'None'|resume_from: '${{CKPT}}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[teacher] RESUMING from ${{CKPT}}"
else
    echo "[teacher] fresh start (no checkpoint at ${{CKPT}})"
fi

python -u -m intermimic.run \\
    --task InterMimic \\
    --cfg_env "$CFG_ENV" \\
    --cfg_train "$CFG_TRAIN" \\
    --num_envs "$NUM_ENVS" \\
    --headless \\
    --output checkpoints
"""

if __name__ == "__main__":
    main()

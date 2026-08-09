#!/usr/bin/env python3
"""Generate k-fold cross-validation teacher configs: rotate which real OMOMO
bodies are held OUT, to test whether held-out difficulty (esp. sub16-style) is
body-specific or systemic.

Fold design
-----------
- fold0 is the EXISTING held-out set {sub10, sub13, sub16} -- the teachers you
  already have serve as that fold; nothing is generated for it.
- Test-eligible bodies = real subs 1..17 minus:
    sub4  (broken MJCF, sim-crasher -- excluded from everything, train AND test)
    sub9  (excluded from test by request; stays in training)
    sub2  (the motion source -- always trained, holding out the identity body
           answers a different question)
    {sub10, sub13, sub16}  (already covered by fold0)
- Of the 11 remaining, the 2 nearest sub2 in beta space are dropped from
  testing (near-identity bodies are the least informative OOD probes; they stay
  in training). The other 9 are sorted by beta-distance to sub2 and dealt
  round-robin into 3 folds, so each fold spans near/mid/far bodies.

Synthetic-leak guard
--------------------
sub121 sits 0.34 (beta L2) from held-out sub13 and contaminated its eval; the
base cfg already drops it. The same leak can happen to ANY fold: a synthetic
interpolant landing near a test body is train/test leakage. Per fold, EVERY
synthetic within the leak threshold of ANY test body is excluded from that
fold's training list, and the distances are printed so the exclusion is
auditable. The threshold is COMPUTED from the data, not hardcoded: the smallest
pairwise distance between two real subjects (sub12<->sub13, ~2.106) -- i.e. no
training body may sit closer to a test body than two distinct real humans ever
are. --leak-threshold overrides it, for experiments only.

Recipe: stock MLP (the best-mean teacher family, near-converged by ~16k epochs
so folds are cheap), on the sub121-free lowbuf base env.

Usage (from repo root):
    python3 scripts/generate_kfold_cfgs.py            # writes cfgs + slurm
    python3 scripts/generate_kfold_cfgs.py --dry-run  # print folds, write nothing
Then per fold on the cluster:  sbatch slurm_teacher_kfold{1,2,3}_src2_mlp.sh
Eval: HELDOUT="<that fold's 3 test bodies>" sh scripts/eval_one.sh kfold1_src2_mlp
"""
import argparse
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
BASE_ENV = os.path.join(CFG, "omomo_teacher_src2_mlp_lowbuf.yaml")
BASE_TRAIN = os.path.join(CFG, "train/rlg/omomo_teacher_src2_aug.yaml")  # stock MLP recipe
BETAS = os.path.join(REPO, "scripts/omomo_betas_neutral_aug.npz")

BROKEN = {"sub4"}                 # sim-crasher MJCF: never train, never test
NO_TEST = {"sub9"}                # by request: train yes, test no
SOURCE = "sub2"                   # always trained; never a test body
FOLD0 = {"sub10", "sub13", "sub16"}   # existing teachers already cover this fold
GLOBAL_SYN_DROP = {"sub121"}      # near-dup of sub13; dropped everywhere (base cfg agrees)
N_FOLDS, FOLD_SIZE = 3, 3


def beta_dist(betas, a, b):
    return float(np.linalg.norm(betas[a] - betas[b]))


def real_human_floor(betas):
    """Smallest pairwise beta distance between two REAL subjects (incl. sub4 --
    its betas are fine even though its MJCF is broken). This is the leak
    threshold: no training body may be nearer a test body than this."""
    reals = [f"sub{i}" for i in range(1, 18)]
    return min(beta_dist(betas, a, b)
               for i, a in enumerate(reals) for b in reals[i + 1:])


def assign_folds(betas):
    """Returns (folds, always_train_note): folds = list of sorted 3-body test sets."""
    reals = [f"sub{i}" for i in range(1, 18)]
    eligible = [b for b in reals
                if b not in BROKEN | NO_TEST | FOLD0 and b != SOURCE]
    # Drop the 2 nearest the source: least informative as OOD probes.
    by_dist = sorted(eligible, key=lambda b: beta_dist(betas, b, SOURCE))
    near_identity, pool = by_dist[:2], by_dist[2:]
    assert len(pool) == N_FOLDS * FOLD_SIZE, (len(pool), pool)
    # Round-robin off the distance ordering -> every fold spans near/mid/far.
    folds = [sorted(pool[i::N_FOLDS], key=lambda b: int(b[3:])) for i in range(N_FOLDS)]
    return folds, near_identity


def fold_train_bodies(betas, test, base_bodies, leak_threshold, log):
    """Training list for one fold: base list + fold0 bodies back in, minus this
    fold's test trio, minus synthetics that sit too close to a test body."""
    reals = [b for b in base_bodies if int(b[3:]) < 100]
    syns = [b for b in base_bodies if int(b[3:]) >= 100]
    # fold0's bodies are held out of the BASE cfg but belong in training here
    # (k-fold trains on everything except its own test trio).
    train_reals = sorted(set(reals) | FOLD0 - set(test) - BROKEN,
                         key=lambda b: int(b[3:]))
    train_reals = [b for b in train_reals if b not in test]
    kept_syns, dropped = [], []
    for s in syns:
        d = min(beta_dist(betas, s, t) for t in test)
        (dropped if d < leak_threshold else kept_syns).append((s, d))
    for s, d in dropped:
        log(f"    LEAK-DROP {s}: {d:.3f} from a test body (< {leak_threshold})")
    return train_reals + [s for s, _ in kept_syns]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--leak-threshold", type=float, default=None,
                    help="beta L2 below which a synthetic is dropped from a fold. "
                         "Default: computed = smallest real-real subject distance "
                         "(~2.106). Known-bad sub121-sub13 leak was 0.34.")
    ap.add_argument("--out-root", default=REPO,
                    help="repo root to write into (tests point this at a tmpdir)")
    args = ap.parse_args(argv)

    raw = np.load(BETAS)
    betas = {k: raw[k] for k in raw.files if k.startswith("sub")}
    if args.leak_threshold is None:
        args.leak_threshold = real_human_floor(betas)
        print(f"[kfold] leak threshold = smallest real-real distance = "
              f"{args.leak_threshold:.3f}")

    base_env = open(BASE_ENV).read()
    m = re.search(r"^(\s*)subjectBodies:\s*\[(.*)\]\s*$", base_env, re.M)
    if not m:
        raise SystemExit(f"FATAL: no single-line subjectBodies in {BASE_ENV}")
    base_bodies = [b.strip().strip("'\"") for b in m.group(2).split(",")]
    for s in GLOBAL_SYN_DROP:
        if s in base_bodies:
            raise SystemExit(f"FATAL: {s} present in base cfg {BASE_ENV} -- wrong base?")
    base_train = open(BASE_TRAIN).read()
    if "full_experiment_name: smplx_teacher_src2_aug" not in base_train:
        raise SystemExit(f"FATAL: unexpected experiment name in {BASE_TRAIN}")

    folds, near_identity = assign_folds(betas)
    print(f"[kfold] fold0 (existing teachers): {sorted(FOLD0)}")
    print(f"[kfold] never-test (nearest {SOURCE}, stay in train): {near_identity}")

    out_cfg = os.path.join(args.out_root, "isaacgym/src/intermimic/data/cfg")
    written = []
    for i, test in enumerate(folds, start=1):
        name = f"kfold{i}_src2_mlp"
        print(f"[kfold] fold{i} TEST = {test} "
              f"(dist to {SOURCE}: {[f'{beta_dist(betas, b, SOURCE):.2f}' for b in test]})")
        train = fold_train_bodies(betas, set(test), base_bodies,
                                  args.leak_threshold, lambda s: print(s))
        assert not set(test) & set(train) and "sub121" not in train
        if args.dry_run:
            continue

        header = (f"# GENERATED by scripts/generate_kfold_cfgs.py -- fold{i} of the body\n"
                  f"# k-fold CV. TEST (held-out) = {test}; do not add them back.\n"
                  f"# Synthetics within {args.leak_threshold} beta-L2 of a test body are\n"
                  f"# excluded (sub121-style leak guard). Base: {os.path.basename(BASE_ENV)}\n")
        env = header + base_env.replace(
            m.group(0),
            f"{m.group(1)}subjectBodies: [{', '.join(repr(b) for b in train)}]")
        train_cfg = base_train.replace("full_experiment_name: smplx_teacher_src2_aug",
                                       f"full_experiment_name: smplx_teacher_{name}")
        slurm = SLURM_TMPL.format(name=name, test=" ".join(test))

        os.makedirs(os.path.join(out_cfg, "train/rlg"), exist_ok=True)
        for path, content in [
                (os.path.join(out_cfg, f"omomo_teacher_{name}.yaml"), env),
                (os.path.join(out_cfg, f"train/rlg/omomo_teacher_{name}.yaml"), train_cfg),
                (os.path.join(args.out_root, f"slurm_teacher_{name}.sh"), slurm)]:
            with open(path, "w") as f:
                f.write(content)
            written.append(path)
            print(f"    wrote {os.path.relpath(path, args.out_root)}")
    return written


SLURM_TMPL = """#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-{name}"
#SBATCH --output=teacher-{name}-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# k-fold CV fold (generated by scripts/generate_kfold_cfgs.py).
# TEST bodies (held out of training): {test}
# Stock MLP recipe on the sub121-free lowbuf base -- MLP is near-converged by
# ~16k epochs, so a fold does not need the full 7 days to be readable.
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{name}.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{name}.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[teacher] KFOLD {name}: test bodies {test} held out; stock MLP recipe, lowbuf 12.0"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_{name}/nn/"

# The generated cfg must actually hold the test bodies out -- fail loudly if not.
for b in {test}; do
    if grep -E "^\\s*subjectBodies:" "$CFG_ENV" | grep -q "'$b'"; then
        echo "[teacher] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2
        exit 1
    fi
done

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
    --headless \\
    --output checkpoints
"""

if __name__ == "__main__":
    main()

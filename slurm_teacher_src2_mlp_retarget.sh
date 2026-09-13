#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-mlp-ret"
#SBATCH --output=teacher-src2_mlp_retarget-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# MLP + RETARGET, STOCK recipe (constant LR 2e-5, no normval) -- "just mlp +
# retargeting". Same env cfg as the nvadlr sibling (retargetedMotionDir +
# cpuMotionData + lowbuf 12.0 + sub121-free body list); the ONLY difference is
# the train cfg's optimizer knobs:
#   vs src2_mlp_retarget_nvadlr : recipe only (normval + adaptive LR off)
#   vs src2_aug                 : retargeting under the recipe src2_aug used
#                                 (but env package differs -- see train cfg note)
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_mlp_retarget/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Reward diagnostics, baked in rather than inherited from the submitting shell
# (sbatch --export=ALL leakage is how earlier runs got them by accident).
# All are print-only; none change training.
export REWARD_BREAKDOWN=1           # per-object/body/beta-cluster/difficulty term table
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1                # why episodes end, per body
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1          # [posechk] dof-alignment sanity (pose term is on)

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_mlp_retarget_lowbuf.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_mlp_retarget.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] MLP + RETARGET (stock recipe): numObs 3230, lowbuf 12.0, cpuMotionData"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_mlp_retarget/nn/"

# Fail loudly rather than silently training the wrong cell: retargeting +
# cpuMotionData must actually be on in the env cfg this script points at.
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: cpuMotionData is not True in $CFG_ENV. Aborting." >&2
    exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retargetedMotionDir missing in $CFG_ENV -- this would" >&2
    echo "[teacher]        train plain MLP, not MLP+retarget. Aborting." >&2
    exit 1
fi

# --- auto-resume: continue from the latest checkpoint if one exists (survives the
# walltime kill / any requeue). resume_from loads mimic.pth at agent-init BEFORE
# any new save, so it never clobbers progress. Fresh start when no checkpoint yet. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    # Match the line whether the yaml says `resume_from: None` or `'None'`: the g3
    # train cfgs are UNQUOTED and the old pattern only matched the quoted form, so
    # the rewrite was a no-op and every resubmission started fresh over its own
    # checkpoints while printing RESUMING. Refuse to start if the rewrite fails.
    sed -E "s|^(\s*resume_from:)\s*'?None'?\s*$|\1 '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    if ! grep -qF "resume_from: '${CKPT}'" "$RESUME_TRAIN"; then
        echo "[teacher] ERROR: could not rewrite resume_from in $CFG_TRAIN -- refusing to" \
             "start fresh over ${CKPT}" >&2; exit 1
    fi
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[teacher] RESUMING from ${CKPT}"
else
    echo "[teacher] fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

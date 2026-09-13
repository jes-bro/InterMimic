#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_xf_aug_normval"
#SBATCH --output=teacher-src2_xf_aug_normval-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# normalize_value A/B teacher: a clone of slurm_teacher_src2_xf_aug.sh with the
# ONLY difference being CFG_TRAIN -> the _normval variant (normalize_value: True).
# Same env cfg (source=sub2 x 13 real + 40 synthetic bodies, neutral betas +
# body-norm + pose, no staging). Because full_experiment_name is _normval, it
# writes to a SEPARATE checkpoint dir and never touches the already-evaluated
# src2_xf_aug baseline -- so you can eval both and diff them directly.
#
# The question: does critic value-target normalization speed/stabilize this
# teacher's convergence vs the baseline (which has normalize_value: False)?
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_normval/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: the ENV cfg is the SAME file as the baseline -- only the TRAIN cfg differs.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_normval.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] NORMVAL A/B: source=sub2 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, normalize_value=True"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_normval/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. resume_from
# loads mimic.pth at agent-init BEFORE any new save, so it never clobbers progress.
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

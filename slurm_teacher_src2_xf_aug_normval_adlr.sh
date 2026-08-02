#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_xf_aug_normval_adlr"
#SBATCH --output=teacher-src2_xf_aug_normval_adlr-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# normval + adaptive-LR teacher: clone of slurm_teacher_src2_xf_aug_normval.sh
# with CFG_TRAIN -> the _normval_adlr variant (normalize_value: True AND
# lr_schedule: adaptive, kl_threshold 0.008). Same env cfg as the whole family.
#
# The question: the normval A/B showed value normalization learns faster mid-run
# but keeps having one-epoch KL blowups (36-155) at constant LR that collapse
# the policy. Does the KL-adaptive LR suppress those and keep the speed win?
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_normval_adlr/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: this arm is TEMPORARILY off the shared baseline env cfg. The _lowbuf copy
# is byte-identical except default_buffer_size_multiplier 20.0 -> 12.0, because the
# shared config runs at ~97% of the 44G card and the epoch-11600 resume died in
# PhysX's startup allocation (job 16433830). The other four arms on the shared yaml
# are NOT switched until this is validated -- see the _lowbuf header for the
# acceptance test (no PhysX warnings, [mem] < ~40G, rcg still ~0.618).
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_lowbuf.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_normval_adlr.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] NORMVAL+ADLR: source=sub2 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, normalize_value=True, lr_schedule=adaptive"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_normval_adlr/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. resume_from
# loads mimic.pth at agent-init BEFORE any new save, so it never clobbers progress.
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    sed "s|resume_from: 'None'|resume_from: '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
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

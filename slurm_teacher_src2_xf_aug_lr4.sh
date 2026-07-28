#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_xf_aug_lr4"
#SBATCH --output=teacher-src2_xf_aug_lr4-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# SPEED arm: clone of slurm_teacher_src2_xf_aug.sh with CFG_TRAIN -> the _lr4
# variant. The ONLY change vs the baseline is learning_rate 2e-5 -> 4e-5
# (normalize_value stays False, lr_schedule stays constant). Same env cfg.
#
# The question: the adaptive-LR arms settle at ~1.1e-5 and learn SLOWER than the
# constant-2e-5 baseline, so in this regime more LR = faster progress per frame.
# Does doubling the LR on the empirically blowup-free (no-normval, constant)
# family buy speed without reintroducing catastrophic policy updates?
#
# WATCH info/kl: ~0.06 at 2e-5. Healthy at 4e-5 is a higher but stable value
# with no spikes >1. Spikes >1 mean 4e-5 is too much -- back off toward 3e-5.
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_lr4/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: the ENV cfg is the SAME file as the baseline -- only the TRAIN cfg differs.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_lr4.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] LR4 SPEED ARM: source=sub2 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, constant LR 4e-5 (normalize_value=False)"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_lr4/nn/"

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

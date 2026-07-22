#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_xf_aug_adlr"
#SBATCH --output=teacher-src2_xf_aug_adlr-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# adaptive-LR A/B teacher: a clone of slurm_teacher_src2_xf_aug.sh with the ONLY
# difference being CFG_TRAIN -> the _adlr variant (lr_schedule: adaptive +
# kl_threshold). Same env cfg (source=sub2, no-sub121 body set on this branch,
# neutral betas + body-norm + pose). full_experiment_name=_adlr, so it writes to
# a SEPARATE checkpoint dir and never touches the constant-LR baseline.
#
# The question: does KL-driven adaptive LR speed/stabilize convergence vs the
# constant 2e-5 baseline? kl_threshold=0.008 is a first probe -- tune from the
# baseline's natural info/kl (see the run's tensorboard info/kl + info/last_lr).
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_adlr/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# ENV cfg is the SAME as the baseline (no-sub121 on this branch) -- only TRAIN differs.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_adlr.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] ADLR A/B: source=sub2, no-sub121 bodies, neutral betas + body-norm + pose, lr_schedule=adaptive (kl_threshold 0.008)"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_adlr/nn/"

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

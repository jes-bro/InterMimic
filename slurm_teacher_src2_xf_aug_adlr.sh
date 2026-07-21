#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_adlr"
#SBATCH --output=teacher-src2_xf_aug_adlr-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# ADAPTIVE-LR arm (branch source-teacher-staged): identical to the no-stage
# src2_xf_aug teacher EXCEPT lr_schedule=adaptive (KL-target 0.008, start 2e-4).
# Tests whether the constant 2e-5 was under-driving learning (near-linear curve).
# Same env config as the baseline; only the optimizer schedule differs.
# Saves to checkpoints/smplx_teacher_src2_xf_aug_adlr/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_adlr.yaml

# auto-resume from the latest checkpoint if one exists (same idiom as the baseline)
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

echo "[teacher] src2 adaptive-LR  host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/${EXP}/nn/"
python -u -m intermimic.run --task InterMimic --cfg_env "$CFG_ENV" --cfg_train "$CFG_TRAIN" --headless --output checkpoints

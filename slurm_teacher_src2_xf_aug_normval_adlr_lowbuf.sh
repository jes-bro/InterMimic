#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_nvadlr_lowbuf"
#SBATCH --output=teacher-src2_xf_aug_normval_adlr_lowbuf-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# CLEAN-PROVENANCE RERUN of the normval + adaptive-LR teacher, from epoch 0, on
# the _lowbuf env cfg (default_buffer_size_multiplier 12.0) throughout.
#
# This does NOT re-test the buffer knob -- the original run's seam already showed
# no learning effect (40.32 before -> 40.75 after, sd ~1.9, and 212 -> 223 ep/h).
# It exists so there is a single-config run to cite, instead of one whose first
# 11,603 epochs ran at multiplier 20.0 and whose last 5,441 ran at 12.0.
#
# Writes to checkpoints/smplx_teacher_src2_xf_aug_normval_adlr_LOWBUF/nn/ -- a
# DIFFERENT dir from the 17,045-epoch original, which is left untouched. Do not
# point this at the original's full_experiment_name.
#
# Reference: the original reached 46.73 smoothed reward at 2.23B frames, ahead of
# `none` (42.27) and `normval` (30.06, mid-collapse) at the same budget. Expect
# this rerun to track that curve; a large divergence means something other than
# the buffer multiplier differs.
#
# Original arm's rationale: the normval A/B showed value normalization learns
# faster mid-run but suffers recurring one-epoch KL blowups at constant LR that
# erase the lead; this arm adds the adaptive KL-based LR to suppress them.
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_normval_adlr_lowbuf/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Env cfg is _lowbuf from epoch 0 (by design -- that is the point of this rerun).
# It is byte-identical to the shared omomo_teacher_src2_xf_aug.yaml except
# default_buffer_size_multiplier 20.0 -> 12.0, which the shared config needs
# because it otherwise runs at ~97% of the 44G card and dies in PhysX's startup
# allocation on restart.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_lowbuf.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_normval_adlr_lowbuf.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] NORMVAL+ADLR (LOWBUF, fresh from epoch 0): source=sub2 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, normalize_value=True, lr_schedule=adaptive"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_normval_adlr_lowbuf/nn/"

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

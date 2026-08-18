#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-rectinj3_t4_warm"
#SBATCH --output=cari4d-bball-rectinj3_t4_warm-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# RECTINJ3_T4_WARM bball experiment: identical to smplx_cari4d_bball_overfit EXCEPT the
# object/ig/contact divergence resets are disabled (resetThresholds block) --
# noreset + HUMAN reset also off: episodes end only on body-fall / NaN / clip
# end. Tests the termination-is-the-binding-constraint hypothesis after noreset
# died 100% by the human reset (~34-step episodes) with object/ig/contact off.
#
# SEPARATE experiment: own cfgs, own checkpoint dir
# (checkpoints/smplx_cari4d_bball_rectinj3_t4_warm/nn/) -- writes NOTHING into the
# original run's directory. FRESH START only --
# no warm-starting from other runs; resubmit to resume its own checkpoints.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_rectinj3_t4_warm_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_rectinj3_t4_warm_train.yaml

# Guard: this experiment IS the resets-off arm -- refuse to run without the block.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-rectinj3_t4_warm] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*[Ff]alse' "$CFG_ENV"; then
    echo "[bball-rectinj3_t4_warm] ERROR: human reset not disabled in $CFG_ENV -- this IS the rectinj3_t4_warm knob" >&2; exit 1
fi

echo "[bball-rectinj3_t4_warm] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-rectinj3_t4_warm] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_rectinj3_t4_warm/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting (Jess rule 2026-08-11: fresh start only; resuming is
# permitted ONLY from this run's own checkpoints, for walltime resubmits).
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-rectinj3_t4_warm] RESUMING own run from ${CKPT}"
else
    echo "[bball-rectinj3_t4_warm] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved)"
fi
if [ -n "$RESUME_FROM" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    sed "s|resume_from:.*|resume_from: '${RESUME_FROM}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    CFG_TRAIN="$RESUME_TRAIN"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

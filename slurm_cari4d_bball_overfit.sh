#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d-bball-overfit"
#SBATCH --output=cari4d-bball-overfit-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# OVERFIT: one EgoExo4D basketball reconstruction (sub100_bball_000.pt, 101
# frames, dribble bounce + layup). Question: can a physics policy execute this
# 4D reconstruction at all? Known data defects ride along on purpose (23cm
# floor offset from monocular depth, last ~8 ball frames unreliable, no hoop)
# -- if the run fails, the floor offset is the first suspect, and free-flight
# reward gating is the first machinery to add.
#
# Saves to checkpoints/smplx_cari4d_bball_overfit/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Reward diagnostics (print-only; none change training) -- same set as gen-2.
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

# Env cfg is self-contained (one clip, one body, 4096 envs) and already lowbuf
# (default_buffer_size_multiplier 12.0 -- an earlier comment here wrongly said
# 20.0). 24h walltime to fit the gen-2 rotation; resubmit to auto-resume.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_train.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] CARI4D BBALL OVERFIT: 1 clip x 1 body (sub100, no retargeting), 4096 envs, numObs 3198, density 86, restitution 0.85/0.85, STOCK upstream optimizer (constant 2e-5, no normval)"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_overfit/nn/"

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

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="curriculum"
#SBATCH --output=curriculum-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Single body-conditioned policy, NO teachers, trained with a data curriculum:
# subjects fold in one at a time (sub2 identity prior first), advancing on a
# reward plateau, with inverse-cumulative-exposure per-(body,source)-pair
# sampling weights. The controller holds this one GPU and manages the training
# subprocess for every stage. See scripts/curriculum_runner.py.
#
# --resume makes requeues safe: the controller persists curriculum_work/<run>/
# state.json after each stage, so if this 48h job is requeued it continues from
# the last completed stage instead of restarting. (On a fresh run state.json is
# absent and it just starts at stage 1.)

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Parametrized so several experiments share this one script. Override per job:
#   RUN_NAME=ist      sbatch slurm_curriculum.sh                      # experimental
#   RUN_NAME=baseline SCHEDULE=coarse BALANCE=uniform sbatch slurm_curriculum.sh
RUN_NAME="${RUN_NAME:-ist}"
SCHEDULE="${SCHEDULE:-identity-source-target}"
BALANCE="${BALANCE:-inverse-exposure}"
EXPOSURE="${EXPOSURE:-estimated}"
# Set MASK_DEAD_ENVS=1 to hard-guarantee no pair leaks (zeroes dead envs' grads).
MASK_DEAD=""
[ -n "${MASK_DEAD_ENVS:-}" ] && MASK_DEAD="--mask-dead-envs"
# NETWORK=transformer for the temporal-transformer policy (vs default MLP).
# Use a DIFFERENT RUN_NAME for the transformer arm so its work/checkpoint dirs
# (and warm-start chain) don't collide with the MLP run.
NETWORK="${NETWORK:-mlp}"
# NUM_ENVS: lower it (e.g. 2048) for later substages -- once many subjects are
# folded in, all their motion clips + 4096 envs of physics state can exceed the
# GPU's memory and PhysX OOMs (cudaErrorMemoryAllocation / illegal access).
NUM_ENVS="${NUM_ENVS:-4096}"
# BETAS_FILE: stock gendered betas by default. Point at omomo_betas_neutral.npz
# for the shared-neutral-space conditioning experiment (bodies/motion unchanged).
BETAS_FILE="${BETAS_FILE:-scripts/omomo_betas.npz}"
# Synthetic target-only training bodies (0=off). Use BETAS_FILE=omomo_betas_neutral_aug.npz
# and make sure smplx_omomo_sub100..sub<99+N>.xml exist (generate_per_subject_mjcfs.py).
NUM_SYNTHETIC="${NUM_SYNTHETIC:-0}"
SYN_ARGS=""
[ "$NUM_SYNTHETIC" -gt 0 ] && SYN_ARGS="--num-synthetic $NUM_SYNTHETIC \
    --synthetic-position ${SYNTHETIC_POSITION:-append} \
    --synthetic-mode ${SYNTHETIC_MODE:-batched} \
    --synthetic-batch-size ${SYNTHETIC_BATCH_SIZE:-5}"
# Body-normalized reward (height-normalize pose error). With synthetic bodies set
# SUBJECT_HEIGHTS_FILE=scripts/synthetic_heights.json so sub100+ have heights.
BODY_NORM=""
[ -n "${BODY_NORM_REWARD:-}" ] && BODY_NORM="--body-norm-reward"
[ -n "${SUBJECT_HEIGHTS_FILE:-}" ] && BODY_NORM="$BODY_NORM --subject-heights-file $SUBJECT_HEIGHTS_FILE"

# Rename the job so `squeue` shows which run this is (widen with:
#   squeue --me -o "%.18i %.24j %.8T %.10M")
scontrol update JobId="$SLURM_JOB_ID" JobName="c-${RUN_NAME}" 2>/dev/null || true

python scripts/curriculum_runner.py \
    --run-name "$RUN_NAME" --schedule "$SCHEDULE" --balance "$BALANCE" \
    --exposure "$EXPOSURE" $MASK_DEAD --network "$NETWORK" \
    --num-envs "$NUM_ENVS" --betas-file "$BETAS_FILE" $SYN_ARGS $BODY_NORM --resume

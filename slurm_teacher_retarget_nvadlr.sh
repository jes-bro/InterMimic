#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-ret-nvadlr"
#SBATCH --output=teacher-retarget_nvadlr-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# RETARGET + cpuMotion + normalize_value + adaptive LR (kl 0.06).
# Arm A of the 2x2. Its partner is ..._nvadlr_cpumotion (same optimizer, NO
# retargeting), so A-vs-C isolates retargeting under the best-known optimizer
# settings rather than under the bare baseline the earlier retarget arms used
# (constant 2e-5, no KL guard -- the config with no brake on an excursion, which
# is where the ~90% dips came from).
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src2_xf_aug_retarget_nvadlr/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Job 16502149 post-mortem: lowbuf + cpuMotionData were BOTH on, yet GPU-used
# crept 42.4 -> 44.3/44G over 344k steps while torch-allocated stayed flat at
# 12.5G, until a ~200MB PhysX narrowphase spike OOM'd it at epoch 10,775. Two
# countermeasures:
#   1. cap torch allocator fragmentation (the streamed per-step motion gathers
#      churn transient GPU tensors; PhysX cannot use memory torch has cached)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
#   2. env count sized to the card: 2048 default (44G L40S -- roughly halves the
#      PhysX footprint that crept into the ceiling). On a bigger GPU:
#      NUM_ENVS=4096 sbatch slurm_teacher_retarget_nvadlr.sh
# minibatch_size 16384 divides both: 2048*32=65536 (4) and 4096*32=131072 (8).
NUM_ENVS="${NUM_ENVS:-2048}"

# Reward diagnostics, baked in rather than inherited from the submitting shell.
# All are print-only; none change training.
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_retarget_nvadlr_lowbuf.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_retarget_nvadlr.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] RETARGET + nvadlr: lowbuf 12.0, cpuMotionData, num_envs=$NUM_ENVS, alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_retarget_nvadlr/nn/"

# Fail loudly rather than silently measuring the wrong thing: this test is
# meaningless if the cfg it points at does not actually have the knob on.
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: cpuMotionData is not True in $CFG_ENV -- this run would" >&2
    echo "[teacher]        just re-measure the baseline. Aborting." >&2
    exit 1
fi

# --- auto-resume: continue from the latest checkpoint if one exists (survives the
# walltime kill / any requeue). resume_from loads mimic.pth at agent-init BEFORE
# any new save, so it never clobbers progress. Fresh start when no checkpoint yet. ---
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
    --num_envs "$NUM_ENVS" \
    --headless \
    --output checkpoints

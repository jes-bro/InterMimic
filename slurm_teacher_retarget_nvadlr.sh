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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_retarget_nvadlr.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_retarget_nvadlr.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] THROUGHPUT TEST: cpuMotionData=True vs job 16390586 (~7100 fps step, 43.7/44G)"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_retarget_cpumotion/nn/"

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
    --headless \
    --output checkpoints

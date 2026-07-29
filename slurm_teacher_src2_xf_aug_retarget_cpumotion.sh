#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-retarget-cpumotion"
#SBATCH --output=teacher-src2_xf_aug_retarget_cpumotion-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# THROUGHPUT TEST, not a science arm. Does cpuMotionData recover the ~20% fps the
# retarget arm lost? Everything except the storage location of the reference
# tensors is identical to slurm_teacher_src2_xf_aug_retarget.sh, and cpuMotionData
# does not change any VALUE -- so any fps difference is attributable to it alone.
#
# BASELINE TO BEAT (job 16390586, cpuMotionData off):
#   fps step  7071 - 7410      fps total  5340 - 5610
#   [mem] motion tensors 7.87G on GPU | GPU used 43.5-43.7/44G from step 201 on
# TARGET: the src2_xf_aug baseline's ~9000 fps step.
#
# 2h is deliberate: ~24s/epoch means ~300 epochs, far more than the ~30 needed to
# read a stable fps. This is not meant to train anything -- kill it once the
# number is clear.
#
# NOTE ON auto-resume: slurm_teacher_src2_xf_aug_retarget.sh resumes from its
# latest checkpoint. That is deliberately ABSENT here. A throughput test must
# start fresh every time, or you measure a warm run against a cold one. If a
# checkpoint dir exists from a previous test, delete it before re-running.
#
# Runs from repo root.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_retarget_cpumotion.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_retarget_cpumotion.yaml
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

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
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
# It won: +43% fps step (7150 -> 10250), +32% fps total (5440 -> 7180), i.e.
# 24.1 -> 18.3 sec/epoch. So this is now a real training arm, and the walltime is
# 7d rather than the original 2h. (scontrol could not extend the running test job
# -- raising a live job's TimeLimit is operator-only on this cluster.)
#
# NOTE ON auto-resume: originally absent here on purpose (a throughput test must
# start cold, or you compare a warm run to a cold one). Now that this is a real
# training arm it needs to survive requeues, so the block from
# slurm_teacher_src2_xf_aug_retarget.sh is in. If you ever want a clean fps
# measurement again, delete the checkpoint dir first so it starts fresh.
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

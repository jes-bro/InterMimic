#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# JOB ARRAY: train all 17 source teachers (src1..src17), at most 4 GPUs at a time.
# The "%4" throttle is the whole point -- SLURM runs <=4 array tasks concurrently
# no matter how many are queued, so this self-limits to 4 GPUs without an admin cap.
#SBATCH --array=1-17%4

#SBATCH --job-name="tch-array"
#SBATCH --output=teacher-array-%A_%a.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# One array task per source: SLURM_ARRAY_TASK_ID = N -> trains src{N}. Replaces the
# 17 separate slurm_teacher_src{N}_xf_aug.sh scripts with a single throttled array.
# On this branch (source-teacher-drop-sub121) the env cfgs already have sub121
# removed, so this trains the no-sub121 teachers.
#
#   sbatch slurm_teacher_array.sh                 # all 17, 4 at a time
#   sbatch --array=1,2,6,9%2 slurm_teacher_array.sh   # only some sources, 2 at a time
#
# Change the concurrency by editing %4 (or overriding --array on the CLI). Each
# task auto-resumes from its own latest checkpoint, exactly like the single scripts.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

N="${SLURM_ARRAY_TASK_ID:?this script must be run as a job array (SLURM_ARRAY_TASK_ID unset)}"
SRC="src${N}"

CFG_ENV="isaacgym/src/intermimic/data/cfg/omomo_teacher_${SRC}_xf_aug.yaml"
CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_${SRC}_xf_aug.yaml"

# No-silent-fallback: a missing cfg means we'd train the wrong/nothing -- fail loud.
[ -f "$CFG_ENV" ]   || { echo "[array] ERROR: missing env cfg $CFG_ENV";   exit 2; }
[ -f "$CFG_TRAIN" ] || { echo "[array] ERROR: missing train cfg $CFG_TRAIN"; exit 2; }

# Give each array task a readable name in squeue.
scontrol update JobId="$SLURM_JOB_ID" JobName="tch-${SRC}_xf_aug" 2>/dev/null || true

echo "[array] task $N -> $SRC  host=$(hostname) job=$SLURM_JOB_ID"
echo "[array] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints"

# --- auto-resume (same as the single teacher scripts): continue from latest ckpt.
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    sed "s|resume_from: 'None'|resume_from: '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[array] $SRC RESUMING from ${CKPT}"
else
    echo "[array] $SRC fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

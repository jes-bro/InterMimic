#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src4_xf_aug_obj"
#SBATCH --output=teacher-src4_xf_aug_obj-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# SOURCE-TEACHER + OBJECT AUG run: source=sub4 driving 13 real + 40 synthetic
# bodies AND size/pose-perturbed objects (scale/yaw/translate, drop object
# terms + hold), body-conditioned (neutral betas), NO staging -- one RL run.
#
# The question this answers: can a per-source teacher learn 1-source x 14-bodies
# without a fold-in curriculum? Watch mean_rewards: if it climbs and converges
# to a good level across bodies, the teacher-student direction is validated
# (14 such teachers -> 1 distilled student, no staging, ~14 teachers not 196).
# If it stalls/struggles, staging wasn't avoidable and the curriculum stands.
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src4_xf_aug_obj/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src4_xf_aug_obj.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src4_xf_aug_obj.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] source=sub4 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, no staging"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src4_xf_aug_obj/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists (survives the
# July 3-5 outage / any requeue). resume_from loads mimic.pth at agent-init BEFORE
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

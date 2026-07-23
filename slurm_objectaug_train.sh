#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="oa-train"
#SBATCH --output=objectaug-train-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# PHYSICS objectAug TRAINING (v1): TRANSFORMER policy with per-object realistic ranges
# (scripts/object_perturbation_ranges.json) + [scale,mass] observation conditioning.
# This is the real training run, not the masschk validation. Before trusting a long
# run, confirm from the first log lines:
#   1. [objectAug] per-object ranges from ...: largetable[..]^2, woodchair[..]^2, ...
#   2. [objectAug] obs conditioning terms=['scale', 'mass'] -> cond_dim=2
#   3. num_obs: 6532            (matches the cfg; divisible by 4 for the transformer)
#   4. mean_rewards climbs (not stuck ~0, not NaN)
# OBJECTAUG_DEBUG=1 also prints [masschk] so you can re-verify mass ~ aug**massExp.
# Saves to checkpoints/smplx_objectaug_train/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

export OBJECTAUG_DEBUG="${OBJECTAUG_DEBUG:-1}"   # print [masschk]; set 0 to silence

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_objectaug_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_objectaug_train.yaml
echo "[objectaug-train] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_objectaug_train/nn/"
echo "[objectaug-train] env=$CFG_ENV train=$CFG_TRAIN"

# --- auto-resume: continue from the latest checkpoint if one exists (survives
# requeue). resume_from loads mimic.pth at agent-init BEFORE any new save. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    sed "s|resume_from: 'None'|resume_from: '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[objectaug-train] RESUMING from ${CKPT}"
else
    echo "[objectaug-train] fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

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

python scripts/curriculum_runner.py \
    --run-name "$RUN_NAME" --schedule "$SCHEDULE" --balance "$BALANCE" \
    --exposure "$EXPOSURE" $MASK_DEAD --resume

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-eval"
#SBATCH --output=bball-eval-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Quantitative eval of the CARI4D bball overfit policy: Start-mode rollouts of
# the ONE clip via the committed eval twin cfg. With a single motion the
# printed Success Rate is 0%/100% ("did the best attempt survive all 101
# frames"); the informative numbers are avg execution steps (how far into the
# clip it gets -- dying around the dribble's flight frames implicates the
# missing free-flight gating) and the human/object pose errors.
#
#   sbatch slurm_cari4d_bball_eval.sh
#   CHECKPOINT=checkpoints/smplx_cari4d_bball_overfit/nn/mimic_00020000.pth sbatch slurm_cari4d_bball_eval.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Death-mode table alongside the metrics (print-only).
export TERM_REASON=1
export TERM_REASON_EVERY=2000

CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_cari4d_bball_overfit/nn/mimic.pth}"
NUM_ENVS="${NUM_ENVS:-512}"
[ -f "$CHECKPOINT" ] || { echo "[bball-eval] ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 2; }

echo "[bball-eval] ckpt=$CHECKPOINT num_envs=$NUM_ENVS host=$(hostname) job=$SLURM_JOB_ID"

python -u -m intermimic.run --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_eval.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_train.yaml \
    --test --checkpoint "$CHECKPOINT" \
    --headless --num_envs "$NUM_ENVS"

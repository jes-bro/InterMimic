#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cp_smoke"
#SBATCH --output=cp_smoke-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Smoke-test a single cross-pair teacher cfg with 10 train iters from random
# init. Verifies env construction, motion loading, and that a checkpoint
# gets written. Don't submit the 18 real teachers until this passes.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"

python -u -m intermimic.run --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_crosspair_b10_s2_largetable.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_crosspair_b10_s2_largetable.yaml \
    --headless --num_envs 64 --max_iterations 10

echo ""
echo "=== smoke test done; checkpoint should be at:"
ls -lh checkpoints/smplx_crosspair_b10_s2_largetable/nn/ 2>&1 || echo "(no checkpoint dir!)"

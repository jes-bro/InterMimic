#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="dst_smoke"
#SBATCH --output=dst_smoke-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Smoke test: small distillation run to verify teacher loading, env construction,
# BC+PPO loop, and checkpoint saving. Should complete in <10 minutes.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"

python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_distill_largetable.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_largetable.yaml \
    --headless --num_envs 64 --max_iterations 50

echo ""
echo "=== smoke test done; checkpoint should be at:"
ls -lh checkpoints/smplx_distill_largetable/nn/ 2>&1 || echo "(no checkpoint dir!)"

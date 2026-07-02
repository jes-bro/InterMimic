#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="oa-test"
#SBATCH --output=objectaug-test-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# PHYSICS objectAug CORE-PORT VALIDATION. OBJECTAUG_DEBUG=1 prints [masschk] so you
# can confirm mass tracks scale**massExp (not scale**3) before trusting it. Watch:
#   1. [objectAug] ON: ...            -> config parsed
#   2. [masschk] env.. aug=.. total_mass=..  -> smaller aug => smaller mass, ~aug^massExp
#   3. mean_rewards climbs (not stuck at ~0, not NaN) -> reward gating + hold + reset
#      relaxation work (perturbed objects don't insta-reset the episode)
# Saves to checkpoints/smplx_objectaug_test/nn/. A validation run -- kill once the
# above look right; it doesn't need to converge.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

export OBJECTAUG_DEBUG=1   # print the [masschk] verification

echo "[objectaug-test] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_objectaug_test/nn/"

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_objectaug.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_objectaug.yaml \
    --headless \
    --output checkpoints

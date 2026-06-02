#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair distillation: Multi-object distillation on largetable+woodchair (16 teachers). body-normalized reward in PPO loss (divides pose-error by per-env body height).
# Saves to checkpoints/smplx_distill_both_normreward/nn/.
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_distill_both_normreward.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_both_normreward.yaml \
    --headless \
    --output checkpoints

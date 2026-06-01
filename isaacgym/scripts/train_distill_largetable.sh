#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair distillation: Single-object distillation on largetable (8 teachers). default reward (no body-shape normalization).
# Saves to checkpoints/smplx_distill_largetable/nn/.
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_distill_largetable.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_largetable.yaml \
    --headless \
    --output checkpoints

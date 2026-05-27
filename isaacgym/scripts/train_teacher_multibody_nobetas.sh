#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Multibody fine-tune ablation: NO BETAS CONDITIONING.
# - Same 5 subjects, same per-subject MJCFs, same env physics
# - But no (source_betas, target_betas) appended to obs (numObs = 3198)
# - Resumes from the ORIGINAL sub2.pth (not widened) since obs stays at 3198
# Goal: isolate whether explicit betas conditioning is load-bearing vs just
# implicit body randomization being enough.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_multibody_nobetas.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_nobetas.yaml \
    --headless \
    --output checkpoints

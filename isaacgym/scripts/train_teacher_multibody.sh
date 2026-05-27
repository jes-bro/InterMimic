#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Multibody body-conditioned fine-tune.
# - Loads K=5 per-subject MJCFs (sub10, sub17, sub9, sub2, sub3) round-robin
# - Appends (source_betas, target_betas) 32-dim to obs (numObs=3230)
# - Resumes from sub2_betas.pth (widen_checkpoint.py output, first layer
#   widened by 32 zero columns, optimizer state dropped)
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_multibody.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --headless \
    --checkpoint checkpoints/smplx_teachers/sub2_betas.pth \
    --output checkpoints

#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Curriculum stage 2: 5 bodies × 2 sources (sub2, sub10).
# Warm-starts from stage-1 (5 bodies × sub2 only) so policy already
# knows multi-body control for sub2's motion; only new thing is sub10.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_multibody_stage2.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_stage2.yaml \
    --headless \
    --output checkpoints

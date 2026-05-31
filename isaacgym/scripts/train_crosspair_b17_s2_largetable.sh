#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair teacher: body=sub17 x source=sub2 x object=largetable.
# Trains from random init (no warm-start) so sub2 and sub6 sources are symmetric.
# Saves to checkpoints/smplx_crosspair_b17_s2_largetable/nn/.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_crosspair_b17_s2_largetable.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_crosspair_b17_s2_largetable.yaml \
    --headless \
    --output checkpoints

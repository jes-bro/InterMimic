#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair teacher: body=sub17, source=sub2
# Resumes from mimic_00003000.pth (stage-1 multibody prior).
# Saves to checkpoints/smplx_crosspair_b17_s2/nn/.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_crosspair_b17_s2.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_crosspair_b17_s2.yaml \
    --headless \
    --output checkpoints

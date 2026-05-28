#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Identity-pair teacher for sub3: sub3's body × sub3's motion.
# Resumes from sub2_betas.pth (widened canonical) to leverage retargeting prior.
# Saves to checkpoints/smplx_teacher_sub3/nn/.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_teacher_sub3.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_sub3.yaml \
    --headless \
    --output checkpoints

#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cold-start multibody training (counterpart to train_teacher_multibody.sh).
# - Same env (5 bodies, 5 source subjects, betas conditioning)
# - But resumes from sub2_betas.pth (widened canonical teacher, no multi-body prior)
# - Saves to checkpoints/smplx_multibody_coldstart/nn/
# Use this in parallel with the warmstart variant for a learning-curve comparison.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_multibody.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_coldstart.yaml \
    --headless \
    --output checkpoints

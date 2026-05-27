#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Stage 2 with gentler hyperparams: LR 5e-6 (vs 2e-5), mini_epochs 3 (vs 6).
# Same env + data as stage 2; same warm-start (mimic_00003000.pth).
# Use if the default-hyperparam stage 2 run isn't converging — slower but
# more stable adaptation to the added source subject.
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_multibody_stage2.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_stage2_lowlr.yaml \
    --headless \
    --output checkpoints

#!/bin/sh
# Replay a CARI4D-reconstructed clip in Isaac Gym. Pipeline:
#   1. scripts/cari4d_to_interact.py     (local)   produces InterAct-format .npz + mesh
#   2. scripts/run_interact2mimic.py     (cluster) wraps interact2mimic.py with mesh=True
#                                                  → subject-shape convex-hull MJCFs
#   3. scripts/cari4d_finalize.py        (cluster) installs outputs into this repo
#   4. this script                                 replays them
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_cari4d.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --test \
    --play_dataset \
    --num_envs 16

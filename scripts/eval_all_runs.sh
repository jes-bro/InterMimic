#!/bin/sh
# Eval the latest "best" checkpoint of each training variant on the same
# 7-subject suite, dump one CSV per run. Submit this as a SLURM job —
# one job covers all variants serially (~7 subjects × ~10 min each × N
# variants = N hours).
#
# Usage on the cluster:
#   sbatch <slurm-wrapper-that-runs-this>
#
# Edit RUNS below to add/remove variants you care about.
#
# Output: eval_<variant_name>.csv at repo root, one row per subject.

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

SUBJECTS="sub2 sub10 sub3 sub17 sub9 sub1 sub5"
NUM_ENVS=1024
TIMEOUT=900   # 15 min per subject

# Each entry: <name>:<checkpoint_path>:<base_yaml>:<train_yaml>
# Comment out runs whose checkpoints don't exist yet.
RUNS="
main:checkpoints/smplx_multibody_stage2/nn/mimic.pth:isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml:isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml
nobetas:checkpoints/smplx_multibody_nobetas/nn/mimic.pth:isaacgym/src/intermimic/data/cfg/omomo_test_multibody_nobetas.yaml:isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_nobetas.yaml
stage1:checkpoints/smplx_multibody_sub2only/nn/mimic.pth:isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml:isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml
"

for ENTRY in $RUNS; do
    [ -z "$ENTRY" ] && continue
    NAME=$(echo "$ENTRY" | cut -d: -f1)
    CKPT=$(echo "$ENTRY" | cut -d: -f2)
    BASE_YAML=$(echo "$ENTRY" | cut -d: -f3)
    TRAIN_YAML=$(echo "$ENTRY" | cut -d: -f4)

    if [ ! -f "$REPO_ROOT/$CKPT" ]; then
        echo "[eval_all_runs] SKIP $NAME — checkpoint $CKPT not found"
        continue
    fi

    OUT_CSV="eval_${NAME}.csv"
    echo ""
    echo "================================================================"
    echo "[eval_all_runs] Running $NAME -> $OUT_CSV"
    echo "  checkpoint: $CKPT"
    echo "================================================================"

    python -u scripts/eval_per_subject.py \
        --checkpoint "$CKPT" \
        --subjects $SUBJECTS \
        --output-csv "$OUT_CSV" \
        --num-envs $NUM_ENVS \
        --timeout-per-subject $TIMEOUT \
        --base-yaml "$BASE_YAML" \
        --train-yaml "$TRAIN_YAML"
done

echo ""
echo "[eval_all_runs] All done. CSVs:"
ls -1 eval_*.csv

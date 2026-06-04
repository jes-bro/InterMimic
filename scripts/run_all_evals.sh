#!/bin/bash
# Run all 9 evaluations sequentially:
#   - 8 per-object × per-method (2 objects × 4 methods)
#   - 1 curriculum stage 2 (sub2 source only, fair comparison)
#
# Logs each run separately. Continues on failure (one bad eval doesn't
# stop the rest).
#
# Usage:
#   ./scripts/run_all_evals.sh
# Or to survive ssh disconnects:
#   nohup ./scripts/run_all_evals.sh > /tmp/run_all_evals.log 2>&1 &

set -u  # error on undefined vars (but not on command failure)
cd "$(dirname "$0")/.."

# --- environment setup ---
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- output ---
mkdir -p ~/eval_results
mkdir -p ~/eval_logs

# --- common args ---
BODIES="sub10 sub17 sub9 sub3 sub1 sub5"
SOURCES="sub2 sub6"
SOURCES_SUB2_ONLY="sub2"
NUM_ENVS=1024
TIMEOUT=600

# --- checkpoints (per-method) ---
CKPT_FULL=/home/ubuntu/vanilla_intermimic/output/smplx_distill_both_normreward/nn/mimic.pth
CKPT_REWARD_ABL=/home/ubuntu/reward_ablation/output/smplx_distill_both/nn/mimic.pth
CKPT_BETAS_ABL=/home/ubuntu/nobetas_ablation/output/smplx_distill_both_nobetas_normreward/nn/mimic.pth
CKPT_VANILLA=/home/ubuntu/vanilla_intermimic/checkpoints/student.pth
CKPT_CURRICULUM=checkpoints/curriculum_stage2.pth

# --- yamls (test + train) ---
TEST_3230_LT=isaacgym/src/intermimic/data/cfg/omomo_test_multibody_largetable.yaml
TEST_3230_WC=isaacgym/src/intermimic/data/cfg/omomo_test_multibody_woodchair.yaml
TEST_3198_LT=isaacgym/src/intermimic/data/cfg/omomo_test_multibody_nobetas_largetable.yaml
TEST_3198_WC=isaacgym/src/intermimic/data/cfg/omomo_test_multibody_nobetas_woodchair.yaml
TEST_3230=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml

TRAIN_FULL=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_both_normreward.yaml
TRAIN_REWARD_ABL=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_both.yaml
TRAIN_BETAS_ABL=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_both_nobetas_normreward.yaml
TRAIN_VANILLA=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_nobetas.yaml
TRAIN_CURRICULUM=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody_stage2.yaml

# --- helper: run one eval, log to file, never fail-out the script ---
run_eval() {
    local name="$1" ckpt="$2" test_yaml="$3" train_yaml="$4" sources="$5"
    local out_csv="$HOME/eval_results/${name}.csv"
    local log="$HOME/eval_logs/${name}.log"

    echo ""
    echo "====================================================="
    echo "=== [$(date '+%H:%M:%S')] running: $name"
    echo "====================================================="
    if [ ! -f "$ckpt" ]; then
        echo "SKIP $name: checkpoint not found at $ckpt"
        return
    fi
    python scripts/eval_per_pair.py \
        --checkpoint "$ckpt" \
        --bodies $BODIES --sources $sources \
        --output-csv "$out_csv" \
        --base-yaml "$test_yaml" \
        --train-yaml "$train_yaml" \
        --num-envs $NUM_ENVS \
        --timeout-per-pair $TIMEOUT \
        > "$log" 2>&1
    rc=$?
    echo "=== [$(date '+%H:%M:%S')] done: $name (rc=$rc) -> $out_csv"
    echo "       log: $log"
    return 0
}

# --- 8 per-object evals (4 methods × 2 objects) ---

run_eval "full_method_largetable"      "$CKPT_FULL"        "$TEST_3230_LT" "$TRAIN_FULL"       "$SOURCES"
run_eval "full_method_woodchair"       "$CKPT_FULL"        "$TEST_3230_WC" "$TRAIN_FULL"       "$SOURCES"
run_eval "reward_ablation_largetable"  "$CKPT_REWARD_ABL"  "$TEST_3230_LT" "$TRAIN_REWARD_ABL" "$SOURCES"
run_eval "reward_ablation_woodchair"   "$CKPT_REWARD_ABL"  "$TEST_3230_WC" "$TRAIN_REWARD_ABL" "$SOURCES"
run_eval "betas_ablation_largetable"   "$CKPT_BETAS_ABL"   "$TEST_3198_LT" "$TRAIN_BETAS_ABL"  "$SOURCES"
run_eval "betas_ablation_woodchair"    "$CKPT_BETAS_ABL"   "$TEST_3198_WC" "$TRAIN_BETAS_ABL"  "$SOURCES"
run_eval "vanilla_largetable"          "$CKPT_VANILLA"     "$TEST_3198_LT" "$TRAIN_VANILLA"    "$SOURCES"
run_eval "vanilla_woodchair"           "$CKPT_VANILLA"     "$TEST_3198_WC" "$TRAIN_VANILLA"    "$SOURCES"

# Curriculum eval intentionally omitted — checkpoint not on this machine.
# To add later: transfer checkpoint to checkpoints/curriculum_stage2.pth and add
# run_eval "curriculum_stage2_sub2" "$CKPT_CURRICULUM" "$TEST_3230" "$TRAIN_CURRICULUM" "$SOURCES_SUB2_ONLY"

# --- summary ---
echo ""
echo "====================================================="
echo "=== ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================="
ls -lh ~/eval_results/
echo ""
echo "Individual logs in ~/eval_logs/"

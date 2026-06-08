#!/bin/bash
# Verify the compressed DAgger schedule + warm-start yaml edits are in place.
# Run from the repo root, or from anywhere — REPO is auto-detected from the
# script's location.
#
# Usage:
#   bash scripts/verify_schedule_update.sh
#
# Expected output: beta_t line should start with `0.7 -`, EV gate epoch is
# 2900, critic warmup elif is 2300, and all 3 yamls have load_checkpoint:
# True + a non-'None' resume_from + a _v2 experiment name.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

echo "Repo root: $REPO"
echo ""
echo "=== Schedule constants in intermimic_agent_distill.py ==="
grep -n "beta_t\s*=\|epoch_num > 2900\|epoch_num > 2300\|actor_update_num / 800" \
    "$REPO/isaacgym/src/intermimic/learning/intermimic_agent_distill.py"

echo ""
echo "=== Yaml resume + experiment name ==="
for f in omomo_distill_both_normreward.yaml omomo_distill_both_nobetas_normreward.yaml omomo_distill_both.yaml; do
    echo "--- $f ---"
    grep -n "load_checkpoint:\|full_experiment_name:\|resume_from:" \
        "$REPO/isaacgym/src/intermimic/data/cfg/train/rlg/$f"
done

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --job-name="rt-check"
#SBATCH --output=rt-check-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# CPU-ONLY validation of the contact retargeter (no GPU, no Isaac Gym). Runs the
# cheap correctness gate so you don't have to sit on the login node:
#   1. FK self-test  -- identity sub2->sub2 must be an exact no-op (0.00cm).
#   2. index test    -- body-major expansion / _to_body_block invariants.
#   3. small batch   -- retarget a few clips onto a few bodies, print per-body
#                       contact error before->after (proof it works on cluster data).
# Read rt-check-<jobid>.out. If 1 & 2 pass and 3 shows large->small, the retargeter
# is validated on the cluster; the only thing left needing a GPU is the Isaac Gym
# smoke run (loader consumption).
#
#   sbatch slurm_retarget_check.sh
#   TARGETS="sub2 sub16 sub10 sub1" LIMIT=5 sbatch slurm_retarget_check.sh
#
# Override TARGETS/LIMIT/ITERS/OUT via env vars. Runs from repo root.

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
# No LD_LIBRARY_PATH / no Isaac Gym import needed -- this is pure torch CPU.
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

TARGETS="${TARGETS:-sub2 sub16 sub10}"
LIMIT="${LIMIT:-3}"
ITERS="${ITERS:-300}"
OUT="${OUT:-/tmp/rt_check_${SLURM_JOB_ID}}"

echo "[rt-check] host=$(hostname) job=$SLURM_JOB_ID  targets=($TARGETS) limit=$LIMIT iters=$ITERS"
echo

echo "================ 1. FK self-test (identity must be a no-op) ================"
python3 scripts/retarget_contact.py --selftest
echo

echo "================ 2. index invariants (offline) ============================"
python3 tests/test_retarget_index.py
echo

echo "================ 3. small batch: contact error before -> after ============"
python3 scripts/retarget_contact.py --batch --source sub2 \
    --targets $TARGETS --limit "$LIMIT" --iters "$ITERS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --out-dir "$OUT"
echo
echo "[rt-check] done. Expect: self-test PASS, 'all index invariants hold', and"
echo "           sub2 ~0.00 (no-op) with sub16/sub10 large->small (~0.1-0.5 cm)."

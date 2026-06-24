#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="curr-eval"
#SBATCH --output=curr-eval-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Eval a curriculum checkpoint -> REAL metrics (success rate, human/object pose
# error, avg steps) per (body, source) pair, written to a CSV. Uses the existing
# eval_per_pair.py + omomo_test_multibody.yaml (Start mode, enableEvaluation,
# numObs 3230 -- matches the curriculum checkpoint arch [1024,1024,512]). This
# turns a checkpoint you ALREADY have into a result -- no more training needed.
#
# Default: the 6 folded-in (in-distribution) subjects, full body x source matrix.
#   diagonal (body==source) = identity = "drive each body with its own motion"
#   off-diagonal            = cross-retarget
# Override via env vars:
#   CHECKPOINT=path/to.pth   BODIES="sub2 sub3"   SOURCES="sub2"
#   NUM_ENVS=1024  TIMEOUT=900  OUT=eval_results/foo.csv
#
# FAST first read (~12 min, 1 pair) -- do this before the full matrix:
#   BODIES=sub2 SOURCES=sub2 sbatch slurm_eval_curriculum.sh
# Held-out generalization (never-trained bodies):
#   BODIES="sub4 sub10 sub16" SOURCES="sub2 sub9" sbatch slurm_eval_curriculum.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_curriculum_ist_meas_s06a_sub1_identity/nn/mimic_00007700.pth}"
BODIES="${BODIES:-sub1 sub2 sub3 sub5 sub9 sub17}"     # the 6 folded-in subjects
SOURCES="${SOURCES:-sub1 sub2 sub3 sub5 sub9 sub17}"
NUM_ENVS="${NUM_ENVS:-1024}"
TIMEOUT="${TIMEOUT:-900}"
OUT="${OUT:-eval_results/curriculum_ist_meas_7700_indist.csv}"
# The base test config restricts to dataObjects ['largetable','woodchair'] (a
# student-eval leftover) -- the curriculum trained on ALL objects, so by default
# we drop that restriction. Set ALL_OBJECTS=0 to keep the base config's filter.
ALL_OBJ=""
[ "${ALL_OBJECTS:-1}" = "1" ] && ALL_OBJ="--all-objects"

mkdir -p "$(dirname "$OUT")"
echo "[eval] checkpoint = $CHECKPOINT"
echo "[eval] bodies=($BODIES)  x  sources=($SOURCES)  all_objects=${ALL_OBJECTS:-1}"
echo "[eval] -> $OUT"
[ -f "$CHECKPOINT" ] || { echo "[eval] ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

python -u scripts/eval_per_pair.py \
    --checkpoint "$CHECKPOINT" \
    --bodies $BODIES \
    --sources $SOURCES \
    --output-csv "$OUT" \
    --num-envs "$NUM_ENVS" \
    --timeout-per-pair "$TIMEOUT" $ALL_OBJ

echo
echo "================ EVAL SUMMARY ================"
echo "full CSV: $OUT"
cat "$OUT"
echo "--- identity pairs only (body == source) ---"
awk -F, 'NR==1 || $1==$2' "$OUT"
echo "============================================="

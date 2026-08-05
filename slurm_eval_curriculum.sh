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

# REQUIRED (no silent defaults): a stale hardcoded checkpoint/OUT would eval
# the WRONG run and mis-attribute the CSV; a missing BETAS_FILE would fall back
# to the base yaml's GENDERED betas (omomo_betas.npz) and silently corrupt the
# 32 beta obs dims for any neutral/aug run. Fail loudly instead.
require() {   # require VAR "hint"
    eval "_v=\${$1:-}"
    if [ -z "$_v" ]; then
        echo "[eval] ERROR: $1 is required (no default). $2" >&2
        exit 2
    fi
}
require CHECKPOINT "e.g. CHECKPOINT=checkpoints/<run>/nn/mimic_XXXX.pth"
require OUT        "e.g. OUT=eval_results/<run>__<ckpt>__heldout.csv"
require BETAS_FILE "must match the run's training betas (scripts/omomo_betas.npz gendered, _neutral, or _neutral_aug); set BETAS_FILE=none only if the run trained with no betas"

# Diagnostics: forwarded EXPLICITLY so they cannot be lost between the caller's
# shell, sbatch, and eval_per_pair's subprocess. TERM_REASON=1 prints the
# per-body termination table (completed/fell/nan/ig/contact); REWARD_BREAKDOWN=1
# the per-group reward terms. Both are read from os.environ inside intermimic.py.
export TERM_REASON="${TERM_REASON:-0}"
export TERM_REASON_EVERY="${TERM_REASON_EVERY:-2000}"
export REWARD_BREAKDOWN="${REWARD_BREAKDOWN:-0}"
[ "$TERM_REASON" = 1 ] && echo "[eval] TERM_REASON=1 (every $TERM_REASON_EVERY steps)"
[ "$REWARD_BREAKDOWN" = 1 ] && echo "[eval] REWARD_BREAKDOWN=1"

BODIES="${BODIES:-sub1 sub2 sub3 sub5 sub9 sub17}"     # the 6 folded-in subjects
SOURCES="${SOURCES:-sub1 sub2 sub3 sub5 sub9 sub17}"
NUM_ENVS="${NUM_ENVS:-1024}"
TIMEOUT="${TIMEOUT:-900}"
# The base test config restricts to dataObjects ['largetable','woodchair'] (a
# student-eval leftover) -- the curriculum trained on ALL objects, so by default
# we drop that restriction. Set ALL_OBJECTS=0 to keep the base config's filter.
ALL_OBJ=""
[ "${ALL_OBJECTS:-1}" = "1" ] && ALL_OBJ="--all-objects"
# Arch-matched configs. Default = MLP. For a TRANSFORMER checkpoint set
#   BASE_YAML=.../omomo_test_multibody_xf.yaml  TRAIN_YAML=<transformer train yaml>.
# BETAS_FILE overrides the base yaml's betas_file so the beta obs matches the
# checkpoint's training betas (neutral vs gendered vs neutral_aug).
BASE_YAML="${BASE_YAML:-isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml}"
TRAIN_YAML="${TRAIN_YAML:-isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml}"
# BETAS_FILE is required above. "none" is the explicit opt-out (run trained with
# no betas -> use the base yaml as-is, no override); anything else must be a real
# file (checked below) and overrides the base yaml's betas_file.
BETAS_ARG=""
if [ "$BETAS_FILE" != "none" ]; then
    [ -f "$BETAS_FILE" ] || { echo "[eval] ERROR: BETAS_FILE not found: $BETAS_FILE (use BETAS_FILE=none only for no-betas runs)" >&2; exit 2; }
    BETAS_ARG="--betas-file $BETAS_FILE"
fi

# Rename the job so `squeue` shows which eval this is (the output CSV stem).
scontrol update JobId="$SLURM_JOB_ID" JobName="ev-$(basename "${OUT%.csv}")" 2>/dev/null || true

mkdir -p "$(dirname "$OUT")"
echo "[eval] checkpoint = $CHECKPOINT"
echo "[eval] bodies=($BODIES)  x  sources=($SOURCES)  all_objects=${ALL_OBJECTS:-1}"
echo "[eval] -> $OUT"
echo "[eval] base=$BASE_YAML train=$TRAIN_YAML betas=${BETAS_FILE:-<base default>}"
[ -f "$CHECKPOINT" ] || { echo "[eval] ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

python -u scripts/eval_per_pair.py \
    --checkpoint "$CHECKPOINT" \
    --bodies $BODIES \
    --sources $SOURCES \
    --output-csv "$OUT" \
    --base-yaml "$BASE_YAML" \
    --train-yaml "$TRAIN_YAML" \
    --num-envs "$NUM_ENVS" \
    --timeout-per-pair "$TIMEOUT" $ALL_OBJ $BETAS_ARG

echo
echo "================ EVAL SUMMARY ================"
echo "full CSV: $OUT"
cat "$OUT"
echo "--- identity pairs only (body == source) ---"
awk -F, 'NR==1 || $1==$2' "$OUT"
echo "============================================="

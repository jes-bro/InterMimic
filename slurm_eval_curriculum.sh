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
# eval_per_pair.py + the old shared template (omomo_test_multibody.yaml) (Start mode, enableEvaluation,
# numObs 3230 -- matches the curriculum checkpoint arch [1024,1024,512]). This
# turns a checkpoint you ALREADY have into a result -- no more training needed.
#
# Default: the 6 folded-in (in-distribution) subjects, full body x source matrix.
#   diagonal (body==source) = identity = "drive each body with its own motion"
#   off-diagonal            = cross-retarget
# Override via env vars:
#   CHECKPOINT=path/to.pth   BODIES="sub2 sub3"   SOURCES="sub2"
#   TIMEOUT=900  OUT=eval_results/foo.csv
# (numEnvs is NOT a launcher knob -- it lives in the eval config; see below.)
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

# REQUIRED (no silent defaults): a stale hardcoded checkpoint/OUT would eval the
# WRONG run and mis-attribute the CSV. ENV_YAML is required for the same reason
# it used to have a default and must not: the default was the old shared template (omomo_test_multibody.yaml),
# a chunk-1 smoke-test template that silently supplied its own retargeting (none),
# betas, obs horizons, reset gating and PhysX buffer to whatever checkpoint it was
# handed. Pass the ARM'S OWN eval config.
require() {   # require VAR "hint"
    eval "_v=\${$1:-}"
    if [ -z "$_v" ]; then
        echo "[eval] ERROR: $1 is required (no default). $2" >&2
        exit 2
    fi
}
require CHECKPOINT "e.g. CHECKPOINT=checkpoints/<run>/nn/mimic_XXXX.pth"
require OUT        "e.g. OUT=eval_results/<run>__<ckpt>__heldout.csv"
require ENV_YAML   "the arm's own eval config, e.g. ENV_YAML=isaacgym/src/intermimic/data/cfg/omomo_eval_g3_bball__f0.yaml (resolve it with: scripts/check_eval_cfg.py --arm <arm>)"
require TRAIN_YAML "the arm's rl_games train config (it carries the network arch)"

# BETAS_FILE is gone. It existed to patch the template's betas_file to match the
# checkpoint -- but an arm's own eval config already states its betas, including
# stating them by ABSENCE for a no-betas arm. The old encoding could not express
# that: BETAS_FILE=none meant "send no override", which LEFT the template's
# gendered omomo_betas.npz in force and would have evaluated a no-betas policy
# with betas. ALL_OBJECTS is gone for the same reason: it existed only to undo
# the template's ['largetable','woodchair'] student-eval leftover.
[ -f "$ENV_YAML" ] || { echo "[eval] ERROR: ENV_YAML not found: $ENV_YAML" >&2; exit 2; }

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
# NUM_ENVS is deliberately GONE. The eval config owns numEnvs, full stop.
# It is a scoring-budget knob, not a resource knob: success is the BEST attempt
# per CLIP -- _max_execution_steps is a running max indexed by seq_id, over a
# clip-count denominator (intermimic.py:1685-1703) -- so more envs can only raise
# the success rate and lower the pose errors. A launcher default of 1024 silently
# beat whatever the config said, which is how two arms could be compared on
# different scoring budgets. To change it, change the eval config, where it is
# reviewable and shared by every arm compared against it.
TIMEOUT="${TIMEOUT:-900}"

# Rename the job so `squeue` shows which eval this is (the output CSV stem).
scontrol update JobId="$SLURM_JOB_ID" JobName="ev-$(basename "${OUT%.csv}")" 2>/dev/null || true

mkdir -p "$(dirname "$OUT")"
echo "[eval] checkpoint = $CHECKPOINT"
echo "[eval] bodies=($BODIES)  x  sources=($SOURCES)"
echo "[eval] -> $OUT"
# Echo the settings that decide what the numbers MEAN, straight out of the config,
# so the job log records them rather than the reader having to go find the file.
echo "[eval] env=$ENV_YAML train=$TRAIN_YAML"
python3 - "$ENV_YAML" <<'PY'
import sys, yaml
e = (yaml.safe_load(open(sys.argv[1])) or {}).get("env", {})
g = (e.get("rewardTerms") or {}).get("freeFlightGate") or {}
print(f"[eval] numEnvs={e.get('numEnvs')} rolloutLength={e.get('rolloutLength')} "
      f"stateInit={e.get('stateInit')} numObs={e.get('numObs')} "
      f"obsHorizons={e.get('obsHorizons', '<stock>')}")
print(f"[eval] betas={e.get('betas_file', '<none>')} "
      f"retarget={e.get('retargetedMotionDir', '<none>')} "
      f"motion={e.get('motion_file')} ffgResets={g.get('resets', False)}")
PY
[ -f "$CHECKPOINT" ] || { echo "[eval] ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

# RESUME=1 keeps the pairs an earlier run of this same CSV already completed and
# evaluates only the missing ones -- for a job that hit its walltime part way
# through. Only pairs with exit_code 0 and real metrics are kept, so failures are
# retried, and a CSV from a different checkpoint is refused rather than merged.
RESUME_ARG=""
[ "${RESUME:-0}" = 1 ] && RESUME_ARG="--resume" && echo "[eval] RESUME=1 (reusing completed pairs in $OUT)"

python -u scripts/eval_per_pair.py \
    --checkpoint "$CHECKPOINT" \
    --bodies $BODIES \
    --sources $SOURCES \
    --output-csv "$OUT" \
    --env-yaml "$ENV_YAML" \
    --train-yaml "$TRAIN_YAML" \
    --timeout-per-pair "$TIMEOUT" $RESUME_ARG

echo
echo "================ EVAL SUMMARY ================"
echo "full CSV: $OUT"
cat "$OUT"
echo "--- identity pairs only (body == source) ---"
awk -F, 'NR==1 || $1==$2' "$OUT"
echo "============================================="

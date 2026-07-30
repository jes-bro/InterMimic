#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="eval-multi"
#SBATCH --output=eval-multi-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Evaluate SEVERAL runs back-to-back in ONE job, for when slurm slots are scarce
# (eval_one.sh submits one job per run).
#
# Each run's plan -- checkpoint, arch-matched base yaml, betas file, sources,
# bodies -- is resolved by eval_one.sh in EMIT mode rather than re-derived here.
# That resolution is the part that fails SILENTLY when it is wrong: a mismatched
# betas file corrupts the 32 beta obs dims and the eval still runs, producing
# plausible numbers for the wrong thing. One implementation, reused.
#
# USAGE (from repo root):
#   RUNS="src2_xf_aug src2_xf_aug_normval src2_xf_aug_adlr src2_xf_aug_normval_adlr" \
#     sbatch slurm_eval_multi.sh
#
#   DRY=1 RUNS="..." bash slurm_eval_multi.sh    # resolve + print, run nothing
#
# Env:
#   RUNS       space-separated run ids (required). An entry may pin a specific
#              checkpoint as run@path, which is how you compare arms at MATCHED
#              training length -- the default picks each run's LATEST, and arms
#              that have trained for different numbers of epochs are not
#              comparable (you would be measuring training length, not the knob).
#   AT_EPOCH   pick each run's checkpoint NEAREST this epoch instead of its
#              latest. This is how you compare arms fairly: they train at
#              different speeds, so their latest checkpoints sit at different
#              epochs and comparing those measures training length. Prints the
#              actual epoch chosen per run, and how far off target it is.
#   NUM_ENVS   default 1024
#   TIMEOUT    seconds per (body,source) pair, default 900
#   N_SYNTHETIC / HELDOUT / BODIES / SOURCES  passed through to eval_one.sh
#
# A run that fails does NOT abort the rest -- with scarce slots, losing the whole
# batch to one bad checkpoint is the expensive outcome. Failures are collected and
# reported in the summary, and the job exits non-zero if any occurred.

set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

RUNS="${RUNS:?set RUNS='run1 run2 ...'}"
NUM_ENVS="${NUM_ENVS:-1024}"
TIMEOUT="${TIMEOUT:-900}"

if [ "${DRY:-0}" != 1 ]; then
    source ~/.bashrc
    conda deactivate
    conda activate intermimic-gym2
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

echo "[eval-multi] host=$(hostname) job=${SLURM_JOB_ID:-<none>}"
echo "[eval-multi] runs: $RUNS"
echo

# --- Resolve every run FIRST, before evaluating any of them. A typo or a missing
# checkpoint should surface in seconds, not after the first eval has burned an
# hour of the slot. ---
PLANS=""
BAD=""
for spec in $RUNS; do
    run="${spec%%@*}"
    ckpt=""
    [ "$spec" != "$run" ] && ckpt="${spec#*@}"
    if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
        echo "[eval-multi] $run: pinned checkpoint not found: $ckpt" >&2
        BAD="$BAD $run"; continue
    fi
    # AT_EPOCH: search the run's checkpoint dir for the snapshot nearest the
    # target. The dir comes from the pinned path when one is given, else from
    # whatever eval_one resolves as the latest -- so this works for borrowed
    # configs (run@other-run-checkpoint) too.
    if [ -n "${AT_EPOCH:-}" ]; then
        if [ -n "$ckpt" ]; then nndir=$(dirname "$ckpt")
        else
            probe=$(EMIT=1 sh scripts/eval_one.sh "$run" 2>/dev/null | sed -n "s/^CHECKPOINT='\(.*\)'$/\1/p")
            nndir=$(dirname "$probe")
        fi
        near=$(ls -1 "$nndir"/mimic_0*.pth 2>/dev/null | while read -r f; do
                   n=$(basename "$f" .pth | sed 's/mimic_0*//')
                   [ -n "$n" ] && echo "$(( n > AT_EPOCH ? n - AT_EPOCH : AT_EPOCH - n )) $n $f"
               done | sort -n | head -1)
        if [ -z "$near" ]; then
            echo "[eval-multi] $run: no mimic_0*.pth in $nndir to match epoch $AT_EPOCH" >&2
            BAD="$BAD $run"; continue
        fi
        off=$(echo "$near" | cut -d' ' -f1); ep=$(echo "$near" | cut -d' ' -f2)
        ckpt=$(echo "$near" | cut -d' ' -f3-)
        echo "[eval-multi] $run: epoch $ep (target $AT_EPOCH, off by $off)"
    fi
    plan=$(EMIT=1 sh scripts/eval_one.sh "$run" $ckpt 2>/dev/null) || { BAD="$BAD $run"; continue; }
    ck=$(printf '%s\n' "$plan" | sed -n "s/^CHECKPOINT='\(.*\)'$/\1/p")
    [ -f "$ck" ] || { echo "[eval-multi] $run: checkpoint missing: $ck" >&2; BAD="$BAD $run"; continue; }
    PLANS="$PLANS$run|$(printf '%s' "$plan" | tr '\n' ';')"$'\n'
done

if [ -n "$BAD" ]; then
    echo "[eval-multi] ERROR: could not resolve:$BAD" >&2
    echo "[eval-multi] (run 'DRY=1 sh scripts/eval_one.sh <run>' to see why)" >&2
    [ -z "$PLANS" ] && exit 2
    echo "[eval-multi] continuing with the runs that did resolve" >&2
fi

echo "[eval-multi] resolved plan:"
printf '%s' "$PLANS" | while IFS='|' read -r run plan; do
    [ -z "$run" ] && continue
    eval "$(printf '%s' "$plan" | tr ';' '\n')"
    echo "   $run -> $(basename "$CHECKPOINT")  betas=$(basename "$BETAS_FILE")  base=$(basename "$BASE_YAML")"
    echo "        bodies: $BODIES"
    echo "        -> $OUT"
done
echo

[ "${DRY:-0}" = 1 ] && { echo "[eval-multi] DRY=1, nothing run."; exit 0; }

FAILED=""
n=0
total=$(printf '%s' "$PLANS" | grep -c .)
while IFS='|' read -r run plan; do
    [ -z "$run" ] && continue
    n=$((n+1))
    eval "$(printf '%s' "$plan" | tr ';' '\n')"
    echo "=============================================================="
    echo "[eval-multi] $n/$total  $run  ($EXP)"
    echo "=============================================================="
    mkdir -p "$(dirname "$OUT")"
    BETAS_ARG=""
    [ "$BETAS_FILE" != "none" ] && BETAS_ARG="--betas-file $BETAS_FILE"
    # `|| true`: one bad run must not cost the remaining slot time.
    python -u scripts/eval_per_pair.py \
        --checkpoint "$CHECKPOINT" \
        --bodies $BODIES \
        --sources $SOURCES \
        --output-csv "$OUT" \
        --base-yaml "$BASE_YAML" \
        --train-yaml "$TRAIN_YAML" \
        --num-envs "$NUM_ENVS" \
        --timeout-per-pair "$TIMEOUT" --all-objects $BETAS_ARG \
        && echo "[eval-multi] $run OK -> $OUT" \
        || { echo "[eval-multi] $run FAILED (continuing)" >&2; FAILED="$FAILED $run"; }
    echo
done <<< "$PLANS"

echo "########################### SUMMARY ###########################"
while IFS='|' read -r run plan; do
    [ -z "$run" ] && continue
    eval "$(printf '%s' "$plan" | tr ';' '\n')"
    if [ -f "$OUT" ]; then
        python3 - "$OUT" "$run" <<'PY'
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]))
      if r.get('exit_code')=='0' and r.get('success_rate')]
if rows:
    sr=[float(r['success_rate']) for r in rows]
    hpe=[float(r['human_pose_error']) for r in rows if r['human_pose_error']]
    print(f"  {sys.argv[2]:<32} {len(rows):>4} pairs   success {st.mean(sr):5.1f}%   hpe {st.mean(hpe):.3f}")
else:
    print(f"  {sys.argv[2]:<32} no usable rows in {sys.argv[1]}")
PY
    else
        echo "  $run: NO CSV (eval failed)"
    fi
done <<< "$PLANS"
echo "###############################################################"
if [ -n "$FAILED" ]; then
    echo "[eval-multi] FAILED runs:$FAILED" >&2
    exit 1
fi

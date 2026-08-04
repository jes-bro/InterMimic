#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="eval-repeat"
#SBATCH --output=eval-repeat-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Evaluate the SAME checkpoint N times, in ONE job, to measure EVAL variance.
#
# Why this exists: retarget @16.2k scored 7.7% on sub16 while retarget_cpumotion
# @16.2k scored 63.5%, and a full diff of both configs shows they differ ONLY in
# cpuMotionData. A 56-point gap with no configured cause means either training
# run-to-run variance or eval-time variance, and those need different fixes. This
# script isolates the second: same checkpoint, same bodies, N repeats. If the
# numbers move, the eval is stochastic and NO single-run per-body number is
# trustworthy -- including every comparison drawn from the 2026-08-03/04 CSVs.
#
# One job rather than N submissions: at low queue priority, N queue waits is the
# expensive part, not the GPU time.
#
# USAGE (from repo root):
#   RUN=src2_xf_aug_retarget_cpumotion \
#   CKPT=checkpoints/smplx_teacher_src2_xf_aug_retarget_cpumotion/nn/mimic_00016200.pth \
#   BODIES=sub16 N=3 sbatch slurm_eval_repeat.sh
#
#   DRY=1 RUN=... CKPT=... BODIES=sub16 N=3 bash slurm_eval_repeat.sh   # resolve only
#
# Env:
#   RUN      run id as eval_one.sh takes it (required)
#   CKPT     checkpoint path; required -- a repeat test must pin the checkpoint,
#            since "latest" could differ between repeats if the run is still training
#   BODIES   space-separated bodies (default sub16 -- the unstable one)
#   N        repeats (default 3)
#   SOURCES  passed through to eval_one.sh (default: the run's own dataSub)
#   NUM_ENVS / TIMEOUT  as slurm_eval_curriculum.sh
#
# Output: eval_results/<exp>__<ckpt>__repeat<i>.csv, one per repeat.

set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

RUN="${RUN:?set RUN=<run id>, e.g. RUN=src2_xf_aug_retarget_cpumotion}"
CKPT="${CKPT:?set CKPT=<path to a specific mimic_000NNNNN.pth> -- a repeat test must pin the checkpoint}"
BODIES="${BODIES:-sub16}"
N="${N:-3}"
NUM_ENVS="${NUM_ENVS:-1024}"
TIMEOUT="${TIMEOUT:-900}"

[ -f "$CKPT" ] || { echo "[repeat] ERROR: checkpoint not found: $CKPT" >&2; exit 2; }

if [ "${DRY:-0}" != 1 ]; then
    source ~/.bashrc
    conda deactivate
    conda activate intermimic-gym2
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Resolve arch/betas/base-yaml through eval_one.sh in EMIT mode -- the same single
# implementation the other eval paths use. Re-deriving it here is exactly how a
# wrong betas file silently corrupts 32 obs dims and still produces numbers.
plan=$(EMIT=1 BODIES="$BODIES" ${SOURCES:+SOURCES="$SOURCES"} sh scripts/eval_one.sh "$RUN" "$CKPT" 2>/dev/null) || {
    echo "[repeat] ERROR: could not resolve '$RUN' (run: DRY=1 sh scripts/eval_one.sh $RUN)" >&2
    exit 2; }
eval "$plan"

echo "[repeat] run        : $RUN  ($EXP)"
echo "[repeat] checkpoint : $CHECKPOINT"
echo "[repeat] bodies     : $BODIES   sources: $SOURCES"
echo "[repeat] betas/base : $BETAS_FILE | $(basename "$BASE_YAML")"
echo "[repeat] repeats    : $N"

ckid=$(basename "$CHECKPOINT" .pth)
ckexp=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")

BETAS_ARG=""
if [ "$BETAS_FILE" != none ]; then
    [ -f "$BETAS_FILE" ] || { echo "[repeat] ERROR: BETAS_FILE not found: $BETAS_FILE" >&2; exit 2; }
    BETAS_ARG="--betas-file $BETAS_FILE"
fi

if [ "${DRY:-0}" = 1 ]; then
    for i in $(seq 1 "$N"); do
        echo "  would write eval_results/${ckexp}__${ckid}__repeat${i}.csv"
    done
    echo "[repeat] DRY=1, nothing run."
    exit 0
fi

mkdir -p eval_results
for i in $(seq 1 "$N"); do
    out="eval_results/${ckexp}__${ckid}__repeat${i}.csv"
    echo "=============================================================="
    echo "[repeat] $i/$N -> $out"
    echo "=============================================================="
    python -u scripts/eval_per_pair.py \
        --checkpoint "$CHECKPOINT" \
        --bodies $BODIES \
        --sources $SOURCES \
        --output-csv "$out" \
        --base-yaml "$BASE_YAML" \
        --train-yaml "$TRAIN_YAML" \
        --num-envs "$NUM_ENVS" \
        --timeout-per-pair "$TIMEOUT" --all-objects $BETAS_ARG
done

echo
echo "================ REPEAT SUMMARY ================"
echo "If success_rate differs across repeats, the EVAL is stochastic and no"
echo "single-run per-body number is trustworthy. If identical, the variance is"
echo "in TRAINING and the arm-vs-arm comparison is measuring something real."
for i in $(seq 1 "$N"); do
    out="eval_results/${ckexp}__${ckid}__repeat${i}.csv"
    [ -f "$out" ] && awk -F, -v r="$i" 'NR>1 {printf "  repeat %s  %-8s %8s%%  %s/%s\n", r, $1, $7, $8, $9}' "$out"
done
echo "================================================"

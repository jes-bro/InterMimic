#!/bin/sh
# run_vanilla_indist_local.sh -- the InterMimic BASELINE's in-distribution eval,
# run locally across several GPUs (ikura), instead of 13 sbatch jobs on simurgh.
#
# WHAT IT SCORES. The paper's released student (checkpoints/vanilla/student.pth)
# on the 13 OMOMO training bodies, each against all 13 sources -- the same exam
# the g3 students take, so the two sit in one table. The held-out counterpart
# (sub10/sub13/sub16) already ran on simurgh and gave 18.8%.
#
# WHY A SCRIPT. 13 bodies over N GPUs, each GPU working through its share
# SEQUENTIALLY (two evals on one card is ~34 GB and risks an OOM mid-sweep).
# That is a nested loop with a per-GPU background subshell -- too much to retype
# correctly, and it gets re-run every time a GPU frees up.
#
# ONE CSV PER BODY. Parallel jobs writing one CSV lose rows (that is how
# ho2_xf_sub10 was clobbered). Each body writes eval_results/<PREFIX>_<body>.csv
# and its own log; a body whose CSV already exists is SKIPPED, so re-running
# after a crash fills the gaps instead of redoing finished work.
#
# Usage (from this worktree's root, in the conda env):
#   GPUS="4 5 6 7" sh scripts/run_vanilla_indist_local.sh
#   GPUS="1 2" BODIES="sub1 sub2" sh scripts/run_vanilla_indist_local.sh
#   DRY=1 GPUS="4 5" sh scripts/run_vanilla_indist_local.sh    # print the plan only
#
# Env:
#   GPUS     GPU indices to use (default "4 5 6 7")
#   BODIES   bodies to score (default: the 13 OMOMO training bodies)
#   SOURCES  sources each body is scored against (default: the same 13)
#   PREFIX   CSV/log stem (default vanilla_indist)
#   TIMEOUT  per-pair timeout in seconds (default 7200, as the simurgh jobs used)
#   DRY=1    print what would run, run nothing
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

OMOMO13="sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17"
GPUS="${GPUS:-4 5 6 7}"
BODIES="${BODIES:-$OMOMO13}"
SOURCES="${SOURCES:-$OMOMO13}"
PREFIX="${PREFIX:-vanilla_indist}"
TIMEOUT="${TIMEOUT:-7200}"

CKPT=checkpoints/vanilla/student.pth
ENV_YAML=isaacgym/src/intermimic/data/cfg/vanilla_intermimic_eval.yaml
TRAIN_YAML=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all.yaml

# Fail loudly on a missing input rather than launching 13 jobs that each die
# deep inside a rollout.
for f in "$CKPT" "$ENV_YAML" "$TRAIN_YAML" slurm_eval_curriculum.sh; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done
mkdir -p eval_results

# Deal the bodies round-robin onto the GPUs: GPU i takes bodies i, i+N, i+2N...
# so the work is spread evenly even when the body count is not a multiple of N.
n_gpu=$(echo "$GPUS" | wc -w)
i=0
plan=$(for b in $BODIES; do
  g=$(echo "$GPUS" | cut -d' ' -f$(( i % n_gpu + 1 )))
  echo "$g $b"
  i=$((i + 1))
done)

echo "== vanilla in-dist: $(echo "$BODIES" | wc -w) bodies over $n_gpu GPU(s) [$GPUS]"
echo "   checkpoint : $CKPT"
echo "   sources    : $SOURCES"
echo "$plan" | while read -r g b; do
  out="eval_results/${PREFIX}_${b}.csv"
  if [ -f "$out" ]; then echo "   gpu $g  $b  SKIP (exists: $out)"; else echo "   gpu $g  $b  -> $out"; fi
done

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

for g in $GPUS; do
  # One subshell per GPU: its bodies run one after another, never concurrently.
  (
    for b in $(echo "$plan" | awk -v g="$g" '$1 == g { print $2 }'); do
      out="eval_results/${PREFIX}_${b}.csv"
      [ -f "$out" ] && continue
      CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      CHECKPOINT="$CKPT" OUT="$out" \
      ENV_YAML="$ENV_YAML" TRAIN_YAML="$TRAIN_YAML" \
      SOURCES="$SOURCES" BODIES="$b" RESUME=0 \
      EVAL_ENTRY=intermimic.run_distill EVAL_TASK=InterMimic_All \
      TIMEOUT="$TIMEOUT" \
      bash slurm_eval_curriculum.sh > "${PREFIX}-${b}.log" 2>&1 || \
        echo "[gpu $g] $b FAILED (see ${PREFIX}-${b}.log)" >&2
    done
  ) &
done
wait
echo "== done. Summarize with:"
echo "   head -1 eval_results/${PREFIX}_$(echo "$BODIES" | cut -d' ' -f1).csv > /tmp/${PREFIX}.csv"
echo "   tail -q -n +2 eval_results/${PREFIX}_*.csv >> /tmp/${PREFIX}.csv"
echo "   python3 scripts/summarize_evals.py /tmp/${PREFIX}.csv"

#!/bin/sh
# run_indist_local.sh -- a body x source matrix across several local GPUs, for any
# policy. Generalised from run_vanilla_indist_local.sh (the baseline's in-dist
# sweep) so the same launcher runs the g3 students on the GCP A100s.
#
# WHAT IT SCORES. Whatever CKPT / ENV_YAML / TRAIN_YAML name, on BODIES x SOURCES.
# Defaults are the InterMimic baseline's in-distribution sweep: the released
# student on the 13 OMOMO training bodies against all 13 sources. Point it at a
# student instead with:
#
#   CKPT=checkpoints/smplx_student_g3_omomo_xf_ret_nvadlr__f0/nn/mimic_00029000.pth \
#   ENV_YAML=isaacgym/src/intermimic/data/cfg/omomo_eval_student_g3_omomo_xf_ret_nvadlr__f0.yaml \
#   TRAIN_YAML=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_omomo_xf_ret_nvadlr__f0.yaml \
#   ENTRY=intermimic.run_distill TASK=InterMimicDistillG3 PREFIX=indist_xf29k \
#   GPUS="1 2 3" sh scripts/run_indist_local.sh
#
# WHY A SCRIPT. 13 bodies over N GPUs, each GPU working through its share
# SEQUENTIALLY (two evals on one card is ~34 GB and risks an OOM mid-sweep).
# That is a nested loop with a per-GPU background subshell -- too much to retype
# correctly, and it gets re-run every time a GPU frees up.
#
# ONE CSV PER BODY. Parallel jobs writing one CSV lose rows (that is how
# ho2_xf_sub10 was clobbered). Each body writes eval_results/<PREFIX>_<body>.csv
# and its own log; a body whose CSV is COMPLETE is SKIPPED, so re-running after a
# crash (or to add GPUs as they free up) fills the gaps instead of redoing
# finished work. "Complete" is defined by done_csv() below -- deliberately NOT
# "the file exists", since both a startup failure and a killed sweep leave files.
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
# SOURCES unset (or "") means: let eval_one resolve them from the ARM's own train
# config. That is what you want for any arm whose source set is not the OMOMO 13
# -- the activity student has 35 -- and it removes a 35-token line from every
# launch, which is a paste error waiting to happen.
SOURCES="${SOURCES-$OMOMO13}"
PREFIX="${PREFIX:-vanilla_indist}"
TIMEOUT="${TIMEOUT:-7200}"

CKPT="${CKPT:-checkpoints/vanilla/student.pth}"
ENV_YAML="${ENV_YAML:-isaacgym/src/intermimic/data/cfg/vanilla_intermimic_eval.yaml}"
TRAIN_YAML="${TRAIN_YAML:-isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all.yaml}"
# The paper's student is InterMimic_All via run_distill; the g3 students are
# InterMimicDistillG3 via the same entry. Both are overridable because this
# launcher is no longer baseline-specific.
ENTRY="${ENTRY:-intermimic.run_distill}"
TASK="${TASK:-InterMimic_All}"

# Fail loudly on a missing input rather than launching 13 jobs that each die
# deep inside a rollout.
for f in "$CKPT" "$ENV_YAML" "$TRAIN_YAML" slurm_eval_curriculum.sh; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done
mkdir -p eval_results

# done_csv(csv, n_sources) -- true only if this body is COMPLETE:
#   * one data row per source (a sweep killed mid-body leaves fewer), and
#   * at least one of them actually scored (success_rate set, exit_code 0).
#
# Both halves are needed. Requiring only a scored row would freeze a body killed
# at 1/13 pairs as "done" forever; requiring only the row count would accept the
# full CSV of exit_code=1 rows that a startup failure writes (a missing MJCF did
# exactly that on ikura). Pair-level failures inside an otherwise finished body
# are fine -- they are recorded as exit_code=1 rows and are visible to the
# summary, so the body is not silently re-run.
done_csv() {
  [ -f "$1" ] || return 1
  awk -F, -v want="$2" '
    NR > 1 && NF > 1 { rows++; if ($7 != "" && $10 == 0) scored = 1 }
    END { exit !(rows >= want && scored) }' "$1"
}

# Deal the bodies round-robin onto the GPUs: GPU i takes bodies i, i+N, i+2N...
# so the work is spread evenly even when the body count is not a multiple of N.
n_gpu=$(echo "$GPUS" | wc -w)
n_src=$(echo "$SOURCES" | wc -w)      # a complete body has one row per source
# With SOURCES unset the count is unknown here, so completeness falls back to
# "has at least one scored row" -- a re-run then redoes only bodies that produced
# nothing, which is the safe direction.
[ -z "$SOURCES" ] && n_src=1
i=0
plan=$(for b in $BODIES; do
  g=$(echo "$GPUS" | cut -d' ' -f$(( i % n_gpu + 1 )))
  echo "$g $b"
  i=$((i + 1))
done)

echo "== in-dist: $(echo "$BODIES" | wc -w) bodies over $n_gpu GPU(s) [$GPUS]"
echo "   checkpoint : $CKPT"
echo "   env cfg    : $ENV_YAML"
echo "   entry/task : $ENTRY / $TASK"
echo "   timeout    : ${TIMEOUT:-7200}s per pair"
echo "   sources    : ${SOURCES:-<from the arm's train cfg>}"
echo "$plan" | while read -r g b; do
  out="eval_results/${PREFIX}_${b}.csv"
  if done_csv "$out" "$n_src"; then echo "   gpu $g  $b  SKIP (complete: $out)"; else echo "   gpu $g  $b  -> $out"; fi
done

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

for g in $GPUS; do
  # One subshell per GPU: its bodies run one after another, never concurrently.
  (
    for b in $(echo "$plan" | awk -v g="$g" '$1 == g { print $2 }'); do
      out="eval_results/${PREFIX}_${b}.csv"
      done_csv "$out" "$n_src" && continue
      # an incomplete or all-failure CSV would otherwise make eval_per_pair refuse
      # to write; it is not a result, so it goes
      rm -f "$out"
      # env, not a bare assignment prefix: in dash (Ubuntu's /bin/sh) a word that
      # comes from an EXPANSION ends the assignment prefix, so
      # `${SOURCES:+SOURCES=...} BODIES=$b cmd` parses BODIES=... as the command
      # and dies with "BODIES=sub1: not found". env takes them as arguments, which
      # expands safely, and an unset SOURCES simply contributes nothing.
      env CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      CHECKPOINT="$CKPT" OUT="$out" \
      ENV_YAML="$ENV_YAML" TRAIN_YAML="$TRAIN_YAML" \
      ${SOURCES:+SOURCES="$SOURCES"} BODIES="$b" RESUME=0 \
      EVAL_ENTRY="$ENTRY" EVAL_TASK="$TASK" \
      TIMEOUT="${TIMEOUT:-7200}" \
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

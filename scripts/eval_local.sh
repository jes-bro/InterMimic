#!/bin/sh
# eval_local.sh -- run ONE eval right here, no slurm (a GCP VM, a workstation).
#
# Byte-for-byte the cluster job: the plan (eval cfg, checkpoint, bodies, sources,
# entry/task, CSV path) is resolved by eval_one.sh in EMIT mode -- the single
# implementation -- and then slurm_eval_curriculum.sh is run INLINE with bash
# instead of being handed to sbatch. Its #SBATCH lines are comments to bash, its
# `scontrol update` is already guarded, and with no SLURM_SUBMIT_DIR it cd's to
# the repo itself. Nothing about the eval differs from the cluster's.
#
# Usage (from the repo root, inside the conda env's shell):
#   sh scripts/eval_local.sh <run>[+variant] [checkpoint]
#   BODIES="sub10 sub13 sub16" SOURCES="sub401 ..." OUT=eval_results/x.csv sh scripts/eval_local.sh <run> <ckpt>
#   DRY=1 sh scripts/eval_local.sh <run> [checkpoint]      # print the plan, run nothing
#
# Same env knobs as eval_one.sh (BODIES, SOURCES, HELDOUT, N_SYNTHETIC, OUT,
# RESUME=1, OVERWRITE=1, TIMEOUT). Refuses to overwrite an existing CSV, like
# eval_one.sh does (EMIT mode returns before that check, so it is repeated here).
# Output is tee'd to eval-local-<csv stem>.log next to the CSV's stem, so a run
# in a detached tmux leaves the same record a slurm .out would.
#
# Several in a row: one tmux window,
#   for r in a b c; do sh scripts/eval_local.sh "$r" ...; done
# a failure in one does not stop the next (each is its own process; the loop
# continues), same as slurm_eval_multi.sh's policy.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

[ $# -ge 1 ] || { echo "usage: sh scripts/eval_local.sh <run>[+variant] [checkpoint]" >&2; exit 2; }

plan=$(EMIT=1 sh scripts/eval_one.sh "$@") || exit 2
eval "$plan"

if [ -f "$OUT" ] && [ "${OVERWRITE:-0}" != 1 ] && [ "${RESUME:-0}" != 1 ]; then
  echo "ERROR: $OUT already exists ($(awk 'END{print NR-1}' "$OUT") data rows)." >&2
  echo "       Refusing to overwrite. OUT=<other path>, RESUME=1 to fill in, or OVERWRITE=1." >&2
  exit 2
fi

log="eval-local-$(basename "${OUT%.csv}").log"
echo "== eval_local: $EXP =="
echo "   eval cfg   : $(basename "$ENV_YAML")"
echo "   checkpoint : $CHECKPOINT"
echo "   entry/task : $EVAL_ENTRY / $EVAL_TASK"
echo "   sources    : $SOURCES"
echo "   bodies     : $BODIES"
echo "   -> csv     : $OUT"
echo "   -> log     : $log"

if [ "${DRY:-0}" = 1 ]; then
  echo "   (DRY=1: not running)"
  exit 0
fi

# The exact variables slurm_eval_curriculum.sh requires, exported the way
# eval_one.sh's sbatch line passes them. bash, not sh: the script is bash.
CHECKPOINT="$CHECKPOINT" OUT="$OUT" \
ENV_YAML="$ENV_YAML" TRAIN_YAML="$TRAIN_YAML" \
SOURCES="$SOURCES" BODIES="$BODIES" RESUME="${RESUME:-0}" \
EVAL_ENTRY="$EVAL_ENTRY" EVAL_TASK="$EVAL_TASK" \
TIMEOUT="${TIMEOUT:-7200}" \
bash slurm_eval_curriculum.sh 2>&1 | tee "$log"

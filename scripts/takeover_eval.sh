#!/bin/sh
# takeover_eval.sh -- submit a TAKEOVER eval for one arm at one k.
#
# Lets a noised copy of the teacher wander for k steps from each episode's start,
# then hands the teacher the wheel and scores whether it can still finish. The
# slope of success vs k is the teacher-quality signal the ordinary eval cannot
# produce: `stateInit: Start` puts the teacher on its own reference trajectory
# every episode, so a normal eval only ever measures on-distribution competence,
# while distillation depends on the teacher labelling states the STUDENT picked
# (intermimic_agent_distill.py: beta decays to 0 by ~epoch 2200, after which the
# student drives and the teacher only labels).
#
# This does NOT reimplement eval resolution. It resolves bodies from the arm's
# own training config and then delegates to eval_one.sh, which owns arm -> eval
# config mapping, checkpoint discovery and submission. TAKEOVER_* reach the sim
# because sbatch defaults to --export=ALL and eval_per_pair.py merges os.environ
# into every per-pair subprocess.
#
# Usage (from repo root, on the cluster):
#   sh scripts/takeover_eval.sh <arm> <k> [checkpoint]
#     <arm>  short id, e.g. g3_omomo_geoall__f0
#     <k>    wander steps before takeover. k=0 submits the ACCEPTANCE run: no
#            takeover at all, which must reproduce the arm's ordinary eval.
#
# Env overrides:
#   NOISE=0.1     wander sigma in action units (default 0.1)
#   SEED=1234     fixed so two arms see the SAME perturbations (default 1234)
#   BODIES="..."  override the scored bodies (default: in-distribution only)
#   DRY=1         print the plan, don't submit
#
# IN-DISTRIBUTION ONLY BY DEFAULT. Under the current split the student holds out
# the same bodies the teachers do, so a teacher is never asked to label sub10/
# sub13/sub16 and its behaviour there says nothing about its fitness to teach.
# Synthetics are dropped for the same reason (N_SYNTHETIC=0).
#
# COMPARE ARMS ONLY AT MATCHED EPOCHS. The gen-2 ranking inverted between 27k and
# 72.1k, so a takeover number from two different epochs compares training budget,
# not teachers. Pass the checkpoint explicitly when the arms are still training.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

ARM="${1:?usage: sh scripts/takeover_eval.sh <arm> <k> [checkpoint]}"
K="${2:?missing k -- the number of wander steps before the teacher takes over}"
CKPT_ARG="${3:-}"

case "$K" in
  ''|*[!0-9]*) echo "ERROR: k must be a non-negative integer, got '$K'" >&2; exit 2 ;;
esac

NOISE="${NOISE:-0.1}"
SEED="${SEED:-1234}"

envc="isaacgym/src/intermimic/data/cfg/omomo_teacher_${ARM}.yaml"
[ -f "$envc" ] || { echo "ERROR: no env config for arm '$ARM': $envc" >&2; exit 2; }

# In-distribution bodies = the real bodies this arm actually TRAINED on. Held-out
# bodies are absent from subjectBodies by construction, and sub100+ are the
# synthetic augmentation bodies, dropped here.
BODIES_DEFAULT=$(python3 - "$envc" <<'PY'
import re, sys, yaml
bodies = (yaml.safe_load(open(sys.argv[1]))["env"].get("subjectBodies") or [])
num = lambda s: int(m.group(1)) if (m := re.match(r"sub(\d+)", str(s))) else -1
print(" ".join(b for b in bodies if 0 <= num(b) < 100))
PY
)
[ -n "$BODIES_DEFAULT" ] || { echo "ERROR: no in-distribution bodies in $envc" >&2; exit 2; }
BODIES="${BODIES:-$BODIES_DEFAULT}"

# Name the CSV for the arm, k, and sigma, so runs at different k never collide
# and a takeover result can never be mistaken for an ordinary eval.
tag="takeover_k${K}_n${NOISE}"
if [ "$K" = 0 ]; then
  tag="takeover_k0_ACCEPTANCE"
fi

echo "== takeover_eval: $ARM =="
echo "   k          : $K  $([ "$K" = 0 ] && echo '(ACCEPTANCE: no takeover; must reproduce the ordinary eval)')"
echo "   noise      : $NOISE   seed: $SEED"
echo "   bodies     : $BODIES"
echo "   tag        : $tag"

# k=0 deliberately leaves TAKEOVER_K UNSET rather than exporting 0. The point of
# the acceptance run is to exercise the same code path an ordinary eval takes --
# exporting 0 would test a zero-length-wander special case instead, which could
# agree by coincidence while the real no-op path was broken.
if [ "$K" != 0 ]; then
  export TAKEOVER_K="$K"
  export TAKEOVER_NOISE="$NOISE"
  export TAKEOVER_SEED="$SEED"
else
  unset TAKEOVER_K TAKEOVER_NOISE TAKEOVER_SEED 2>/dev/null || true
fi

# Resolve the checkpoint the same way eval_one.sh will, only so the CSV name can
# carry the epoch -- eval_one.sh remains the thing that actually picks it.
exp=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' \
      "isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_${ARM}.yaml" | awk '{print $2}')
if [ -n "$CKPT_ARG" ]; then
  CKPT="$CKPT_ARG"
else
  CKPT=$(ls -1 "checkpoints/$exp/nn"/mimic_0*.pth 2>/dev/null | sort | tail -1)
  [ -z "$CKPT" ] && CKPT="checkpoints/$exp/nn/mimic.pth"
fi
id=$(basename "$CKPT" .pth)
echo "   checkpoint : $CKPT"

OUT="eval_results/${exp}__${id}__${tag}.csv"
echo "   -> csv     : $OUT"

if [ "${DRY:-0}" = 1 ]; then
  echo "   (DRY=1: not submitting)"
  exit 0
fi

# N_SYNTHETIC=0: synthetics are training bodies, not a teaching-fitness signal.
N_SYNTHETIC=0 BODIES="$BODIES" OUT="$OUT" \
  sh scripts/eval_one.sh "$ARM" "$CKPT"

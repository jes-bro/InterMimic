#!/bin/sh
# eval_one.sh -- submit an eval for ONE finished teacher run.
#
# Resolves the arm's OWN eval config (scripts/check_eval_cfg.py --arm), which
# states the whole environment -- arch, obs horizons, betas, retargeting, reset
# gating, object physics, scoring budget. There is NO generic template and no
# fallback: an arm with no eval config is an error, because the template that used
# to fill that role silently substituted its own value for every feature it
# predated. Source and training bodies still come from the arm's train config,
# since they decide which pairs are worth scoring. Evaluates against:
#   - IN-DISTRIBUTION real training bodies (how well it does on what it saw)
#   - HELD-OUT test bodies sub10/sub16/sub13 (generalization to never-trained bodies)
#   - a sample of SYNTHETIC augmentation bodies sub100+ (if the run trained on any),
#     evenly spread across the range so you can examine performance on them.
#
# Usage (from repo root, on the cluster):
#   sh scripts/eval_one.sh <run> [checkpoint]
#     <run>        short id (src2_xf_aug) or full dir (smplx_teacher_src2_xf_aug)
#     [checkpoint] optional path; default = latest mimic_0*.pth in the run's nn/
# Env overrides:
#   N_SYNTHETIC=5      how many synthetic bodies to include (default 5; 0 = none)
#   SOURCES="sub2"     override the source set (default = the run's own dataSub)
#   HELDOUT="sub10 sub16 sub13"  override held-out bodies (sub4 excluded: broken)
#   DRY=1              print the resolved plan + sbatch vars, don't submit
#
# NOTE: synthetic-body eval needs those bodies' MJCFs present on the cluster and
# their betas in the run's betas file -- if either is missing the job fails loudly.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
CFG=isaacgym/src/intermimic/data/cfg

RUN="${1:?usage: sh scripts/eval_one.sh <run> [checkpoint]   e.g. src2_xf_aug}"
CKPT_ARG="${2:-}"

# Resolve run -> config paths. The config *name* (src9) and the checkpoint *dir*
# (smplx_teacher_src9_neutral) don't always match, so DON'T guess the dir from the
# run name -- read the authoritative full_experiment_name out of the train config,
# same as the teacher slurm scripts do.
texp="${RUN#smplx_teacher_}"
envc="$CFG/omomo_teacher_${texp}.yaml"
trainc="$CFG/train/rlg/omomo_teacher_${texp}.yaml"
[ -f "$trainc" ] || { echo "ERROR: train config not found: $trainc (needed for the network arch)" >&2; exit 2; }

# Some arms vary ONLY the train config -- normval / adlr / normval_adlr / lr4 all
# reuse omomo_teacher_src2_xf_aug.yaml as their env, so there is no env yaml with
# a matching name. Recover the env cfg the arm actually ran from its own slurm
# script, which is authoritative (it is literally what was submitted). Not a
# guess and not a fallback to some default: if neither the same-named env yaml
# nor a slurm script naming one exists, that is an error.
if [ ! -f "$envc" ]; then
  slurm="slurm_teacher_${texp}.sh"
  if [ -f "$slurm" ]; then
    from_slurm=$(grep -m1 '^CFG_ENV=' "$slurm" | cut -d= -f2-)
    if [ -n "$from_slurm" ] && [ -f "$from_slurm" ]; then
      echo "[eval_one] $texp has no env cfg of its own; using the one its slurm script ran: $from_slurm" >&2
      envc="$from_slurm"
    fi
  fi
fi
[ -f "$envc" ] || {
  echo "ERROR: env config not found: $envc" >&2
  echo "       and no CFG_ENV recoverable from slurm_teacher_${texp}.sh" >&2
  echo "       (train-only arms reuse another arm's env yaml -- pass it explicitly" >&2
  echo "        or add a slurm script that names it)" >&2
  exit 2; }

exp=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$trainc" | awk '{print $2}')
[ -n "$exp" ] || { echo "ERROR: no full_experiment_name in $trainc" >&2; exit 2; }
ckdir="checkpoints/$exp/nn"

# checkpoint: explicit arg, else latest numbered snapshot in the run's nn/
if [ -n "$CKPT_ARG" ]; then
  CKPT="$CKPT_ARG"
else
  CKPT=$(ls -1 "$ckdir"/mimic_0*.pth 2>/dev/null | sort | tail -1)
  [ -z "$CKPT" ] && CKPT="$ckdir/mimic.pth"
fi
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "       (run '$texp' -> experiment '$exp' -> dir $ckdir)" >&2
  if [ -d "$ckdir" ]; then
    echo "       files present in $ckdir:" >&2
    ls -1 "$ckdir" >&2 2>/dev/null | sed 's/^/         /' >&2
  else
    echo "       $ckdir does not exist. Teacher checkpoint dirs that DO exist:" >&2
    ls -1d checkpoints/smplx_teacher_*/ 2>/dev/null | sed 's/^/         /' >&2
  fi
  exit 2
fi

# Pull the SOURCE and BODY sets from the run's own env config -- what this arm
# trained on, which decides which pairs are worth scoring. Nothing about the
# observation layout is derived here any more: arch, obs width, betas and every
# other environment key now live in the arm's eval config, and check_eval_cfg.py
# is the single place that validates them (having two implementations of the obs
# derivation is how they drift).
N_SYNTHETIC="${N_SYNTHETIC:-5}"
eval "$(python3 - "$envc" "$N_SYNTHETIC" <<'PY'
import sys, re, yaml
envc, nsyn = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(envc))["env"]
src    = c.get("dataSub") or []
bodies = c.get("subjectBodies") or []
num = lambda s: (int(re.match(r"sub(\d+)", str(s)).group(1)) if re.match(r"sub(\d+)", str(s)) else -1)
real = [b for b in bodies if 0 <= num(b) < 100]
syn  = [b for b in bodies if num(b) >= 100]
pick = []
if syn and nsyn > 0:                      # evenly-spaced sample across the range
    n = min(nsyn, len(syn))
    idx = sorted({round(i * (len(syn) - 1) / (n - 1)) if n > 1 else 0 for i in range(n)})
    pick = [syn[i] for i in idx]
print(f'SRC_DEFAULT="{" ".join(src)}"')
print(f'REAL_BODIES="{" ".join(real)}"')
print(f'SYN_PICK="{" ".join(pick)}"')
print(f'N_SYN_AVAIL={len(syn)}')
PY
)"

# The arm's OWN eval config -- resolved, never guessed.
#
# This used to be a binary arch test: useTransformerObs set -> the 6524-dim
# template, else the 3230-dim one. Two things were wrong with that. It cannot
# express a multi-horizon arm at all (g3 is a SIX-horizon MLP wanting 9594/9626,
# and fell to the 3230 template), and even when the width happened to match, the
# template supplied its own value for every key it did not know about -- so every
# retargeting arm in gen-2 was scored against the UN-retargeted reference, and a
# gen-3 arm would have been scored with its free-flight gate off.
#
# check_eval_cfg.py maps arm -> config via each config's own `evalFor:` list AND
# re-proves that the config still mirrors the arm on every key outside the small
# eval-owned set. An arm with no eval config is a hard error here: there is no
# generic template left to fall back to, which is the point.
ENV_YAML=$(python3 scripts/check_eval_cfg.py --arm "$texp") || exit 2

SOURCES="${SOURCES:-$SRC_DEFAULT}"
# Held-out default is FOLD-AWARE (same __fN filename rule as summarize_evals.py):
# an __f1 run's test trio is sub5/sub7/sub12 -- the old fold0-only default would
# have scored fold1 TRAINING bodies as "held-out" and skipped the real test trio.
case "$texp" in
  *__f1*) HELDOUT_DEFAULT="sub5 sub7 sub12" ;;
  *)      HELDOUT_DEFAULT="sub10 sub16 sub13" ;;
esac
HELDOUT="${HELDOUT:-$HELDOUT_DEFAULT}"
# BODIES="sub11 sub16" evaluates ONLY those bodies (e.g. one suspect + one control).
# Honor it -- this used to be an unconditional assignment that silently clobbered
# the caller's override and evaluated all 21 bodies anyway.
BODIES_DEFAULT="$REAL_BODIES $HELDOUT $SYN_PICK"
BODIES="${BODIES:-$BODIES_DEFAULT}"

id=$(basename "$CKPT" .pth)
# Name the CSV after the experiment the CHECKPOINT actually belongs to, not the
# config's full_experiment_name. Those differ whenever a checkpoint is pinned from
# another run -- e.g. evaluating smplx_teacher_src2_xf_aug_scratch, whose own cfg
# was never committed, by borrowing src2_xf_aug's arch/betas. Naming it for the
# config would file the result under the wrong run.
ckexp=$(basename "$(dirname "$(dirname "$CKPT")")")
if [ "$ckexp" != "$exp" ] && [ -n "$ckexp" ]; then
  echo "[eval_one] checkpoint belongs to '$ckexp', not this config's '$exp' -- naming the CSV for '$ckexp'" >&2
  exp_out="$ckexp"
else
  exp_out="$exp"
fi
OUT="${OUT:-eval_results/${exp_out}__${id}__indist+heldout+syn.csv}"

# EMIT=1: print the resolved plan as shell KEY='VALUE' lines and exit, so a
# multi-run driver can `eval` it instead of re-implementing the resolution.
# Arch/betas resolution is the part that fails SILENTLY when wrong (a mismatched
# betas file corrupts the 32 beta obs dims and still runs), so it must have
# exactly one implementation -- this one.
if [ "${EMIT:-0}" = 1 ]; then
  printf "CHECKPOINT='%s'\nOUT='%s'\nENV_YAML='%s'\nTRAIN_YAML='%s'\nSOURCES='%s'\nBODIES='%s'\nEXP='%s'\n" \
    "$CKPT" "$OUT" "$ENV_YAML" "$trainc" "$SOURCES" "$BODIES" "$exp"
  exit 0
fi

# Refuse to overwrite an existing CSV. Each of these costs 1-4 GPU-hours, and a
# narrower re-run (fewer BODIES, or a TERM_REASON diagnostic) resolves to the SAME
# default OUT path -- which is how a 21-body result got replaced by a 2-body one.
# OUT=<path> for a variant, OVERWRITE=1 to genuinely replace.
# RESUME=1 is the deliberate exception: it does not discard the file, it keeps
# the pairs that succeeded and fills in the rest (eval_per_pair.py --resume).
if [ -f "$OUT" ] && [ "${OVERWRITE:-0}" != 1 ] && [ "${RESUME:-0}" != 1 ]; then
  echo "ERROR: $OUT already exists ($(awk 'END{print NR-1}' "$OUT") data rows)." >&2
  echo "       Refusing to overwrite -- an eval costs GPU-hours and a narrower" >&2
  echo "       re-run writes to this same path." >&2
  echo "       Use OUT=<other path> for a variant, or OVERWRITE=1 to replace it." >&2
  exit 2
fi

echo "== eval_one: $exp =="
echo "   eval cfg   : $(basename "$ENV_YAML")"
echo "   checkpoint : $CKPT"
echo "   sources    : $SOURCES"
if [ "$BODIES" = "$BODIES_DEFAULT" ]; then
  echo "   in-dist    : $REAL_BODIES"
  echo "   held-out   : $HELDOUT"
  echo "   synthetic  : ${SYN_PICK:-<none>}  (of $N_SYN_AVAIL available; N_SYNTHETIC=$N_SYNTHETIC)"
else
  echo "   BODIES     : $BODIES   <-- caller override (default set NOT used)"
fi
echo "   train cfg  : $(basename "$trainc")"
# Print the settings that decide what the numbers MEAN, out of the config that
# will actually run -- so the submission log records them and a stale eval config
# is visible at submit time rather than after a GPU-hour.
python3 - "$ENV_YAML" <<'EOPY'
import sys, yaml
e = (yaml.safe_load(open(sys.argv[1])) or {}).get("env", {})
g = (e.get("rewardTerms") or {}).get("freeFlightGate") or {}
print(f"   obs        : numObs={e.get('numObs')} horizons={e.get('obsHorizons', '<stock>')} "
      f"xf={bool(e.get('useTransformerObs'))} betas={e.get('betas_file', '<none>')}")
print(f"   scoring    : numEnvs={e.get('numEnvs')} rolloutLength={e.get('rolloutLength')} "
      f"stateInit={e.get('stateInit')} ffgResets={g.get('resets', False)}")
print(f"   reference  : motion={e.get('motion_file')} retarget={e.get('retargetedMotionDir', '<none>')}")
EOPY
echo "   -> csv     : $OUT"

if [ "${DRY:-0}" = 1 ]; then
  echo "   (DRY=1: not submitting)"
  exit 0
fi


# Nodes to keep off. Every gen-2 eval that landed on simurgh6 died during setup
# with "RuntimeError: CUDA error: uncorrectable ECC error encountered" -- failing
# GPU memory. Those jobs exit COMPLETED in ~45 s having written a FULL CSV whose
# every row is exit_code=1 with empty metrics, which reads as a result until you
# check the metrics column.
#
# simurgh4 is NOT excluded: it was suspect alongside simurgh6, but it ran a
# 16-body eval to completion the same day the ECC failures happened. Add it back
# with EXCLUDE_NODES=simurgh4,simurgh6 if it starts eating jobs again.
#
# This MUST be an sbatch CLI flag. #SBATCH headers are read before the job script
# runs, and sbatch does not honour an SBATCH_EXCLUDE environment variable -- a
# submission that set one still landed on simurgh6. Note it only binds at
# submission: on an already-running job the placement is fixed, and on a PENDING
# one it can still be changed with `scontrol update JobId=N ExcNodeList=...`.
#
# EXCLUDE_NODES="" opts out; EXCLUDE_NODES=nodeA,nodeB overrides the list.
EXCLUDE_NODES="${EXCLUDE_NODES-simurgh6}"
[ -n "$EXCLUDE_NODES" ] && echo "   exclude    : $EXCLUDE_NODES"

CHECKPOINT="$CKPT" OUT="$OUT" \
ENV_YAML="$ENV_YAML" TRAIN_YAML="$trainc" \
SOURCES="$SOURCES" BODIES="$BODIES" RESUME="${RESUME:-0}" \
sbatch ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} slurm_eval_curriculum.sh

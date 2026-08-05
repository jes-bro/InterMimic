#!/bin/sh
# eval_one.sh -- submit an eval for ONE finished teacher run.
#
# Auto-detects arch (MLP/transformer), betas, source, and training bodies from
# the run's OWN env config -- no fallbacks (a wrong arch/betas loads but produces
# silently-wrong numbers). Evaluates the run's own source against:
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

# Pull arch/betas/source/bodies from the run's OWN env config. python parses the
# yaml (subjectBodies is a long list -- grep/sed would be fragile) and emits
# shell assignments; on an unknown numObs it emits a hard error, no arch guess.
N_SYNTHETIC="${N_SYNTHETIC:-5}"
eval "$(python3 - "$envc" "$N_SYNTHETIC" <<'PY'
import sys, re, yaml
envc, nsyn = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(envc))["env"]
nobs = int(c["numObs"])
arch = {3230: "mlp", 6524: "transformer"}.get(nobs)
if arch is None:
    print(f'echo "ERROR: unexpected numObs={nobs} in {envc} (want 3230 MLP / 6524 transformer)" >&2; exit 2')
    sys.exit()
betas  = c.get("betas_file")
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
print(f'ARCH={arch}')
print(f'NOBS={nobs}')
print(f'BETAS="{betas or "none"}"')
print(f'SRC_DEFAULT="{" ".join(src)}"')
print(f'REAL_BODIES="{" ".join(real)}"')
print(f'SYN_PICK="{" ".join(pick)}"')
print(f'N_SYN_AVAIL={len(syn)}')
PY
)"

# arch -> arch-matched test base yaml (python already hard-errored on unknown numObs)
if [ "$ARCH" = transformer ]; then BASE="$CFG/omomo_test_multibody_xf.yaml"
else                               BASE="$CFG/omomo_test_multibody.yaml"; fi

SOURCES="${SOURCES:-$SRC_DEFAULT}"
HELDOUT="${HELDOUT:-sub10 sub16 sub13}"
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
  printf "CHECKPOINT='%s'\nOUT='%s'\nBETAS_FILE='%s'\nBASE_YAML='%s'\nTRAIN_YAML='%s'\nSOURCES='%s'\nBODIES='%s'\nEXP='%s'\n" \
    "$CKPT" "$OUT" "$BETAS" "$BASE" "$trainc" "$SOURCES" "$BODIES" "$exp"
  exit 0
fi

# Refuse to overwrite an existing CSV. Each of these costs 1-4 GPU-hours, and a
# narrower re-run (fewer BODIES, or a TERM_REASON diagnostic) resolves to the SAME
# default OUT path -- which is how a 21-body result got replaced by a 2-body one.
# OUT=<path> for a variant, OVERWRITE=1 to genuinely replace.
if [ -f "$OUT" ] && [ "${OVERWRITE:-0}" != 1 ]; then
  echo "ERROR: $OUT already exists ($(awk 'END{print NR-1}' "$OUT") data rows)." >&2
  echo "       Refusing to overwrite -- an eval costs GPU-hours and a narrower" >&2
  echo "       re-run writes to this same path." >&2
  echo "       Use OUT=<other path> for a variant, or OVERWRITE=1 to replace it." >&2
  exit 2
fi

echo "== eval_one: $exp =="
echo "   arch/betas : $ARCH (numObs=$NOBS) / $BETAS"
echo "   checkpoint : $CKPT"
echo "   sources    : $SOURCES"
if [ "$BODIES" = "$BODIES_DEFAULT" ]; then
  echo "   in-dist    : $REAL_BODIES"
  echo "   held-out   : $HELDOUT"
  echo "   synthetic  : ${SYN_PICK:-<none>}  (of $N_SYN_AVAIL available; N_SYNTHETIC=$N_SYNTHETIC)"
else
  echo "   BODIES     : $BODIES   <-- caller override (default set NOT used)"
fi
echo "   base/train : $(basename "$BASE") | $(basename "$trainc")"
echo "   -> csv     : $OUT"

if [ "${DRY:-0}" = 1 ]; then
  echo "   (DRY=1: not submitting)"
  exit 0
fi


CHECKPOINT="$CKPT" OUT="$OUT" BETAS_FILE="$BETAS" \
BASE_YAML="$BASE" TRAIN_YAML="$trainc" \
SOURCES="$SOURCES" BODIES="$BODIES" ALL_OBJECTS=1 \
sbatch slurm_eval_curriculum.sh

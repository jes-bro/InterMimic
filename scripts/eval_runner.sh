#!/bin/sh
# eval_runner.sh -- auto-eval each run's LATEST checkpoint on the held-out test set.
#
# For every teacher / curriculum run it picks the newest checkpoint, detects arch
# (MLP vs transformer) + betas from that run's config, and sbatch'es an
# arch/betas-matched eval via slurm_eval_curriculum.sh covering BOTH slices:
#     training-source x test-target   AND   test-source x test-target
# Test people = {10,16} (sub4 skipped -- broken asset). Teachers ALSO test sub13
# (their extra held-out body). Curriculum sources = its folded-in subjects; teacher
# source = its one source. Test sources = {10,16} for both.
#
# Marker files in eval_results/.watched/ prevent re-evaluating the same checkpoint,
# so this is safe to run repeatedly / on a schedule. Login-node, no GPU.
# Dry-run by default; CONFIRM=1 to sbatch.  Automatic eval via cron, e.g. daily:
#   0 6 * * *  cd <repo> && CONFIRM=1 sh scripts/eval_runner.sh >> eval_runner.log 2>&1
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
CFG=isaacgym/src/intermimic/data/cfg
XF_BASE="$CFG/omomo_test_multibody_xf.yaml"
MLP_BASE="$CFG/omomo_test_multibody.yaml"
XF_TRAIN="$CFG/train/rlg/omomo_teacher_src9_xf.yaml"   # generic transformer net (eval only)
MLP_TRAIN="$CFG/train/rlg/omomo_multibody.yaml"
TEST_TGT="sub10 sub16"
MARK=eval_results/.watched; mkdir -p "$MARK"

subs_of()   { grep -oE 'dataSub:.*' "$1" 2>/dev/null | grep -oE 'sub[0-9]+' | tr '\n' ' '; }
numobs_of() { grep -oE 'numObs:[[:space:]]*[0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | head -1; }
betas_of()  { grep -oE 'betas_file:[[:space:]]*[^[:space:]]+' "$1" 2>/dev/null | awk '{print $2}' | head -1; }

seen=" "
for d in $(ls -dt checkpoints/smplx_teacher_*/ checkpoints/smplx_curriculum_*/ 2>/dev/null); do
    exp=$(basename "$d")
    case "$exp" in
      smplx_teacher_*)
        runkey="$exp"; texp=${exp#smplx_teacher_}
        envc="$CFG/omomo_teacher_${texp}.yaml"
        trainc="$CFG/train/rlg/omomo_teacher_${texp}.yaml"
        kind=teacher ;;
      smplx_curriculum_*)
        # dir = smplx_curriculum_<run>_s<suffix>_sub<N>_<role>; anchor on the _sub<N>_<role>
        # tail so a run/suffix that itself starts with 's' (e.g. sub14) isn't mis-split.
        run=$(echo "$exp" | sed -E 's/^smplx_curriculum_(.+)_s[^_]+_sub[0-9]+_[a-z]+$/\1/')
        suf=$(echo "$exp" | sed -E 's/^smplx_curriculum_.+_s([^_]+)_sub[0-9]+_[a-z]+$/\1/')
        runkey="$run"
        envc="curriculum_work/$run/cfgs/env_s${suf}.yaml"
        trainc="curriculum_work/$run/cfgs/train_s${suf}.yaml"
        kind=curriculum ;;
      *) continue ;;
    esac
    case "$seen" in *" $runkey "*) continue ;; esac   # newest checkpoint dir per run only
    seen="$seen$runkey "

    ckpt=$(ls -1 "$d"nn/mimic_0*.pth 2>/dev/null | sort | tail -1)
    [ -z "$ckpt" ] && ckpt="${d}nn/mimic.pth"
    [ -f "$ckpt" ] || { echo "skip $runkey: no checkpoint in ${d}nn/"; continue; }
    id=$(basename "$ckpt" .pth)
    marker="$MARK/${exp}__${id}.done"
    [ -f "$marker" ] && continue

    [ -f "$envc" ] || { echo "skip $runkey: config not found ($envc)"; continue; }
    nobs=$(numobs_of "$envc"); betas=$(betas_of "$envc"); tsrc=$(subs_of "$envc")
    if [ "$nobs" = "6524" ]; then BASE="$XF_BASE"; TRAIN="$XF_TRAIN"; archname=transformer
    else BASE="$MLP_BASE"; TRAIN="$MLP_TRAIN"; archname=MLP; fi
    [ -f "$trainc" ] && TRAIN="$trainc"    # prefer the run's own train yaml

    if [ "$kind" = teacher ]; then BODIES="$TEST_TGT sub13"; else BODIES="$TEST_TGT"; fi
    SOURCES="$tsrc$TEST_TGT"
    OUT="eval_results/${exp}__${id}__heldout.csv"

    printf '\n== %s ==\n' "$runkey"
    printf '   kind/arch  : %s / %s (numObs=%s)\n' "$kind" "$archname" "${nobs:-?}"
    printf '   checkpoint : %s\n' "$ckpt"
    printf '   bodies     : %s\n' "$BODIES"
    printf '   sources    : %s\n' "$SOURCES"
    printf '   betas      : %s\n' "${betas:-<base default>}"
    printf '   base/train : %s | %s\n' "$(basename "$BASE")" "$(basename "$TRAIN")"
    printf '   -> csv     : %s\n' "$OUT"
    if [ "${CONFIRM:-0}" = "1" ]; then
        export CHECKPOINT="$ckpt" BODIES="$BODIES" SOURCES="$SOURCES" OUT="$OUT"
        export BASE_YAML="$BASE" TRAIN_YAML="$TRAIN" BETAS_FILE="$betas" ALL_OBJECTS=1
        export NUM_ENVS="${NUM_ENVS:-1024}" TIMEOUT="${TIMEOUT:-900}"
        jid=$(sbatch --parsable --export=ALL slurm_eval_curriculum.sh)
        touch "$marker"
        printf '   action     : SUBMITTED -> job %s\n' "$jid"
    else
        printf '   action     : WOULD SUBMIT (CONFIRM=1 to run)\n'
    fi
done

#!/bin/sh
# Resubmit source-teacher runs (slurm_teacher_src*.sh) that are NOT currently
# running. Each teacher slurm auto-resumes from checkpoints/<exp>/nn/mimic.pth,
# so resubmitting CONTINUES the run (never restarts from scratch).
#
# PLAIN login-node script (no #SBATCH / no GPU): it only inspects state and
# sbatch'es the individual teacher GPU jobs. For each teacher it prints the
# run/experiment name, why it's being resumed, the checkpoint it continues from,
# the config, and the exact invocation -- so you can see what will run and why.
# POSIX sh (works with `sh` or `bash`).
#
# DRY RUN by default. Set CONFIRM=1 to actually sbatch.  Tip: tee to a log:
#   sh scripts/resubmit_teachers.sh | tee resubmit_teachers_$(date +%F_%H%M).log
#   CONFIRM=1 sh scripts/resubmit_teachers.sh
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

field() { grep -oE "^$1=.*" "$2" 2>/dev/null | head -1 | cut -d= -f2-; }

n_sub=0; n_run=0; n_fresh=0
for f in slurm_teacher_src*.sh; do
    [ -f "$f" ] || { echo "MISSING  $f"; continue; }
    jn=$(grep -oE 'job-name="[^"]+"' "$f" | head -1 | sed 's/.*job-name="//; s/"//')
    cfg_env=$(field CFG_ENV "$f")
    cfg_train=$(field CFG_TRAIN "$f")
    exp=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$cfg_train" 2>/dev/null | awk '{print $2}' | tr -d "'\"")
    [ -z "$exp" ] && exp="(unknown)"

    printf '\n== %s ==\n' "${jn:-$f}"
    printf '   run/experiment : %s\n' "$exp"

    # is a job with this name already queued/running?
    rinfo=$(squeue -u "$USER" -h -n "$jn" -o "%i|%T|%M" 2>/dev/null | head -1)
    if [ -n "$rinfo" ]; then
        rjid=$(echo "$rinfo" | cut -d'|' -f1)
        rst=$(echo "$rinfo" | cut -d'|' -f2)
        rtm=$(echo "$rinfo" | cut -d'|' -f3)
        printf '   status         : %s (job %s, elapsed %s) -> skip\n' "$rst" "$rjid" "$rtm"
        n_run=$((n_run + 1)); continue
    fi

    ckpt="checkpoints/${exp}/nn/mimic.pth"
    if [ -f "$ckpt" ]; then
        mt=$(stat -c '%y' "$ckpt" 2>/dev/null | cut -d. -f1)
        latest=$(ls -1 "checkpoints/${exp}/nn/"mimic_0*.pth 2>/dev/null | sort | tail -1)
        printf '   status         : NOT running -> RESUME\n'
        printf '   checkpoint     : %s (last modified %s)\n' "$ckpt" "${mt:-?}"
        [ -n "$latest" ] && printf '   latest step    : %s\n' "$(basename "$latest")"
        printf '   why            : auto-resume continues from that checkpoint\n'
    else
        printf '   status         : NOT running -> FRESH START (no checkpoint at %s)\n' "$ckpt"
        n_fresh=$((n_fresh + 1))
    fi
    printf '   env cfg        : %s\n' "${cfg_env:-?}"
    printf '   train cfg      : %s\n' "${cfg_train:-?}"
    printf '   invocation     : python -u -m intermimic.run --task InterMimic --cfg_env %s --cfg_train %s --headless --output checkpoints\n' "${cfg_env:-?}" "${cfg_train:-?}"

    if [ "${CONFIRM:-0}" = "1" ]; then
        jid=$(sbatch --parsable "$f")
        printf '   action         : SUBMITTED  -> job %s\n' "$jid"
    else
        printf '   action         : WOULD SUBMIT  (CONFIRM=1 to run: sbatch %s)\n' "$f"
    fi
    n_sub=$((n_sub + 1))
done

printf '\n-----\n'
if [ "${CONFIRM:-0}" = "1" ]; then
    printf 'submitted %d (%d fresh-start), already-running %d\n' "$n_sub" "$n_fresh" "$n_run"
else
    printf '%d would be submitted (%d fresh-start), %d already running. Re-run with CONFIRM=1.\n' "$n_sub" "$n_fresh" "$n_run"
fi

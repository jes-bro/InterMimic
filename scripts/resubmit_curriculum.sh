#!/bin/sh
# Resubmit curriculum runs that are NOT currently running, reproducing each
# run's ORIGINAL config from its recorded '[curriculum] invocation:' log line
# (nothing reconstructed by hand). Each continues via curriculum_work/<run>/
# state.json + --resume.
#
# PLAIN login-node script (no #SBATCH / no GPU): it inspects state and sbatch'es
# the curriculum GPU job (slurm_curriculum_resume.sh). Scans curriculum-*.out
# newest-first, so it uses each run's LATEST invocation; dedupes by --run-name;
# skips runs already in squeue; and prints run name, why, source log, and the
# exact invocation for each. POSIX sh.
#
# DRY RUN by default. Set CONFIRM=1 to sbatch.  Skip runs with EXCLUDE="a b".
# NEUTRAL_ONLY=1 resubmits only neutral-beta runs (skips gendered-beta ones).
# DEDUP_CONFIG=1 skips runs whose config (args minus --run-name) matches an
#   already-selected run -- i.e. redundant duplicate experiments.
#   sh scripts/resubmit_curriculum.sh | tee resubmit_curriculum_$(date +%F_%H%M).log
#   NEUTRAL_ONLY=1 DEDUP_CONFIG=1 sh scripts/resubmit_curriculum.sh       # preview
#   CONFIRM=1 NEUTRAL_ONLY=1 DEDUP_CONFIG=1 sh scripts/resubmit_curriculum.sh
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

seen=" "        # run-names already handled
seen_sigs=""    # config signatures already seen (for DEDUP_CONFIG)
found=0
for log in $(ls -t curriculum-*.out 2>/dev/null); do
    line=$(grep -m1 'invocation: scripts/curriculum_runner.py' "$log") || continue
    args=${line#*invocation: scripts/curriculum_runner.py }
    run=$(printf '%s\n' "$args" | grep -oE -- '--run-name [^ ]+' | awk '{print $2}')
    [ -z "$run" ] && continue
    case "$seen" in *" $run "*) continue ;; esac   # already handled (latest wins)
    seen="$seen$run "

    printf '\n== c-%s ==\n' "$run"
    printf '   run        : %s\n' "$run"
    printf '   from log   : %s\n' "$log"

    case " ${EXCLUDE:-} " in *" $run "*)
        printf '   status     : EXCLUDED -> skip\n'; continue ;;
    esac

    # NEUTRAL_ONLY=1: only resubmit neutral-beta runs (gendered betas break
    # cross-gender conditioning). Gendered = --betas-file scripts/omomo_betas.npz;
    # neutral = omomo_betas_neutral[_aug].npz.
    if [ "${NEUTRAL_ONLY:-0}" = "1" ]; then
        bf=$(printf '%s\n' "$args" | grep -oE -- '--betas-file [^ ]+' | awk '{print $2}')
        case "$bf" in
            *neutral*) : ;;
            *) printf '   status     : SKIP gendered betas (%s) [NEUTRAL_ONLY=1]\n' "${bf:-default(gendered)}"; continue ;;
        esac
    fi

    # DEDUP_CONFIG=1: two different run-names with the SAME config (args minus the
    # run-name / --resume) are redundant experiments -> submit one, skip the rest.
    if [ "${DEDUP_CONFIG:-0}" = "1" ]; then
        sig=$(printf '%s' "$args" | sed -E 's/--run-name +[^ ]*//; s/--resume//; s/ +/ /g' | md5sum | cut -d' ' -f1)
        case "$seen_sigs" in
            *" $sig="*)
                owner=$(printf '%s' "$seen_sigs" | grep -oE " $sig=[^ ]+" | head -1 | cut -d= -f2)
                printf '   status     : DUPLICATE config of c-%s -> skip [DEDUP_CONFIG=1]\n' "$owner"
                continue ;;
        esac
        seen_sigs="$seen_sigs $sig=$run"
    fi

    rinfo=$(squeue -u "$USER" -h -n "c-$run" -o "%i|%T|%M" 2>/dev/null | head -1)
    if [ -n "$rinfo" ]; then
        printf '   status     : %s (job %s, elapsed %s) -> skip\n' \
            "$(echo "$rinfo" | cut -d'|' -f2)" "$(echo "$rinfo" | cut -d'|' -f1)" "$(echo "$rinfo" | cut -d'|' -f3)"
        continue
    fi
    found=$((found + 1))
    case "$args" in *--resume*) ;; *) args="$args --resume" ;; esac
    if [ -f "curriculum_work/$run/state.json" ]; then
        printf '   status     : NOT running -> RESUME (curriculum_work/%s/state.json present)\n' "$run"
    else
        printf '   status     : NOT running -> RESUME but NO state.json (curriculum_work/%s) -> STARTS FRESH at stage 1\n' "$run"
    fi
    printf '   invocation : scripts/curriculum_runner.py %s\n' "$args"

    if [ "${CONFIRM:-0}" = "1" ]; then
        CURRICULUM_ARGS="$args"; export CURRICULUM_ARGS
        jid=$(sbatch --parsable --export=ALL slurm_curriculum_resume.sh)
        printf '   action     : SUBMITTED -> job %s\n' "$jid"
    else
        printf '   action     : WOULD SUBMIT  (CONFIRM=1 to run)\n'
    fi
done

printf '\n-----\n'
if [ "$found" -eq 0 ] && [ "$seen" = " " ]; then
    printf "No '[curriculum] invocation:' lines found in curriculum-*.out.\n"
    printf "Are you in the repo root, and are the logs there? (ls curriculum-*.out)\n"
else
    printf '%d run(s) to resubmit%s.\n' "$found" "$([ "${CONFIRM:-0}" = 1 ] && echo ' (submitted)' || echo ' (dry-run; CONFIRM=1 to submit)')"
fi

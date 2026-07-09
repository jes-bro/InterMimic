#!/bin/bash
# Resubmit curriculum runs that are NOT currently running, reproducing each
# run's ORIGINAL config from its recorded '[curriculum] invocation:' log line
# (so nothing is reconstructed by hand). Each continues via
# curriculum_work/<run>/state.json + --resume.
#
# It scans curriculum-*.out (newest first) so it picks up each run's LATEST
# invocation, dedupes by --run-name, skips runs already in squeue, and submits
# the rest through slurm_curriculum_resume.sh.
#
# DRY RUN by default. Set CONFIRM=1 to actually sbatch.
#   sh scripts/resubmit_curriculum.sh            # preview (shows each run's args)
#   CONFIRM=1 sh scripts/resubmit_curriculum.sh  # submit the missing ones
#
# Run from the repo root (it cd's there itself). Needs the curriculum-*.out logs
# present in the repo root (that's where slurm writes them by default).
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

running=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
declare -A seen
found=0

# newest logs first -> first invocation seen per run-name is the latest one
for log in $(ls -t curriculum-*.out 2>/dev/null); do
    line=$(grep -m1 'invocation: scripts/curriculum_runner.py' "$log") || continue
    args=${line#*invocation: scripts/curriculum_runner.py }
    run=$(printf '%s\n' "$args" | grep -oE -- '--run-name [^ ]+' | awk '{print $2}')
    [ -z "$run" ] && continue
    [ -n "${seen[$run]:-}" ] && continue
    seen[$run]=1
    # EXCLUDE="run1 run2" to skip specific runs (e.g. redundant/cancelled ones)
    case " ${EXCLUDE:-} " in *" $run "*) echo "EXCLUDED      c-$run  -- skip"; continue ;; esac
    found=$((found + 1))

    if printf '%s\n' "$running" | grep -qx "c-$run"; then
        echo "RUNNING       c-$run  -- skip"
        continue
    fi
    case "$args" in *--resume*) ;; *) args="$args --resume" ;; esac   # ensure resume

    if [ "${CONFIRM:-0}" = "1" ]; then
        export CURRICULUM_ARGS="$args"
        jid=$(sbatch --parsable --export=ALL slurm_curriculum_resume.sh)
        echo "SUBMITTED     c-$run  -> job $jid   (from $log)"
    else
        echo "WOULD SUBMIT  c-$run   (from $log)"
        echo "      args: $args"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No '[curriculum] invocation:' lines found in curriculum-*.out."
    echo "Are you in the repo root, and are the logs there? (ls curriculum-*.out)"
fi

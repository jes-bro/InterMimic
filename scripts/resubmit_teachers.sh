#!/bin/bash
# Resubmit source-teacher runs (slurm_teacher_src*.sh) that are NOT currently
# running. Each teacher slurm auto-resumes from checkpoints/<exp>/nn/mimic.pth,
# so resubmitting safely CONTINUES the run (never restarts from scratch).
#
# DRY RUN by default -- prints what it would do. Set CONFIRM=1 to actually sbatch.
#   sh scripts/resubmit_teachers.sh            # preview
#   CONFIRM=1 sh scripts/resubmit_teachers.sh  # submit the missing ones
#
# Run from the repo root (or anywhere; it cd's to the repo root itself).
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

running=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)

n_sub=0; n_run=0
for f in slurm_teacher_src*.sh; do
    [ -f "$f" ] || { echo "MISSING  $f"; continue; }
    jn=$(grep -oE 'job-name="[^"]+"' "$f" | head -1 | sed 's/.*job-name="//; s/"//')
    [ -z "$jn" ] && { echo "NO JOB-NAME in $f -- skip"; continue; }
    if printf '%s\n' "$running" | grep -qx "$jn"; then
        echo "RUNNING       $jn   ($f)  -- skip"
        n_run=$((n_run + 1))
    elif [ "${CONFIRM:-0}" = "1" ]; then
        jid=$(sbatch --parsable "$f")
        echo "SUBMITTED     $jn   ($f)  -> job $jid"
        n_sub=$((n_sub + 1))
    else
        echo "WOULD SUBMIT  $jn   ($f)"
        n_sub=$((n_sub + 1))
    fi
done
echo "---"
if [ "${CONFIRM:-0}" = "1" ]; then
    echo "submitted $n_sub, already-running $n_run"
else
    echo "$n_sub would be submitted, $n_run already running.  Re-run with CONFIRM=1 to submit."
fi

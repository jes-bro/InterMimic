#!/bin/bash
# List cross-pair teacher training progress.
#
# Usage:
#   scripts/teacher_progress.sh                    # use default CHECKPOINTS_DIR
#   scripts/teacher_progress.sh /path/to/checkpoints
#
# Output: each teacher run's highest snapshot epoch, sorted descending,
# plus a summary at the bottom (most/least progress, total runs found).

set -e

CHECKPOINTS_DIR="${1:-/simurgh2/projects/ret-hoi/InterMimic/checkpoints}"

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CHECKPOINTS_DIR"
    echo "Pass a different path as the first argument."
    exit 1
fi

echo "Checkpoint dir: $CHECKPOINTS_DIR"
echo ""

# Collect (epoch, run_name) pairs
results=()
n_total=0
n_no_snapshot=0
for d in "$CHECKPOINTS_DIR"/smplx_crosspair_*/nn; do
    [ -d "$d" ] || continue
    n_total=$((n_total + 1))
    run_name=$(basename "$(dirname "$d")")

    # Find highest mimic_<N>.pth in this nn/ dir
    latest_epoch=$(ls "$d" 2>/dev/null \
        | grep -oE '^mimic_[0-9]+\.pth$' \
        | grep -oE '[0-9]+' \
        | sort -n \
        | tail -1)

    if [ -z "$latest_epoch" ]; then
        # No snapshot yet, but mimic.pth might exist
        if [ -f "$d/mimic.pth" ]; then
            results+=("0|$run_name|mimic.pth only")
        else
            n_no_snapshot=$((n_no_snapshot + 1))
            results+=("-1|$run_name|(no checkpoint)")
        fi
    else
        # Get mtime for "still running?" hint
        snap_file="$d/mimic_${latest_epoch}.pth"
        mtime=$(stat -c %Y "$snap_file" 2>/dev/null || stat -f %m "$snap_file")
        now=$(date +%s)
        age_min=$(( (now - mtime) / 60 ))
        if [ "$age_min" -lt 30 ]; then
            tag="(active, ${age_min}m ago)"
        elif [ "$age_min" -lt 1440 ]; then
            tag="(${age_min}m ago)"
        else
            tag="($((age_min / 60))h ago)"
        fi
        results+=("$latest_epoch|$run_name|$tag")
    fi
done

if [ ${#results[@]} -eq 0 ]; then
    echo "No smplx_crosspair_*/nn/ directories found in $CHECKPOINTS_DIR"
    exit 1
fi

# Sort results by epoch number (descending)
printf '%s\n' "${results[@]}" | sort -t'|' -k1,1 -n -r | while IFS='|' read -r epoch name tag; do
    if [ "$epoch" = "-1" ]; then
        printf '  %-50s %s\n' "$name" "$tag"
    else
        # Force base-10 interpretation: "00004900" -> 4900 (avoids octal error)
        epoch=$((10#$epoch))
        printf '  %-50s epoch %6d  %s\n' "$name" "$epoch" "$tag"
    fi
done

# Summary
echo ""
echo "Total teacher runs found: $n_total"
echo "Runs with no checkpoint yet: $n_no_snapshot"

max_epoch=$(printf '%s\n' "${results[@]}" | awk -F'|' '$1 > 0 { print $1 }' | sort -n | tail -1)
min_epoch=$(printf '%s\n' "${results[@]}" | awk -F'|' '$1 > 0 { print $1 }' | sort -n | head -1)
if [ -n "$max_epoch" ]; then
    echo "Furthest along: epoch $max_epoch"
    echo "Least far:      epoch $min_epoch"
fi

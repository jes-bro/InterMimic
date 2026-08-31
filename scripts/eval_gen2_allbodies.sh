#!/bin/sh
# eval_gen2_allbodies.sh -- submit FULL-BODY evals for the gen-2 grid.
#
# The gen-2 evals so far only scored each fold's three HELD-OUT bodies, so every
# in-distribution bar in the per-subject plot is empty. This scores all 16 real
# OMOMO subjects against the run's own source, for both folds, so
# scripts/plot_gen2_by_subject.py can draw in-distribution beside held-out and
# report the generalization gap.
#
# sub4 IS EXCLUDED. Its MJCF crashes the simulator; including it kills the whole
# job rather than that one pair.
#
# Usage (from the repo root, on the cluster):
#   sh scripts/eval_gen2_allbodies.sh            # mlp and xf, folds 0 and 1
#   sh scripts/eval_gen2_allbodies.sh mlp        # one architecture
#   DRY=1 sh scripts/eval_gen2_allbodies.sh      # print the plan, submit nothing
#
# Env overrides:
#   FOLDS="f0"          just one fold (default "f0 f1")
#   CELLS="ret_stock"   just one grid cell (default all four)
#   F0_EPOCH=00054600   which numbered checkpoint the fold-0 runs are scored at.
#                       "latest" takes the newest, PRINTED so it stays on record.
#   F0_ROOT / F1_ROOT   where each fold's checkpoints live
#   TAG=allbodies       goes in the CSV name
#
# Fold 1's checkpoints came from a collaborator and live under a different root
# with a rolling mimic.pth rather than numbered snapshots, which is why the two
# folds are resolved separately instead of by one pattern.
#
# Everything about arch, betas, source and test yaml is resolved by
# scripts/eval_one.sh from each run's OWN config -- this script only decides
# WHICH runs, WHICH checkpoint and WHICH bodies. Keeping that resolution in one
# place is deliberate: a mismatched betas file corrupts the 32 beta observation
# dims and still runs to completion, producing numbers that look fine.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

ARCHS="${1:-mlp xf}"
FOLDS="${FOLDS:-f0 f1}"
CELLS="${CELLS:-plain_stock plain_nvadlr ret_stock ret_nvadlr}"
TAG="${TAG:-allbodies}"
F0_EPOCH="${F0_EPOCH:-00054600}"
F0_ROOT="${F0_ROOT:-checkpoints}"
F1_ROOT="${F1_ROOT:-collab/jm/checkpointsjm}"

# All real OMOMO subjects except sub4. Held-out vs in-distribution is NOT
# decided here -- every run is scored on every body, and the plotter labels each
# one from the fold design. That way one CSV serves both halves of the figure.
BODIES="sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub10 sub11 sub12 sub13 sub14 sub15 sub16 sub17"

n_sub=0
n_skip=0
skipped=""

for A in $ARCHS; do
for F in $FOLDS; do
for C in $CELLS; do
    run="g2_${A}_${C}__${F}"
    dir="smplx_teacher_${run}"

    case "$F" in
      f0)
        if [ "$F0_EPOCH" = latest ]; then
            ck=$(ls -1 "$F0_ROOT/$dir/nn"/mimic_0*.pth 2>/dev/null | sort | tail -1)
            [ -n "$ck" ] && echo "[gen2-eval] $run: F0_EPOCH=latest resolved to $(basename "$ck")"
        else
            ck="$F0_ROOT/$dir/nn/mimic_${F0_EPOCH}.pth"
        fi
        out="eval_results/g2_${A}_${C}__f0_ep${F0_EPOCH}_${TAG}.csv"
        ;;
      f1)
        # Rolling checkpoint: these runs saved no numbered snapshots.
        ck="$F1_ROOT/$dir/nn/mimic.pth"
        out="eval_results/g2_${A}_${C}__f1_final_${TAG}.csv"
        ;;
      *)
        echo "[gen2-eval] ERROR: unknown fold '$F' (expected f0 or f1)" >&2
        exit 2
        ;;
    esac

    # A missing checkpoint is REPORTED, not guessed around. The transformer arms
    # and the collaborator's fold-1 runs are not necessarily where fold-0's are,
    # and submitting a job that dies in three seconds hides that.
    if [ -z "${ck:-}" ] || [ ! -f "$ck" ]; then
        echo "[gen2-eval] SKIP $run: no checkpoint at ${ck:-<none found>}"
        n_skip=$((n_skip + 1))
        skipped="$skipped $run"
        continue
    fi
    # An existing CSV is only a result if it holds usable rows. A job that hit a
    # bad GPU writes a FULL csv whose every row is exit_code=1 with empty
    # metrics -- 17 lines, looks finished, contains nothing. Count the usable
    # rows (metrics present AND exit_code 0) and redo the file when there are
    # none, saying so rather than skipping over a hole.
    n_bodies=$(echo $BODIES | wc -w)
    resume=0
    if [ -f "$out" ]; then
        ok=$(awk -F, 'FNR>1 && $5!="" && $10=="0"{n++} END{print n+0}' "$out")
        if [ "$ok" -ge "$n_bodies" ]; then
            echo "[gen2-eval] SKIP $run: $out is complete ($ok/$n_bodies)"
            n_skip=$((n_skip + 1))
            skipped="$skipped $run"
            continue
        elif [ "$ok" -gt 0 ]; then
            # Partial -- a walltime timeout, or a crash part way through. Keep
            # what it paid for and evaluate only the missing bodies.
            echo "[gen2-eval] RESUME $run: $out has $ok/$n_bodies -- running the remaining $((n_bodies - ok))"
            resume=1
        else
            echo "[gen2-eval] REDO $run: $out has 0/$n_bodies usable rows (every pair failed) -- replacing"
            rm -f "$out"
        fi
    fi

    echo "[gen2-eval] $run -> $out"
    if [ "${DRY:-0}" = 1 ]; then
        DRY=1 RESUME="$resume" BODIES="$BODIES" OUT="$out" sh scripts/eval_one.sh "$run" "$ck"
    else
        RESUME="$resume" BODIES="$BODIES" OUT="$out" sh scripts/eval_one.sh "$run" "$ck"
    fi
    n_sub=$((n_sub + 1))
done
done
done

echo
echo "[gen2-eval] ${n_sub} submitted, ${n_skip} skipped${skipped:+ ($skipped )}"
[ "$n_sub" -gt 0 ] && cat <<'EOF'

Each job scores 16 bodies at ~4-5 min/pair (measured), so ~70-90 min. The job
script asks for 8 h, which blocks backfill -- prefix SBATCH_TIMELIMIT=02:00:00
to ask for a realistic slot instead.

A job that finishes in under a minute did NOT succeed: check the CSVs with
  awk -F, 'FNR>1 && ($5=="" || $10!="0"){print FILENAME": "$1}' eval_results/g2_*allbodies*.csv
Silence means every row is filled. Re-running this script replaces any CSV whose
rows all failed.

When they finish:
  rsync -av --include='g2_*_allbodies.csv' --exclude='*' \
    <cluster>:/simurgh2/projects/ret-hoi/InterMimic/eval_results/ \
    ~/Downloads/eval_resultsaug31/
  python3 scripts/plot_gen2_by_subject.py --in ~/Downloads/eval_resultsaug31 \
    --include 'g2_mlp_*allbodies*' \
    --out ~/Downloads/eval_resultsaug31/gen2_mlp_by_subject.png
EOF
exit 0

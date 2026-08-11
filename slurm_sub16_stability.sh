#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="sub16-stability"
#SBATCH --output=sub16-stability-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Within-run stability probe: eval ONE (body, source) pair at several earlier
# checkpoints of one run, to tell "the final checkpoint dipped" apart from
# "the run never learned this body". Motivated by g2_mlp_ret_nvadlr__f0's
# sub16=9.6 at 10.6k, vs its sibling ret_stock's 44.2 -- with eval noise
# measured ~= 0 (repeat_A/B), any spread across checkpoints here is real
# within-run variation, not measurement error.
#
# Reuses slurm_eval_curriculum.sh per checkpoint (run with `bash`, so its
# #SBATCH lines are ignored) -- betas/arch guards and TERM_REASON forwarding
# are shared, not reimplemented. ~12 min per checkpoint.
#
# Usage (from repo root):
#   sbatch slurm_sub16_stability.sh                       # defaults below
#   RUN=g2_mlp_ret_stock__f0 EPOCHS="00008000 00009000 00010000" \
#     BODY=sub16 SOURCE=sub2 sbatch slurm_sub16_stability.sh
set -u
cd "$(dirname "$0")"

RUN="${RUN:-g2_mlp_ret_nvadlr__f0}"
EPOCHS="${EPOCHS:-00008000 00009000 00010000}"
BODY="${BODY:-sub16}"
SOURCE="${SOURCE:-sub2}"

CKDIR="checkpoints/smplx_teacher_${RUN}/nn"
TRAINY="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_${RUN}.yaml"
[ -d "$CKDIR" ] || { echo "[stability] ERROR: no checkpoint dir $CKDIR" >&2; exit 2; }
[ -f "$TRAINY" ] || { echo "[stability] ERROR: no train cfg $TRAINY" >&2; exit 2; }

# Fail on missing snapshots BEFORE burning GPU time on the ones that exist.
for E in $EPOCHS; do
    [ -f "$CKDIR/mimic_${E}.pth" ] || {
        echo "[stability] ERROR: $CKDIR/mimic_${E}.pth not found. Available:" >&2
        ls -1v "$CKDIR" | tail -8 >&2; exit 2; }
done

echo "[stability] $RUN: $BODY x $SOURCE at epochs $EPOCHS  (job=$SLURM_JOB_ID host=$(hostname))"
mkdir -p eval_results

for E in $EPOCHS; do
    echo "[stability] ==== epoch $E ===="
    CHECKPOINT="$CKDIR/mimic_${E}.pth" \
    OUT="eval_results/${RUN}__${E}__${BODY}only.csv" \
    BETAS_FILE=scripts/omomo_betas_neutral_aug.npz \
    BODIES="$BODY" SOURCES="$SOURCE" \
    TERM_REASON=1 \
    TRAIN_YAML="$TRAINY" \
    bash slurm_eval_curriculum.sh
done

echo "[stability] ==== summary ===="
for E in $EPOCHS; do
    f="eval_results/${RUN}__${E}__${BODY}only.csv"
    [ -f "$f" ] && echo "epoch $E: $(tail -1 "$f" | cut -d, -f7)% success" || echo "epoch $E: NO CSV (crashed?)"
done

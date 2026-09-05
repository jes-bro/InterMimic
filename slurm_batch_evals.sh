#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="batch-eval"
#SBATCH --output=batch-eval-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Run the 6 curriculum evals on ONE gpu, sequentially, so they cost ONE gpu slot
# instead of one-per-eval.
#
# IMPORTANT walltime note: a FULL-matrix eval (6x6 bodies) is ~8h each, so 6 of
# them bundled = ~48h on one gpu -- too slow. Use this bundle for the FAST
# single-pair SANITY evals (default BODIES=sub2 SOURCES=sub2, ~12min each -> 6
# runs in ~1-2h on one gpu) to validate all 6 checkpoints cheaply. For the full
# 8h matrices, run them as SEPARATE jobs under a 4-gpu account cap instead.
#
#   sbatch slurm_batch_evals.sh                          # fast sanity, all 6, one gpu
#   BODIES="sub1 sub2 sub3 sub5 sub9 sub17" SOURCES="sub2 sub6" sbatch slurm_batch_evals.sh   # bigger (slower)
#
# Reuses slurm_eval_curriculum.sh per run (its #SBATCH lines ignored under `bash`),
# so betas/arch/checkpoint handling + no-silent-fallback guards are shared.

set -u
cd "$(dirname "$0")"
mkdir -p eval_results

# Default to the FAST single pair so the bundle stays short on one gpu.
export BODIES="${BODIES:-sub2}"
export SOURCES="${SOURCES:-sub2}"

# run | betas file | arch   (checkpoint resolved from the run's *final* dir per loop)
RUNS=(
  "ALLON_NOSUB_XF|scripts/omomo_betas_neutral_aug.npz|xf"
  "baseline_flong|scripts/omomo_betas.npz|mlp"
  "ist_flong_bn_mlp|scripts/omomo_betas.npz|mlp"
  "ist_neutral_bn_mlp|scripts/omomo_betas_neutral.npz|mlp"
  "ist_neutral_bn_xf|scripts/omomo_betas_neutral.npz|xf"
  "SYNAUG_XF_POSE|scripts/omomo_betas_neutral_aug.npz|xf"
)

echo "[batch-eval] bodies=($BODIES) x sources=($SOURCES) on ONE gpu host=$(hostname) job=$SLURM_JOB_ID"
for E in "${RUNS[@]}"; do
    IFS='|' read -r R B A <<< "$E"
    # final checkpoint = highest-step .pth in the run's *final* dir
    C=$(ls -1v checkpoints/*"$R"*final*/nn/*.pth 2>/dev/null | tail -1)
    if [ -z "$C" ]; then echo "[batch-eval] SKIP $R: no final checkpoint found"; continue; fi
    if [ "$A" = xf ]; then
        BY=isaacgym/src/intermimic/data/cfg/omomo_eval_v1_multibody_xf.yaml
        TY=$(ls -t curriculum_work/"$R"/cfgs/train_*.yaml 2>/dev/null | head -1)
        if [ -z "$TY" ]; then echo "[batch-eval] SKIP $R: no transformer train yaml"; continue; fi
    else
        BY=isaacgym/src/intermimic/data/cfg/omomo_eval_v1_multibody_mlp.yaml
        TY=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml
    fi
    echo "======== eval $R ($A)  ckpt=$C ========"
    CHECKPOINT="$C" BETAS_FILE="$B" BASE_YAML="$BY" TRAIN_YAML="$TY" \
        OUT="eval_results/${R}__batch.csv" \
        bash slurm_eval_curriculum.sh \
        || echo "[batch-eval] $R FAILED -- continuing to next"
done
echo "[batch-eval] all done. CSVs in eval_results/*__batch.csv"

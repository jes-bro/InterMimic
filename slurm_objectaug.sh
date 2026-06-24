#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="objectaug"
#SBATCH --output=objectaug-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# objectAug curriculum: warm-start the fold-in policy, then train on objects that
# are scaled / rotated / translated, widening the perturbation across stages. The
# reward COMBO is fixed for the run (VARIANT); only the ranges widen. The
# controller holds this one GPU and manages the training subprocess per stage.
# See scripts/objectaug_runner.py.
#
# --resume makes requeues safe: the controller persists objectaug_work/<run>/
# state.json after each stage, so a requeued 48h job continues from the last
# completed stage instead of restarting. (Fresh run: state.json absent -> stage 0.)
#
# One job = one VARIANT (separate sbatch per job; no arrays). To sweep all eight:
#   for V in drop_base drop_pose drop_hold drop_both \
#            keep_base keep_pose keep_hold keep_both; do
#     VARIANT=$V INIT_CKPT=checkpoints/<foldin>/nn/mimic.pth \
#       sbatch --job-name="oa_$V" slurm_objectaug.sh
#   done

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Required per job:
#   VARIANT   one of drop_{base,pose,hold,both} / keep_{base,pose,hold,both}
#   INIT_CKPT fold-in policy .pth to warm-start stage 0
# Optional:
#   RUN_NAME  names objectaug_work/<run>/ + checkpoints (default: oa_$VARIANT)
VARIANT="${VARIANT:?set VARIANT=drop_both (or another of the 8 combos)}"
INIT_CKPT="${INIT_CKPT:?set INIT_CKPT=checkpoints/<foldin>/nn/mimic.pth}"
RUN_NAME="${RUN_NAME:-oa_$VARIANT}"

python scripts/objectaug_runner.py \
    --run-name "$RUN_NAME" --variant "$VARIANT" \
    --init-checkpoint "$INIT_CKPT" --resume

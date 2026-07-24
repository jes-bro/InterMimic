#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

#SBATCH --job-name="rt-viz"
#SBATCH --output=rt-viz-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# CPU-ONLY render of the contact retargeting (no GPU, no Isaac Gym). For each
# (body, clip) it writes a side-by-side GIF: BEFORE = target body driven by the
# SOURCE's dof (the reference the policy currently gets, hands off the object) vs
# AFTER = retargeted (hands on the object). WATCH the gifs to judge -- the numbers
# already say the gap closed; this shows WHERE.
#
# GIF (no ffmpeg needed). Lands in render_results/retarget_viz_<jobid>/. Override
# TARGETS / CLIP / RETARGET_DIR / STRIDE. Runs from repo root.
#
#   sbatch slurm_visualize_retarget.sh
#   TARGETS="sub6 sub9" CLIP=sub2_largetable_005.pt sbatch slurm_visualize_retarget.sh

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="${SOURCE:-sub2}"
TARGETS="${TARGETS:-sub6 sub9}"
CLIP="${CLIP:-sub2_largetable_000.pt}"
RETARGET_DIR="${RETARGET_DIR:-InterAct/OMOMO_retarget_contact_smoke}"
STRIDE="${STRIDE:-4}"
DEST="render_results/retarget_viz_${SLURM_JOB_ID:-local}"
mkdir -p "$DEST"

echo "[rt-viz] host=$(hostname) job=$SLURM_JOB_ID  source=$SOURCE targets=($TARGETS) clip=$CLIP"
echo "[rt-viz] gifs -> $DEST"
for T in $TARGETS; do
    echo "-- $SOURCE -> $T --"
    python3 scripts/visualize_retarget.py \
        --clip "InterAct/OMOMO_new/$CLIP" --source "$SOURCE" --target "$T" \
        --retarget-dir "$RETARGET_DIR" --stride "$STRIDE" \
        --out "$DEST/retarget_${T}_${CLIP%.pt}.gif" || echo "[rt-viz] FAILED for $T"
done
echo
echo "[rt-viz] done. gifs in $DEST -- BEFORE|AFTER; watch the red hands move onto the blue object."
ls -lh "$DEST"/*.gif 2>/dev/null

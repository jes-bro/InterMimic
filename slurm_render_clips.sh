#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="render-clips"
#SBATCH --output=render-clips-%A_%a.out
#SBATCH --array=0-12          # one task per object (13 objects)
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=END,FAIL

# Render every OMOMO clip to an Isaac Gym replay mp4, one object per array task.
# Submit:   sbatch slurm_render_clips.sh
# Single object (no array):  OBJECT=woodchair sbatch --array=0 slurm_render_clips.sh
# Output:   clip_videos/<object>/<clip>.mp4   (override dir with OUT=...)

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The 13 OMOMO objects, ordered smallest clip-count first so the quick ones
# finish (and can be eyeballed) early.
OBJECTS=(monitor smallbox plasticbox trashcan largebox suitcase largetable \
         floorlamp clothesstand smalltable woodchair whitechair tripod)

# OBJECT env var overrides the array index (for a one-off single-object run).
OBJ="${OBJECT:-${OBJECTS[$SLURM_ARRAY_TASK_ID]}}"
export OUT="${OUT:-clip_videos}"

echo "[slurm] task=$SLURM_ARRAY_TASK_ID object=$OBJ out=$OUT/$OBJ"
scontrol update JobId="$SLURM_JOB_ID" JobName="render-$OBJ" 2>/dev/null || true

OBJECT="$OBJ" OUT="$OUT" sh scripts/render_clips.sh

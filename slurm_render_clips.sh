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
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=END,FAIL

# Render every OMOMO clip to an Isaac Gym replay mp4, replaying each clip on the
# SUBJECT and OBJECT it was actually recorded with.
#
# One array task per (subject, object) pair -- NOT per object. Both the object mesh
# and the humanoid MJCF are fixed at env creation (env e gets object e%num_objects
# and asset e%num_assets), so at num_envs=1 env 0 holds exactly one of each. Pinning
# only the object -- the old behavior -- silently rendered all 17 subjects on the
# generic smplx/omomo.xml body. See scripts/render_clips.sh.
#
# The pair list is generated from the motion dir (files are sub<N>_<object>_<idx>.pt),
# so only pairs that actually have clips get a task -- no empty jobs.
#
# Submit (from repo root):
#   sh scripts/gen_render_pairs.sh                       # -> render_pairs.txt
#   sbatch --array=0-$(( $(wc -l < render_pairs.txt) - 1 ))%16 slurm_render_clips.sh
#
# Re-running is safe: render_all_clips skips clips whose mp4 already exists, so a
# timed-out task resumes where it stopped.
#
# Output: $OUT/<object>/<clip>.mp4  (clip names carry the subject, e.g.
#         clip_videos/woodchair/sub2_woodchair_000.mp4)

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PAIRS="${PAIRS:-render_pairs.txt}"
[ -f "$PAIRS" ] || { echo "[slurm] FATAL: $PAIRS missing -- run: sh scripts/gen_render_pairs.sh"; exit 1; }

# Array index -> the (subject object) on that line. sed is 1-indexed, array is 0-indexed.
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIRS")
[ -n "$line" ] || { echo "[slurm] FATAL: no line $((SLURM_ARRAY_TASK_ID + 1)) in $PAIRS"; exit 1; }
read -r SUB OBJ <<< "$line"

export OUT="${OUT:-clip_videos}"

echo "[slurm] task=$SLURM_ARRAY_TASK_ID subject=$SUB object=$OBJ out=$OUT/$OBJ"
scontrol update JobId="$SLURM_JOB_ID" JobName="render-${SUB}-${OBJ}" 2>/dev/null || true

SUBJECT="$SUB" OBJECT="$OBJ" OUT="$OUT" sh scripts/render_clips.sh

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="render-sub2"
#SBATCH --output=render-sub2-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=END,FAIL

# Render EVERY clip of one subject (default sub2) to an Isaac Gym replay mp4, each
# replayed on THAT subject's own body + the object it was recorded with -- so you can
# watch them to hand-annotate activity/style labels.
#
# One job, no array setup: it discovers the subject's objects from the motion dir and
# loops scripts/render_clips.sh over them (which pins dataSub=subjectBodies=[SUBJECT]
# and dataObjects=[OBJECT] so env 0 has exactly that body + object). sub2 = 52 clips
# across 4 objects, so this finishes well inside the wall-clock.
#
# Submit from the repo root:
#   sbatch slurm_render_sub2.sh
#   SUBJECT=sub6 sbatch slurm_render_sub2.sh          # reuse for another subject
#   OUT=/scratch/$USER/clip_videos sbatch slurm_render_sub2.sh   # bigger subjects
#
# Output: $OUT/<object>/<clip>.mp4  e.g. sub2_clip_videos/woodchair/sub2_woodchair_000.mp4
# Re-running is safe: render_all_clips skips clips whose mp4 already exists.

set -eu
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SUBJECT="${SUBJECT:-sub2}"
# Where the clips live -- MUST match omomo_render.yaml's motion_file so we discover
# exactly the clips render_clips.sh will load. Override if your cluster path differs.
MOTION_DIR="${MOTION_DIR:-InterAct/OMOMO_new}"
OUT="${OUT:-${SUBJECT}_clip_videos}"

[ -d "$MOTION_DIR" ] || { echo "[render-sub2] FATAL: motion dir '$MOTION_DIR' not found (set MOTION_DIR=)"; exit 1; }

# Discover this subject's objects the SAME way intermimic does: clip files are
# sub<N>_<object>_<idx>.pt and object = the split('_')[-2] field. No hardcoded list,
# so it can't silently render the wrong/stale set.
OBJECTS=$(ls "$MOTION_DIR"/${SUBJECT}_*.pt 2>/dev/null \
          | xargs -n1 basename \
          | sed -E 's/\.pt$//' \
          | awk -F_ '{print $(NF-1)}' \
          | sort -u)

[ -n "$OBJECTS" ] || { echo "[render-sub2] FATAL: no clips '${SUBJECT}_*.pt' under '$MOTION_DIR'"; exit 1; }

NCLIPS=$(ls "$MOTION_DIR"/${SUBJECT}_*.pt 2>/dev/null | wc -l)
echo "[render-sub2] subject=$SUBJECT  motion_dir=$MOTION_DIR  clips=$NCLIPS  out=$OUT"
echo "[render-sub2] objects: $(echo $OBJECTS | tr '\n' ' ')"

for OBJ in $OBJECTS; do
    echo "[render-sub2] === rendering $SUBJECT / $OBJ ==="
    SUBJECT="$SUBJECT" OBJECT="$OBJ" OUT="$OUT" sh scripts/render_clips.sh
done

echo "[render-sub2] DONE -> $OUT/<object>/*.mp4"

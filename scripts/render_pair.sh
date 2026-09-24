#!/bin/sh
# render_pair.sh -- one qualitative video of ONE (body, source) pair, for today's
# arms. Companion to render_qualitative.sh, which builds a specific Sept-21 figure
# from configs it writes itself (numObs 3230/3198, crosspair teachers) and does
# not run against the g3 students.
#
# WHICH PAIR. One that SCORES WELL, read off the eval CSVs -- a figure should show
# the method working, and the numbers say where it does. For a baseline-vs-ours
# comparison, render the SAME body and source with both checkpoints: same person,
# same motion, so the difference is the policy and nothing else.
#
# CAMERA. Framed on the clip env 0 actually plays (scripts/cam_for_clip.py). The
# default view (3,3,2.5)->(0,0,1) frames the origin, so any motion that covers
# ground renders as an empty court.
#
# Usage (repo root; conda env active, LD_LIBRARY_PATH exported):
#   BODY=sub16 SOURCE=sub458 OUT=$HOME/render_bball.mp4 GPU=0 \
#   ARM=student_g3_act_xf_ret_nvadlr__f0 \
#   CKPT=checkpoints/smplx_student_g3_act_xf_ret_nvadlr__f0/nn/mimic_00100000.pth \
#   sh scripts/render_pair.sh
#
#   DRY=1 ...                      print the plan (incl. the clip and camera), run nothing
#   DUMP=$HOME/render.npz ...      also dump the trajectory, for a mesh render
#
# ARM picks the arm's own eval + train configs, so the video comes from the same
# environment as its numbers. A policy with no arm (the InterMimic baseline) sets
# them directly:
#   ENV_YAML=isaacgym/src/intermimic/data/cfg/vanilla_intermimic_eval.yaml \
#   TRAIN_YAML=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all.yaml \
#   TASK=InterMimic_All CKPT=checkpoints/vanilla/student.pth
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

BODY="${BODY:?set BODY, e.g. BODY=sub16}"
SOURCE="${SOURCE:?set SOURCE, e.g. SOURCE=sub458}"
OUT="${OUT:?set OUT, e.g. OUT=\$HOME/render_bball.mp4}"
CKPT="${CKPT:?set CKPT}"
ARM="${ARM:-}"
GPU="${GPU:-0}"
FRAMES="${FRAMES:-400}"
NUM_ENVS="${NUM_ENVS:-16}"
ENTRY="${ENTRY:-intermimic.run_distill}"
TASK="${TASK:-InterMimicDistillG3}"
CFG=isaacgym/src/intermimic/data/cfg

if [ -n "$ARM" ]; then
  ENV_YAML="${ENV_YAML:-$CFG/omomo_eval_${ARM}.yaml}"
  TRAIN_YAML="${TRAIN_YAML:-$CFG/train/rlg/omomo_${ARM}.yaml}"
else
  : "${ENV_YAML:?set ENV_YAML (or ARM)}"
  : "${TRAIN_YAML:?set TRAIN_YAML (or ARM)}"
fi

for f in "$CKPT" "$ENV_YAML" "$TRAIN_YAML"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done

# env 0 plays the FIRST clip of the source, so that is the clip to frame. Framing
# a different clip of the same source gives a correct-looking command and an
# empty court.
TREE=$(grep -m1 '^[[:space:]]*retargetedMotionDir:' "$ENV_YAML" | awk '{print $2}')
[ -n "$TREE" ] || { echo "ERROR: no retargetedMotionDir in $ENV_YAML" >&2; exit 2; }
CLIP=$(ls "$TREE/$BODY/${SOURCE}_"*.pt 2>/dev/null | head -1)
[ -n "$CLIP" ] || { echo "ERROR: no ${SOURCE}_*.pt under $TREE/$BODY -- is that pair in this tree?" >&2; exit 2; }

echo "== render  body=$BODY  source=$SOURCE  gpu=$GPU"
echo "   checkpoint : $CKPT"
echo "   env cfg    : $ENV_YAML"
echo "   entry/task : $ENTRY / $TASK"
echo "   clip (env0): $CLIP"
echo "   -> video   : $OUT${DUMP:+    -> dump: $DUMP}"

CAM=$(python scripts/cam_for_clip.py "$CLIP")
echo "   camera     : $(echo "$CAM" | tr '\n' ' ')"

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

eval "$CAM"
# env, not a bare assignment prefix: in dash a word from an expansion (the
# ${DUMP:+...} below) ends the prefix, and everything after it is parsed as the
# command -- the same way a sweep died earlier with "BODIES=sub1: not found".
env RECORD_VIDEO="$OUT" MAX_VIDEO_FRAMES="$FRAMES" \
${DUMP:+DUMP_TRAJ="$DUMP"} ${DUMP:+DUMP_FRAMES="$FRAMES"} \
RECORD_VIDEO_CAM_POS="${RECORD_VIDEO_CAM_POS:-}" \
RECORD_VIDEO_CAM_TARGET="${RECORD_VIDEO_CAM_TARGET:-}" \
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}" \
python -u -m "$ENTRY" --task "$TASK" --cfg_env "$ENV_YAML" --cfg_train "$TRAIN_YAML" \
    --test --headless --num_envs "$NUM_ENVS" --checkpoint "$CKPT" \
    --subject_bodies "$BODY" --data_sub "$SOURCE"

echo "== wrote:"
ls -l "$OUT" ${DUMP:+"$DUMP"}

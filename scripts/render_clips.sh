#!/bin/sh
# Render ONE Isaac Gym replay mp4 per motion clip -- one OBJECT per invocation.
#
# Why per-object: each env's object MESH is fixed at creation (env e owns object
# e % num_objects), so with num_envs=1 env 0 can only correctly render clips of a
# single object. We therefore filter the dataset to one object per launch and let
# render_all_clips (intermimic.py) sweep every clip of that object through env 0.
#
# This is a kinematic replay: the SMPL-X body + object are posed directly from the
# mocap each frame (contacts colored red) -- the real sim view, not a skeleton.
#
# Usage (from repo root, in the intermimic-gym conda env, ON A GPU):
#   OBJECT=woodchair OUT=clip_videos sh scripts/render_clips.sh
# Output: <OUT>/<object>/<clip>.mp4  (e.g. clip_videos/woodchair/sub2_woodchair_000.mp4)
set -eu
REPO="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO"

OBJECT="${OBJECT:?set OBJECT=<name>, e.g. OBJECT=woodchair}"
# OUT base defaults to <repo>/clip_videos, but SET OUT=/abs/scratch/dir on the
# cluster -- 4421 mp4s are several GB, don't dump them in the repo/quota'd FS.
# Resolve to an ABSOLUTE path so it's deterministic no matter the process CWD.
OUT_BASE="${OUT:-$REPO/clip_videos}"
case "$OUT_BASE" in /*) : ;; *) OUT_BASE="$REPO/$OUT_BASE" ;; esac
OUT="$OUT_BASE/$OBJECT"
NUM_ENVS="${NUM_ENVS:-1}"       # only env 0 is captured; keep it at 1
FPS="${FPS:-30}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_render.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml

# Write a temp config that restricts dataObjects to just this object, so only its
# clips load and env 0's mesh matches every clip we render.
TMP="$(mktemp --suffix="_render_${OBJECT}.yaml")"
python3 - "$BASE" "$TMP" "$OBJECT" <<'PY'
import re, sys
base, tmp, obj = sys.argv[1:4]
t = open(base).read()
line = f"  dataObjects: ['{obj}']"
if re.search(r'^\s*dataObjects:', t, re.M):
    t = re.sub(r'^\s*dataObjects:.*$', line, t, flags=re.M)
else:                                   # insert right after the dataSub line
    t = re.sub(r'(^\s*dataSub:.*$)', r'\1\n' + line, t, flags=re.M)
open(tmp, 'w').write(t)
PY

mkdir -p "$OUT"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
export RENDER_CLIPS_OUT="$OUT" RENDER_CLIPS_FPS="$FPS"
# RENDER_CLIPS_LO / RENDER_CLIPS_HI can subset the clip range if you want to split
# a big object across jobs; unset = all clips of this object.

echo "[render] object=$OBJECT -> $OUT  (num_envs=$NUM_ENVS, fps=$FPS)"
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$TMP" \
    --cfg_train "$TRAIN" \
    --test --play_dataset --headless \
    --num_envs "$NUM_ENVS"

rm -f "$TMP"
echo "[render] done: $OUT"

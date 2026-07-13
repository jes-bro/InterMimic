#!/bin/sh
# Render ONE Isaac Gym replay mp4 per motion clip -- one (SUBJECT, OBJECT) pair
# per invocation.
#
# Why the pair, and not just the object: BOTH the object mesh and the humanoid
# MJCF are fixed at env-creation time --
#   object: env e owns object e % num_objects  (intermimic.py)
#   body:   env e owns asset  e % num_assets   (humanoid.py:309)
# -- so with num_envs=1, env 0 has exactly ONE object mesh and ONE body. To replay
# each clip on the subject and object it was actually recorded with, we must pin
# BOTH: dataObjects=[OBJECT], and dataSub=subjectBodies=[SUBJECT]. Pinning only the
# object (the old behavior) silently replayed all 17 subjects' clips on the generic
# smplx/omomo.xml body, so limb lengths didn't match the performer and hands/feet
# drifted off the object. Same invariant slurm_replay.sh enforces.
#
# This is a kinematic replay: the SMPL-X body + object are posed directly from the
# mocap each frame (contacts colored red) -- the real sim view, not a skeleton.
#
# Usage (from repo root, in the intermimic-gym conda env, ON A GPU):
#   SUBJECT=sub2 OBJECT=woodchair OUT=clip_videos sh scripts/render_clips.sh
# Output: <OUT>/<object>/<clip>.mp4  (e.g. clip_videos/woodchair/sub2_woodchair_000.mp4)
# Clip names already carry the subject, so pairs sharing an object coexist in one dir.
set -eu
REPO="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO"

OBJECT="${OBJECT:?set OBJECT=<name>, e.g. OBJECT=woodchair}"
# SUBJECT pins BOTH the clips we load (dataSub) and the body in sim (subjectBodies)
# to the performer who actually recorded them. Required -- no default, because a
# silent fallback here means rendering the wrong body for a whole object.
SUBJECT="${SUBJECT:?set SUBJECT=<subN>, e.g. SUBJECT=sub2}"
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

# Write a temp config pinning this launch to exactly one (subject, object):
#   dataObjects   -> [OBJECT]   so env 0's object mesh matches every clip
#   dataSub       -> [SUBJECT]  so only that performer's clips load
#   subjectBodies -> [SUBJECT]  so env 0's MJCF *is* that performer's body
# subjectBodies must equal dataSub or we replay a subject's motion on someone
# else's skeleton. Each key is replaced if present, else inserted after dataSub.
TMP="$(mktemp --suffix="_render_${SUBJECT}_${OBJECT}.yaml")"
python3 - "$BASE" "$TMP" "$OBJECT" "$SUBJECT" <<'PY'
import re, sys
base, tmp, obj, sub = sys.argv[1:5]
t = open(base).read()

def set_key(text, key, value):
    line = f"  {key}: {value}"
    if re.search(rf'^\s*{key}:', text, re.M):
        return re.sub(rf'^\s*{key}:.*$', line, text, flags=re.M)
    # Not present -> insert right after dataSub (always present in the base cfg).
    return re.sub(r'(^\s*dataSub:.*$)', r'\1\n' + line, text, flags=re.M)

t = set_key(t, 'dataSub',       f"['{sub}']")
t = set_key(t, 'subjectBodies', f"['{sub}']")
t = set_key(t, 'dataObjects',   f"['{obj}']")

# No-silent-fallback: if any key failed to land, the render would quietly use the
# wrong body/object for every clip. Fail here instead.
for k, v in (('dataSub', sub), ('subjectBodies', sub), ('dataObjects', obj)):
    if not re.search(rf"^\s*{k}: \['{v}'\]$", t, re.M):
        sys.exit(f"[render] FATAL: failed to set {k} -> ['{v}'] in {base}")
open(tmp, 'w').write(t)
PY

mkdir -p "$OUT"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
export RENDER_CLIPS_OUT="$OUT" RENDER_CLIPS_FPS="$FPS"
# RENDER_CLIPS_LO / RENDER_CLIPS_HI can subset the clip range if you want to split
# a big object across jobs; unset = all clips of this object.

echo "[render] subject=$SUBJECT object=$OBJECT -> $OUT  (num_envs=$NUM_ENVS, fps=$FPS)"
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$TMP" \
    --cfg_train "$TRAIN" \
    --test --play_dataset --headless \
    --num_envs "$NUM_ENVS"

rm -f "$TMP"
echo "[render] done: $OUT"

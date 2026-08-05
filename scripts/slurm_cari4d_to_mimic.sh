#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d2mimic"
#SBATCH --output=cari4d2mimic-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Carry a CARI4D reconstruction into Isaac Gym: convert, retarget, install, and
# record the replay. The four steps documented at the top of
# isaacgym/scripts/data_replay_cari4d.sh, run as one queued job, ending in an
# MP4 rather than a viewer nobody is sitting in front of.
#
#   sbatch scripts/slurm_cari4d_to_mimic.sh
#   SUBJECT_ID=101 OBJECT_NAME=chair sbatch scripts/slurm_cari4d_to_mimic.sh
#   RECORD_VIDEO_CAM_POS=8,8,5 sbatch scripts/slurm_cari4d_to_mimic.sh
#
# Queued rather than interactive because every step here imports scipy or torch
# from /simurgh2, and a cold import of those on a login node takes minutes --
# long enough to look like a hang. A compute node's page cache makes it quick,
# and the job keeps a log either way.
#
# 1h is generous for what is a few seconds of real computation once the imports
# are warm. Nothing here trains; the GPU is only needed because interact2mimic
# and intermimic.run import isaacgym, which enumerates devices at import.

set -euo pipefail

INTERMIMIC=/simurgh2/projects/ret-hoi/InterMimic
INTERACT=/simurgh2/projects/ret-hoi/InterAct
CARI4D=/simurgh2/projects/ret-hoi/CARI4D
CACHE_ROOT=/simurgh2/projects/ret-hoi

# The reconstruction to convert. Defaults to the egoexo4d basketball take.
BUNDLE="${BUNDLE:-$CARI4D/output/opt/cari4d-release+step031397_demo-hy3d3-optv2/Date03_Sub01_bball_dribble.pth}"
MESH="${MESH:-$CARI4D/data/cari4d-demo/meshes-metric/Date03_Sub01_bball_dribble_064_align/Date03_Sub01_bball_dribble_064_align.obj}"

# InterMimic parses a clip's subject from its filename as int(prefix[3:])
# (intermimic.py:63), so this must be an integer and the two scripts disagree on
# how to spell it: cari4d_to_interact.py wants 100, cari4d_finalize.py wants
# sub100. Both are derived from this one value so they cannot drift apart.
SUBJECT_ID="${SUBJECT_ID:-100}"
OBJECT_NAME="${OBJECT_NAME:-bball}"
CLIP_IDX="${CLIP_IDX:-000}"

# SMPL-H gender used during reconstruction. CARI4D's stage 2 predicts this per
# sequence (nlf-smplh-gender-sepK), so a wrong value here does not error -- it
# silently retargets onto the other body model and reads as a bad retarget.
GENDER="${GENDER:-male}"

# Must start with 'behave' or interact2mimic.py skips its SMPL-H/num_betas=10
# branch. It is a directory under InterAct/data/ shared by every converted clip;
# subjects coexist there and the env YAML's dataSub picks between them.
DATASET_TAG="${DATASET_TAG:-behave_cari4d}"

# MESH=1 builds per-bone convex-hull STLs instead of capsules. Off by default
# for two reasons. It does not work on this branch at all: smpl_local_robot.py
# :1505 passes zero_pose= to get_mesh_offsets for every non-smplx model, and
# SMPLH_Parser.get_mesh_offsets (smpl_parser.py:20091) has no such parameter, so
# SMPL-H plus hulls is a TypeError inside InterAct's vendored uhc. And capsules
# are the better rig regardless: their bone lengths come from the subject's own
# betas, so proportions still match, without the seam cracks and convex-hull
# infill the hulls introduce at joint rotations.
MESH="${MESH:-0}"

# Env config for the replay smoke test. Written by hand, not generated here.
CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball.yaml}"

# Set REPLAY=0 to stop after installing. Otherwise the replay runs headless and
# records to RECORD_VIDEO, the same mechanism render_qualitative.sh uses -- so a
# batch job with no display still produces something watchable.
REPLAY="${REPLAY:-1}"
RECORD_DIR="${RECORD_DIR:-$INTERMIMIC/renders}"
RECORD_VIDEO="${RECORD_VIDEO:-$RECORD_DIR/sub${SUBJECT_ID}_${OBJECT_NAME}_${CLIP_IDX}.mp4}"

# Frames to capture. The basketball clip is 101 frames, but the sim steps at a
# different rate than the source video, so this is not the clip length -- it is
# a ceiling. Too low silently truncates the motion.
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-600}"

# Camera, as "x,y,z". The default (3,3,2.5) looking at (0,0,1) frames a person
# at the origin. A subject who walks -- or whose world transform is wrong, which
# is the open question for this clip -- can leave frame entirely, so being able
# to pull the camera back matters more here than usual.
RECORD_VIDEO_CAM_POS="${RECORD_VIDEO_CAM_POS:-}"
RECORD_VIDEO_CAM_TARGET="${RECORD_VIDEO_CAM_TARGET:-}"

log() { echo "[c2m $(date -u +%H:%M:%S)] $*"; }

# Caches to project space; /sailhome is over quota.
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# Unbuffered, so a killed job still shows which step it reached.
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MIMIC_ENV:-intermimic-gym2}"

# isaacgym's gym_38.so links against libpython3.8.so.1.0, which ships in the
# conda env rather than any system path. Without this every isaacgym import
# dies with ImportError before doing anything.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=$CONDA_DEFAULT_ENV"
log "bundle=$BUNDLE"
log "mesh=$MESH"
log "subject=sub$SUBJECT_ID object=$OBJECT_NAME gender=$GENDER tag=$DATASET_TAG"

for required in "$BUNDLE" "$MESH" "$INTERACT/simulation/interact2mimic.py"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

# Step 1: CARI4D bundle -> InterAct format (human.npz, object.npz, mesh).
log "step 1/4: cari4d_to_interact"
cd "$INTERMIMIC"
python scripts/cari4d_to_interact.py \
    --bundle "$BUNDLE" \
    --mesh "$MESH" \
    --interact-root "$INTERACT" \
    --dataset-tag "$DATASET_TAG" \
    --gender "$GENDER" \
    --subject-id "$SUBJECT_ID" \
    --object-name "$OBJECT_NAME" \
    --clip-idx "$((10#$CLIP_IDX))"

# Step 2: InterAct -> InterMimic. Either rig derives its proportions from the
# subject's betas; see MESH above for why hulls are not the default.
#
# --only restricts this to the clip step 1 just wrote. Without it every sequence
# under sequences_canonical is retargeted, and step 3 then renames all of them
# to --subject-id -- so a second clip lands on the same MJCF path and one body
# silently overwrites the other.
SEQ_NAME="sub${SUBJECT_ID}_${OBJECT_NAME}_${CLIP_IDX}"
MESH_FLAG=""
if [ "$MESH" = "1" ]; then MESH_FLAG="--mesh"; fi
log "step 2/4: run_interact2mimic ($([ "$MESH" = "1" ] && echo hulls || echo capsules), only $SEQ_NAME)"
python scripts/run_interact2mimic.py \
    --interact-root "$INTERACT" \
    --dataset-name "$DATASET_TAG" \
    --only "$SEQ_NAME" \
    $MESH_FLAG

# Step 3: install the .pt and MJCF into this repo under the sub<N> naming.
# --subject-id takes the string form here; step 1 took the bare integer.
log "step 3/4: cari4d_finalize"
python scripts/cari4d_finalize.py \
    --interact-root "$INTERACT" \
    --intermimic-root "$INTERMIMIC" \
    --dataset-tag "$DATASET_TAG" \
    --subject-id "sub$SUBJECT_ID" \
    --clip-index "$CLIP_IDX"

log "installed artifacts:"
ls -la "$INTERACT/$DATASET_TAG"/sub"$SUBJECT_ID"_*.pt 2>/dev/null || true
ls -la "$INTERMIMIC/isaacgym/src/intermimic/data/assets/smplx/smplh_"*_sub"$SUBJECT_ID".xml 2>/dev/null || true

# Step 3.5: put the motion in a gravity-aligned frame. CARI4D reconstructs in
# the camera's frame, where 'up' is wherever that camera's up pointed -- for a
# camera looking level at a scene that is roughly world -Z, which is why an
# unrotated clip replays upside down. The calibration states the rotation
# exactly, so this is not a guess-and-check flip.
#
#   ROTATE_CALIB=/path/to/trajectory/gopro_calibs.csv:cam04
#
# Runs after step 3 because that is what installs the .pt, and before step 4 so
# the render shows the corrected motion. Re-running the whole job is safe: step
# 3 reinstalls an unrotated file each time, so the rotation is never applied
# twice.
ROTATE_CALIB="${ROTATE_CALIB:-}"

# ROTATE_AXIS/ROTATE_DEGREES is usually the right choice, NOT ROTATE_CALIB.
# interact2mimic.py:795 already applies its upright_start correction, which
# gravity-aligns the clip -- measured on the basketball take, the rig's local +Z
# came out 1.5 degrees from world -Z, i.e. exactly inverted and nothing else.
# --from-calib then rotates an already-world-aligned tensor by the full
# camera-to-world rotation and rolls it 19 degrees off vertical. Use the
# calibration only for a tensor still in raw camera frame.
#
#   ROTATE_AXIS=x ROTATE_DEGREES=180
ROTATE_AXIS="${ROTATE_AXIS:-}"
ROTATE_DEGREES="${ROTATE_DEGREES:-180}"

# Rotate about each frame's root rather than the world origin. Keeps the figure
# where it was instead of swinging it below the floor. Set 0 when the goal is to
# match a real camera's viewpoint -- that needs true world coordinates, and this
# mode deliberately leaves translations alone.
ROTATE_AROUND_ROOT="${ROTATE_AROUND_ROOT:-1}"

if [ -n "$ROTATE_CALIB" ] || [ -n "$ROTATE_AXIS" ]; then
    PT_PATH="$INTERMIMIC/InterAct/$DATASET_TAG/${SEQ_NAME}.pt"
    if [ ! -f "$PT_PATH" ]; then
        echo "ERROR: rotation requested but no motion tensor at $PT_PATH" >&2
        exit 1
    fi
    if [ -n "$ROTATE_CALIB" ] && [ -n "$ROTATE_AXIS" ]; then
        echo "ERROR: set ROTATE_CALIB or ROTATE_AXIS, not both" >&2
        exit 1
    fi
    AROUND_ROOT_FLAG=""
    if [ "$ROTATE_AROUND_ROOT" = "1" ]; then AROUND_ROOT_FLAG="--around-root"; fi
    if [ -n "$ROTATE_CALIB" ]; then
        log "step 3.5: rotate_pt --from-calib $ROTATE_CALIB"
        python scripts/rotate_pt.py "$PT_PATH" \
            --from-calib "$ROTATE_CALIB" \
            $AROUND_ROOT_FLAG
    else
        log "step 3.5: rotate_pt --axis $ROTATE_AXIS --degrees $ROTATE_DEGREES"
        python scripts/rotate_pt.py "$PT_PATH" \
            --axis "$ROTATE_AXIS" --degrees "$ROTATE_DEGREES" \
            $AROUND_ROOT_FLAG
    fi
    # Report the result rather than trusting it: this reads root_rot, which is
    # what the replay actually renders, so a wrong rotation shows up here
    # instead of costing a render to discover.
    python scripts/check_pt_orientation.py "$PT_PATH" || true
fi

if [ "$REPLAY" != "1" ]; then
    log "REPLAY=0, stopping after install"
    exit 0
fi

# Step 4: headless replay, recorded to MP4. The config is not generated here on
# purpose -- dataSub and robotType must name sub<N>, and silently rewriting
# someone's env YAML is worse than failing.
if [ ! -f "$INTERMIMIC/$CFG_ENV" ]; then
    echo "ERROR: no env config at $CFG_ENV." >&2
    echo "  Copy one and point it at this subject, e.g.:" >&2
    echo "    sed 's/sub99/sub$SUBJECT_ID/g' isaacgym/src/intermimic/data/cfg/omomo_cari4d.yaml > $CFG_ENV" >&2
    exit 1
fi

export DEBUG_MOTION_LOAD=1
log "step 4/4: headless replay -> $RECORD_VIDEO"
mkdir -p "$(dirname "$RECORD_VIDEO")"
export PYTHONPATH="$INTERMIMIC/isaacgym/src:$INTERMIMIC:${PYTHONPATH:-}"

# num_envs 1, not 16: the recording camera is attached to a single env
# (RECORD_VIDEO_ENV_IDX, default 0) and each env's object mesh is fixed at
# creation, so extra envs cost simulation time and appear in no frame.
# Exported only when set: the player falls back to its own defaults on an unset
# variable but parses an empty string as a failed parse. `[ -n x ] && export`
# would be shorter, but under set -e the false branch is the last command in the
# list and would end the job.
export RECORD_VIDEO MAX_VIDEO_FRAMES
if [ -n "$RECORD_VIDEO_CAM_POS" ]; then export RECORD_VIDEO_CAM_POS; fi
if [ -n "$RECORD_VIDEO_CAM_TARGET" ]; then export RECORD_VIDEO_CAM_TARGET; fi
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --test --play_dataset --headless --num_envs 1

# The player errors rather than exiting quietly when RECORD_VIDEO is set and no
# frames were captured, so reaching here with no file means something rarer.
if [ ! -s "$RECORD_VIDEO" ]; then
    echo "ERROR: replay finished but $RECORD_VIDEO is missing or empty" >&2
    exit 1
fi
log "done"
ls -lh "$RECORD_VIDEO"

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="mesh-replay"
#SBATCH --output=mesh-replay-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Watch a simulated interaction as a body and an object rather than as capsules.
# Two steps: run the sim recording its own per-frame state, then fit an SMPL-X
# surface to that state and render it with the object mesh posed alongside.
#
#   sbatch scripts/slurm_mesh_replay.sh
#   CHECKPOINT=checkpoints/mine/mimic.pth sbatch scripts/slurm_mesh_replay.sh
#   STRIDE=2 IK_ITERS=50 sbatch scripts/slurm_mesh_replay.sh
#
# Queued because the first step needs a GPU -- isaacgym enumerates devices when
# it imports -- and because the second is slow enough to lose to a dropped ssh
# session: the IK fit runs per frame, and matplotlib draws every mesh face as
# its own polygon.

set -euo pipefail

INTERMIMIC=/simurgh2/projects/ret-hoi/InterMimic

SUBJECT_ID="${SUBJECT_ID:-100}"
OBJECT_NAME="${OBJECT_NAME:-bball}"
CLIP_IDX="${CLIP_IDX:-000}"
SEQ_NAME="sub${SUBJECT_ID}_${OBJECT_NAME}_${CLIP_IDX}"

CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball.yaml}"
CFG_TRAIN="${CFG_TRAIN:-isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml}"

# Empty replays the reference kinematically; a path runs that policy instead and
# renders what it actually did. The rest of the job is identical either way,
# which is the point -- reference and rollout come out comparable frame for
# frame because they went through the same fit and the same camera.
CHECKPOINT="${CHECKPOINT:-}"

# Body shape for the SMPL-X surface. scripts/add_subject_betas.py writes this
# from the sequence's human.npz; the OMOMO archive has no entry for a subject
# reconstructed from video.
BETAS="${BETAS:-scripts/cari4d_betas.npz}"

# SMPL-X model files, which smplx_pose.py reads as SMPLX_{MALE,FEMALE,NEUTRAL}.npz.
# Its default points at a Downloads folder on someone's laptop, so this has to be
# set for the fit to find a body at all.
SMPLX_MODELS="${SMPLX_MODELS:-/simurgh2/projects/ret-hoi/InterAct/models/smplx}"

# The object drawn alongside the body. Unset falls back to a marker.
OBJECT_MESH="${OBJECT_MESH:-$INTERMIMIC/isaacgym/src/intermimic/data/assets/objects/objects/${OBJECT_NAME}/${OBJECT_NAME}.obj}"
OBJ_FACES="${OBJ_FACES:-800}"

# Cost controls. The IK fit dominates: iterations x frames. Halve STRIDE or
# IK_ITERS first if a run is too slow to iterate on.
STRIDE="${STRIDE:-1}"
IK_ITERS="${IK_ITERS:-100}"
DUMP_FRAMES="${DUMP_FRAMES:-400}"
ELEV="${ELEV:-12}"
AZIM="${AZIM:-55}"

# CAM_POS overrides ELEV/AZIM with a real viewpoint. Feed it what
# scripts/cam_from_bundle.py prints to look from where the take was filmed,
# which is what makes the render comparable to the source footage.
CAM_POS="${CAM_POS:-}"
CAM_TARGET="${CAM_TARGET:-}"
BG="${BG:-black}"

# smplh is the model CARI4D fits, so its betas mean what they say. smplx reads
# the same numbers in a different shape basis and thickens the build.
MODEL_TYPE="${MODEL_TYPE:-smplh}"

TAG="${TAG:-$SEQ_NAME}"
RENDER_DIR="${RENDER_DIR:-$INTERMIMIC/renders}"
DUMP_NPZ="${DUMP_NPZ:-$RENDER_DIR/${TAG}_rollout.npz}"
OUT_MP4="${OUT_MP4:-$RENDER_DIR/${TAG}_mesh.mp4}"

log() { echo "[mesh $(date -u +%H:%M:%S)] $*"; }

export PYTHONUNBUFFERED=1
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MIMIC_ENV:-intermimic-gym2}"

# gym_38.so links against libpython3.8.so.1.0, which ships in the conda env
# rather than any system path. Without this every isaacgym import fails.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$INTERMIMIC"
mkdir -p "$RENDER_DIR"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=$CONDA_DEFAULT_ENV"
log "sequence=$SEQ_NAME  mode=$([ -n "$CHECKPOINT" ] && echo policy || echo reference)"
log "dump=$DUMP_NPZ"
log "out=$OUT_MP4"

export SMPLX_MODELS
if [ ! -f "$SMPLX_MODELS/SMPLX_MALE.npz" ]; then
    echo "ERROR: no SMPL-X models at $SMPLX_MODELS" >&2
    echo "  Expected SMPLX_MALE.npz there. Find them with:" >&2
    echo "    find /simurgh2 /sailhome/\$USER -name SMPLX_MALE.npz 2>/dev/null" >&2
    echo "  then re-run with SMPLX_MODELS=<that directory> sbatch ..." >&2
    exit 1
fi
log "smplx models: $SMPLX_MODELS"

if [ ! -f "$BETAS" ]; then
    echo "ERROR: no betas archive at $BETAS." >&2
    echo "  Build one from the sequence's shape:" >&2
    echo "    python scripts/add_subject_betas.py --human <InterAct>/data/<tag>/sequences_canonical/${SEQ_NAME}/human.npz --subject sub${SUBJECT_ID}" >&2
    exit 1
fi

# Step 1: run the sim, recording the humanoid's global body state per frame.
# --num_envs 1 because the dump takes env 0 only; more would simulate bodies
# nothing reads.
log "step 1/2: simulate and dump"
export PYTHONPATH="$INTERMIMIC/isaacgym/src:$INTERMIMIC:${PYTHONPATH:-}"
export DUMP_TRAJ="$DUMP_NPZ"
export DUMP_FRAMES

RUN_ARGS=(--task InterMimic --cfg_env "$CFG_ENV" --cfg_train "$CFG_TRAIN"
          --test --headless --num_envs 1)
if [ -n "$CHECKPOINT" ]; then
    RUN_ARGS+=(--checkpoint "$CHECKPOINT")
else
    RUN_ARGS+=(--play_dataset)
fi
python -u -m intermimic.run "${RUN_ARGS[@]}"

if [ ! -s "$DUMP_NPZ" ]; then
    echo "ERROR: the sim finished but $DUMP_NPZ is missing or empty" >&2
    exit 1
fi
log "dumped $(du -h "$DUMP_NPZ" | cut -f1)"

# Step 2: fit an SMPL-X surface to those body positions and render it. This is
# an IK fit rather than an inverse -- the MJCF has its own joint definitions, so
# the mesh is a faithful reading of where the sim put the bodies, not a lossless
# reconstruction of them.
log "step 2/2: IK fit and render (stride=$STRIDE, iters=$IK_ITERS)"
OBJ_ARGS=()
if [ -f "$OBJECT_MESH" ]; then
    OBJ_ARGS+=(--object "$OBJECT_MESH" --obj-faces "$OBJ_FACES")
    log "object mesh: $OBJECT_MESH"
else
    log "no object mesh at $OBJECT_MESH -- drawing a marker instead"
fi

python -u scripts/render_mesh_replay.py \
    --dump "$DUMP_NPZ" \
    --betas "$BETAS" \
    --out "$OUT_MP4" \
    --stride "$STRIDE" \
    --ik-iters "$IK_ITERS" \
    --elev "$ELEV" --azim "$AZIM" --bg "$BG" --model-type "$MODEL_TYPE" \
    ${CAM_POS:+--cam-pos "$CAM_POS"} ${CAM_TARGET:+--cam-target "$CAM_TARGET"} \
    ${OBJ_ARGS[@]+"${OBJ_ARGS[@]}"}

log "done"
ls -lh "$OUT_MP4"

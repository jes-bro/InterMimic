#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="sim-figure"
#SBATCH --output=sim-figure-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Figure-quality render of what the simulator did: nvdiffrast through the real
# camera intrinsics, composited over the source footage, with a novel view
# beside it -- the same treatment CARI4D gives its own reconstructions.
#
#   sbatch scripts/slurm_sim_figure.sh
#   CHECKPOINT=checkpoints/mine/mimic.pth TAG=policy sbatch scripts/slurm_sim_figure.sh
#   REUSE_DUMP=1 sbatch scripts/slurm_sim_figure.sh
#
# Three stages across TWO conda environments, which is why this is one job
# rather than three commands: isaacgym and nvdiffrast do not coexist, so the
# rollout is produced in one env and rendered in the other.
#
#   1. simulate, recording the humanoid's global body state   (intermimic-gym2)
#   2. fit SMPL in the camera's frame, write a prediction file (intermimic-gym2)
#   3. rasterise it with nvdiffrast                            (cari4d)

set -euo pipefail

INTERMIMIC=/simurgh2/projects/ret-hoi/InterMimic
CARI4D=/simurgh2/projects/ret-hoi/CARI4D

SUBJECT_ID="${SUBJECT_ID:-100}"
OBJECT_NAME="${OBJECT_NAME:-bball}"
CLIP_IDX="${CLIP_IDX:-000}"
SEQ_NAME="sub${SUBJECT_ID}_${OBJECT_NAME}_${CLIP_IDX}"

# The reconstruction this clip came from. It supplies the camera frame, the
# betas, and the gt/in panels the figure puts beside the prediction.
SEQ="${SEQ:-Date03_Sub01_bball_dribble}"
BUNDLE="${BUNDLE:-$CARI4D/output/opt/cari4d-release+step031397_demo-hy3d3-optv2/${SEQ}.pth}"
VIDEO="${VIDEO:-$CARI4D/sam3masks/trimmed_vids-aligned/${SEQ}.0.color.mp4}"

CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball.yaml}"
CFG_TRAIN="${CFG_TRAIN:-isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml}"

# Empty replays the reference kinematically; a path runs that policy. Both go
# through the identical rest of the job, so the two figures are comparable.
CHECKPOINT="${CHECKPOINT:-}"

BETAS="${BETAS:-scripts/cari4d_betas.npz}"
SMPLX_MODELS="${SMPLX_MODELS:-/simurgh2/projects/ret-hoi/InterAct/models/smplx}"
MODEL_TYPE="${MODEL_TYPE:-smplh}"
IK_ITERS="${IK_ITERS:-100}"
DUMP_FRAMES="${DUMP_FRAMES:-400}"

GYM_ENV="${GYM_ENV:-intermimic-gym2}"
CARI4D_ENV="${CARI4D_ENV:-newcari4d}"

TAG="${TAG:-reference}"
DUMP_NPZ="${DUMP_NPZ:-$INTERMIMIC/renders/${SEQ_NAME}_${TAG}_rollout.npz}"
PRED_PTH="${PRED_PTH:-$CARI4D/output/sim/${TAG}/${SEQ}.pth}"
OUT_ROOT="${OUT_ROOT:-$CARI4D/output/viz-sim/${TAG}}"

# Skip the simulation and reuse an existing rollout. Useful when only the render
# is being iterated on -- stage 1 is the only part that needs the GPU twice.
REUSE_DUMP="${REUSE_DUMP:-0}"

log() { echo "[fig $(date -u +%H:%M:%S)] $*"; }

export PYTHONUNBUFFERED=1
source "$(conda info --base)/etc/profile.d/conda.sh"

# Saved so the second environment does not inherit the first's linker and module
# paths. isaacgym needs its conda lib on LD_LIBRARY_PATH; carrying that into the
# cari4d env is how one env quietly loads the other's libraries.
ORIG_LD="${LD_LIBRARY_PATH:-}"
ORIG_PYTHONPATH="${PYTHONPATH:-}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "sequence=$SEQ_NAME  tag=$TAG  mode=$([ -n "$CHECKPOINT" ] && echo policy || echo reference)"
log "bundle=$BUNDLE"
log "video=$VIDEO"

for required in "$BUNDLE" "$VIDEO" "$SMPLX_MODELS/SMPLX_MALE.npz"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- stage 1 + 2
conda activate "$GYM_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ORIG_LD"
export SMPLX_MODELS
cd "$INTERMIMIC"
mkdir -p "$(dirname "$DUMP_NPZ")" "$(dirname "$PRED_PTH")"

if [ "$REUSE_DUMP" = "1" ] && [ -s "$DUMP_NPZ" ]; then
    log "stage 1/3: reusing $DUMP_NPZ"
else
    log "stage 1/3: simulate and dump ($GYM_ENV)"
    export PYTHONPATH="$INTERMIMIC/isaacgym/src:$INTERMIMIC:$ORIG_PYTHONPATH"
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
    unset DUMP_TRAJ
fi

if [ ! -s "$DUMP_NPZ" ]; then
    echo "ERROR: no rollout at $DUMP_NPZ" >&2
    exit 1
fi

log "stage 2/3: fit in the camera's frame and write a prediction file"
python -u scripts/sim_to_cari4d_bundle.py \
    --dump "$DUMP_NPZ" \
    --bundle "$BUNDLE" \
    --pt "$INTERMIMIC/InterAct/behave_cari4d/${SEQ_NAME}.pt" \
    --betas "$BETAS" \
    --models "$SMPLX_MODELS" \
    --model-type "$MODEL_TYPE" \
    --ik-iters "$IK_ITERS" \
    --out "$PRED_PTH"

if [ ! -s "$PRED_PTH" ]; then
    echo "ERROR: no prediction file at $PRED_PTH" >&2
    exit 1
fi

# ---------------------------------------------------------------- stage 3
conda deactivate
conda activate "$CARI4D_ENV"
# Restored, not extended: the render must not see the gym env's libraries.
export LD_LIBRARY_PATH="$ORIG_LD"
export PYTHONPATH="$ORIG_PYTHONPATH"
cd "$CARI4D"

log "stage 3/3: nvdiffrast render ($CARI4D_ENV) -> $OUT_ROOT"
mkdir -p "$OUT_ROOT"
python -u tools/viz_pred.py \
    -pf "$PRED_PTH" \
    --wild_video --kid 0 \
    --video "$VIDEO" \
    --out_root "$OUT_ROOT"

log "done"
find "$OUT_ROOT" -name "*.mp4" -newermt "-3 hours" -exec ls -lh {} \;

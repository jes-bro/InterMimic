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
# smoke-test the replay. The four steps documented at the top of
# isaacgym/scripts/data_replay_cari4d.sh, run as one queued job.
#
#   sbatch scripts/slurm_cari4d_to_mimic.sh
#   SUBJECT_ID=101 OBJECT_NAME=chair sbatch scripts/slurm_cari4d_to_mimic.sh
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

# Env config for the replay smoke test. Written by hand, not generated here.
CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball.yaml}"

# Set REPLAY=0 to stop after installing. The replay runs headless, so it proves
# the motion loads and the MJCF is valid but shows nothing -- watching it needs
# a display, which a batch job does not have.
REPLAY="${REPLAY:-1}"

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

# Step 2: InterAct -> InterMimic. --mesh builds per-bone convex hulls from the
# subject's own shape, so the MJCF matches this body rather than a mean one.
log "step 2/4: run_interact2mimic (--mesh)"
python scripts/run_interact2mimic.py \
    --interact-root "$INTERACT" \
    --dataset-name "$DATASET_TAG" \
    --mesh

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

if [ "$REPLAY" != "1" ]; then
    log "REPLAY=0, stopping after install"
    exit 0
fi

# Step 4: headless replay. Proves the clip loads and the MJCF is valid. The
# config is not generated here on purpose -- dataSub and robotType must name
# sub<N>, and silently rewriting someone's env YAML is worse than failing.
if [ ! -f "$INTERMIMIC/$CFG_ENV" ]; then
    echo "ERROR: no env config at $CFG_ENV." >&2
    echo "  Copy one and point it at this subject, e.g.:" >&2
    echo "    sed 's/sub99/sub$SUBJECT_ID/g' isaacgym/src/intermimic/data/cfg/omomo_cari4d.yaml > $CFG_ENV" >&2
    exit 1
fi

log "step 4/4: headless replay smoke test ($CFG_ENV)"
export PYTHONPATH="$INTERMIMIC/isaacgym/src:$INTERMIMIC:${PYTHONPATH:-}"
python -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --test --play_dataset --headless --num_envs 16

log "done"

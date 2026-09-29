#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="replay-sweep"
#SBATCH --output=replay-sweep-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# ONE job that replays a pinned clip through many bodies in sequence, with the
# REPLAY_TRACE instrumentation on, so a whole roster can be screened for the
# joint-angle rollover trap (sub4, 2026-09-28) in a single allocation instead
# of one sbatch per body. Each body takes ~2-3 min; 47 bodies ~ 2 h.
#
#   BODIES="sub1 sub2 ..." sbatch --exclude=simurgh6,simurgh2 slurm_replay_sweep.sh
#   ASSET=inertial BODIES="..." sbatch ...   # replay smplx_omomo_<body>_inertial.xml instead
#
# Defaults: BODIES = the 17 OMOMO reals + the 30 gen4 bodies; CLIP =
# sub2_largetable_005; SOURCE = sub2; traces -> renders/sweep/replay_trace_<body>_<asset>.npz.
# Read with: python3 scripts/summarize_replay_traces.py renders/sweep/replay_trace_*_<asset>.npz
# A body whose replay fails is logged and the sweep continues.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

BODIES="${BODIES:-sub1 sub2 sub3 sub4 sub5 sub6 sub7 sub8 sub9 sub10 sub11 sub12 sub13 sub14 sub15 sub16 sub17 $(seq -f sub%g 600 629)}"
CLIP="${CLIP:-sub2_largetable_005}"
SOURCE="${SOURCE:-sub2}"
ASSET="${ASSET:-stock}"           # stock | inertial
BASE="${REPLAY_BASE_CFG:-isaacgym/src/intermimic/data/cfg/omomo_replay_v1_allbodies.yaml}"   # betas for all 47 bodies
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

SRC_CLIP="$(pwd)/InterAct/OMOMO_new/${CLIP}.pt"
[ -f "$SRC_CLIP" ] || { echo "[sweep] ERROR: clip not found: $SRC_CLIP" >&2; exit 1; }
CLIPDIR="/tmp/replay_sweep_clip_${CLIP}_$$"
mkdir -p "$CLIPDIR" renders/sweep; ln -sf "$SRC_CLIP" "$CLIPDIR/${CLIP}.pt"

echo "[sweep] asset=$ASSET clip=$CLIP source=$SOURCE bodies: $BODIES"
ok=0; failed=""
for BODY in $BODIES; do
    OUT="renders/sweep/replay_trace_${BODY}_${ASSET}.npz"
    if [ "$ASSET" = inertial ]; then
        F="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${BODY}_inertial.xml"
        [ -f "$F" ] || { echo "[sweep] $BODY: MISSING $F (run scripts/mjcf_add_inertials.py --all)"; failed="$failed $BODY"; continue; }
        export REPLAY_ASSET_FILE="smplx/smplx_omomo_${BODY}_inertial.xml"
    else
        unset REPLAY_ASSET_FILE
    fi
    echo "[sweep] === $BODY ($ASSET) -> $OUT  $(date +%H:%M:%S)"
    # DUMP_TRAJ makes the player stop after ONE pass over the clip (without a
    # video or dump request it replays forever); the dump is a useful by-product.
    if REPLAY_TRACE=1 REPLAY_TRACE_OUT="$OUT" DUMP_TRAJ="renders/sweep/dump_${BODY}_${ASSET}.npz" \
        python -u -m intermimic.run --task InterMimic \
            --cfg_env "$BASE" --cfg_train "$TRAIN" \
            --subject_bodies "$BODY" --data_sub "$SOURCE" --data_objects all --motion_file "$CLIPDIR" \
            --test --play_dataset --headless --num_envs 1 > "renders/sweep/log_${BODY}_${ASSET}.txt" 2>&1 \
       && [ -f "$OUT" ]; then
        ok=$((ok+1)); tail -1 "renders/sweep/log_${BODY}_${ASSET}.txt"
    else
        echo "[sweep] $BODY FAILED (see renders/sweep/log_${BODY}_${ASSET}.txt)"; failed="$failed $BODY"
    fi
done
rm -rf "$CLIPDIR"
echo "[sweep] done: $ok traces written; failed:${failed:- none}"
echo "[sweep] summarize with: python3 scripts/summarize_replay_traces.py renders/sweep/replay_trace_*_${ASSET}.npz"

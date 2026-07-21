#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="replay-xb"
#SBATCH --output=replay-xb-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# DECOUPLED kinematic replay (NO policy): play SOURCE's ground-truth motion on
# BODY's MJCF. play_dataset poses via FK from root_rot + dof_pos (_set_env_state
# -> the sim places bodies through BODY's MJCF), so the BODY/MJCF genuinely
# matters. This isolates a broken BODY from broken MOTION -- which slurm_replay.sh
# (which forces subjectBodies == dataSub) cannot do:
#
#   BODY=sub2 SOURCE=sub4 sbatch slurm_replay_xbody.sh   # sub4 MOTION on a good body -> broken? = MOTION
#   BODY=sub4 SOURCE=sub2 sbatch slurm_replay_xbody.sh   # good motion on sub4 BODY -> broken? = BODY
#
# By default ALL of SOURCE's matching clips load and each env samples one at
# reset, so two runs differ. To compare bodies on the IDENTICAL motion, pin one
# clip with CLIP (exact filename, no extension):
#   CLIP=sub2_largetable_005 BODY=sub16 SOURCE=sub2 sbatch slurm_replay_xbody.sh
#   CLIP=sub2_largetable_005 BODY=sub10 SOURCE=sub2 sbatch slurm_replay_xbody.sh
# (CLIP must start with SOURCE's subN so the dataSub filter keeps it.)
#
# Same skeleton topology across subjects (153 dof), so source dof drives any
# body's MJCF dimensionally fine. Video -> renders/replayxb_<body>_<source>.mp4

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

BODY="${BODY:-sub2}"           # MJCF loaded in sim (subjectBodies)
SOURCE="${SOURCE:-sub4}"       # ground-truth motion played (dataSub)
OBJECT="${OBJECT:-}"           # empty = all objects
CLIP="${CLIP:-}"               # empty = all clips; set e.g. sub2_largetable_005 to pin ONE
FRAMES="${FRAMES:-300}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

OBJLINE="dataObjects: []"
[ -n "$OBJECT" ] && OBJLINE="dataObjects: ['$OBJECT']"
TAG="${BODY}_${SOURCE}${OBJECT:+_$OBJECT}${CLIP:+_${CLIP##*_}}"
scontrol update JobId="$SLURM_JOB_ID" JobName="rxb-$TAG" 2>/dev/null || true

# CLIP pin: load exactly one clip so every body sees the IDENTICAL motion. Build a
# temp motion dir holding only that clip and override motion_file to point at it.
MOTLINE=""
if [ -n "$CLIP" ]; then
    case "$CLIP" in
        "${SOURCE}"_*) : ;;   # must belong to SOURCE or the dataSub filter drops it
        *) echo "[replayxb] ERROR: CLIP=$CLIP does not start with SOURCE=$SOURCE" >&2; exit 1 ;;
    esac
    SRC_CLIP="$(pwd)/InterAct/OMOMO_new/${CLIP}.pt"
    [ -f "$SRC_CLIP" ] || { echo "[replayxb] ERROR: clip not found: $SRC_CLIP" >&2; exit 1; }
    CLIPDIR="/tmp/replayxb_clip_${CLIP}"
    mkdir -p "$CLIPDIR"; ln -sf "$SRC_CLIP" "$CLIPDIR/${CLIP}.pt"
    MOTLINE="; s|motion_file:.*|motion_file: $CLIPDIR|"
    echo "[replayxb] CLIP pinned: only ${CLIP}.pt (motion_file -> $CLIPDIR)"
fi

mkdir -p renders
CFG="/tmp/replayxb_$TAG.yaml"
# play SOURCE's motion (dataSub) on BODY's MJCF (subjectBodies)
sed "s|dataSub:.*|dataSub: ['$SOURCE']|; s|subjectBodies:.*|subjectBodies: ['$BODY']|; s|dataObjects:.*|$OBJLINE|$MOTLINE" \
    "$BASE" > "$CFG"

echo "[replayxb] body=$BODY  source=$SOURCE  object=${OBJECT:-ALL}  -> renders/replayxb_$TAG.mp4"
RECORD_VIDEO="renders/replayxb_$TAG.mp4" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG" --cfg_train "$TRAIN" \
        --test --play_dataset --headless --num_envs 1

echo
echo "[replayxb] done:"
ls -lh "renders/replayxb_$TAG.mp4"

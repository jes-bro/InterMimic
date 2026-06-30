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
FRAMES="${FRAMES:-300}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

OBJLINE="dataObjects: []"
[ -n "$OBJECT" ] && OBJLINE="dataObjects: ['$OBJECT']"
TAG="${BODY}_${SOURCE}${OBJECT:+_$OBJECT}"
scontrol update JobId="$SLURM_JOB_ID" JobName="rxb-$TAG" 2>/dev/null || true

mkdir -p renders
CFG="/tmp/replayxb_$TAG.yaml"
# play SOURCE's motion (dataSub) on BODY's MJCF (subjectBodies), all objects
sed "s|dataSub:.*|dataSub: ['$SOURCE']|; s|subjectBodies:.*|subjectBodies: ['$BODY']|; s|dataObjects:.*|$OBJLINE|" \
    "$BASE" > "$CFG"

echo "[replayxb] body=$BODY  source=$SOURCE  object=${OBJECT:-ALL}  -> renders/replayxb_$TAG.mp4"
RECORD_VIDEO="renders/replayxb_$TAG.mp4" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG" --cfg_train "$TRAIN" \
        --test --play_dataset --headless --num_envs 1

echo
echo "[replayxb] done:"
ls -lh "renders/replayxb_$TAG.mp4"

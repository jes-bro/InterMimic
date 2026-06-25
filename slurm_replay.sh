#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="replay"
#SBATCH --output=replay-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Kinematic replay (NO policy) of subjects' GROUND-TRUTH motion on their OWN
# bodies, recorded to mp4 via the RECORD_VIDEO infra (headless, no display).
# Use it to judge whether a subject's reference data / MJCF is broken (body
# inverted, penetrating, garbage) vs. fine (then the failure is control/method,
# not data -- so you can't drop it).
#
#   SUBJECTS="sub4 sub6" sbatch slurm_replay.sh
#   FRAMES=600 SUBJECTS="sub10" sbatch slurm_replay.sh
# Videos land in renders/replay_<sub>.mp4 (persistent; scp them to watch).

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SUBJECTS="${SUBJECTS:-sub4 sub6}"
FRAMES="${FRAMES:-300}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

mkdir -p renders
for S in $SUBJECTS; do
    CFG="/tmp/replay_$S.yaml"
    # patch the multibody test config to this subject (motion + body), all objects
    sed "s|dataSub:.*|dataSub: ['$S']|; s|subjectBodies:.*|subjectBodies: ['$S']|; s|dataObjects:.*|dataObjects: []|" \
        "$BASE" > "$CFG"
    echo "[replay] $S -> renders/replay_$S.mp4 ($FRAMES frames, kinematic)"
    RECORD_VIDEO="renders/replay_$S.mp4" MAX_VIDEO_FRAMES="$FRAMES" \
        python -u -m intermimic.run --task InterMimic \
            --cfg_env "$CFG" --cfg_train "$TRAIN" \
            --test --play_dataset --headless --num_envs 1
done

echo
echo "[replay] done. Videos:"
ls -lh renders/replay_*.mp4

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="render-pol"
#SBATCH --output=render-pol-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Render ONE policy rollout (a trained curriculum/baseline checkpoint DRIVING the
# humanoid) to mp4 -- the policy analog of slurm_replay.sh (which is kinematic,
# no policy). Reuses the SAME configs the eval uses (omomo_test_multibody.yaml +
# omomo_multibody.yaml), which already load these MLP curriculum checkpoints
# (numObs 3230, plain betas). Start-mode init, 1 env.
#
#   # finished baseline (default), held-out body sub10 driven by sub2 motion:
#   sbatch slurm_render_policy.sh
#   # the (undertrained) bn_mlp checkpoint instead:
#   CHECKPOINT=checkpoints/smplx_curriculum_ist_flong_bn_mlp_s10b_sub6_source/nn/mimic_00009200.pth sbatch slurm_render_policy.sh
#   # pick body / source / object / length:
#   BODY=sub2 SOURCE=sub2 OBJECT=largetable FRAMES=600 sbatch slurm_render_policy.sh
# Video -> renders/policy_<body>_<source>[_<object>].mp4 (persistent; scp to watch).

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Default = the FINISHED baseline_flong policy (sfinal). Override CHECKPOINT for bn_mlp etc.
CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_curriculum_baseline_flong_sfinal_sub14_final/nn/mimic.pth}"
BODY="${BODY:-sub10}"          # body in sim (subjectBodies)
SOURCE="${SOURCE:-sub2}"       # reference motion (dataSub)
OBJECT="${OBJECT:-}"           # empty = all objects
FRAMES="${FRAMES:-300}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

[ -f "$CHECKPOINT" ] || { echo "[render] ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

OBJLINE="dataObjects: []"
[ -n "$OBJECT" ] && OBJLINE="dataObjects: ['$OBJECT']"
TAG="${BODY}_${SOURCE}${OBJECT:+_$OBJECT}"
scontrol update JobId="$SLURM_JOB_ID" JobName="rp-$TAG" 2>/dev/null || true

mkdir -p renders
CFG="/tmp/render_policy_$TAG.yaml"
# patch the multibody test config to this (body, source, object)
sed "s|dataSub:.*|dataSub: ['$SOURCE']|; s|subjectBodies:.*|subjectBodies: ['$BODY']|; s|dataObjects:.*|$OBJLINE|" \
    "$BASE" > "$CFG"

echo "[render] ckpt=$CHECKPOINT"
echo "[render] body=$BODY source=$SOURCE object=${OBJECT:-ALL} frames=$FRAMES -> renders/policy_$TAG.mp4"
RECORD_VIDEO="renders/policy_$TAG.mp4" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG" --cfg_train "$TRAIN" \
        --test --checkpoint "$CHECKPOINT" --headless --num_envs 1

echo
echo "[render] done:"
ls -lh "renders/policy_$TAG.mp4"

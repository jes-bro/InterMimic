#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src9"
#SBATCH --output=teacher-src9-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# SOURCE-TEACHER validation #2: source=sub9 (FEMALE) driving 13 train bodies,
# body-conditioned (neutral betas), NO staging. Sibling of slurm_teacher_src2.sh
# with only the source changed -> directly comparable. Stresses the cross-gender
# case (female motion onto male+female bodies) the neutral betas should fix.
# Same question: does mean_rewards climb and converge WITHOUT staging?
#
# Runs from repo root. Saves to checkpoints/smplx_teacher_src9_neutral/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src9.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src9.yaml

echo "[teacher] source=sub9 x 13 train bodies (sub13 held out), neutral betas + body-norm + pose, no staging"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src9_neutral/nn/"

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

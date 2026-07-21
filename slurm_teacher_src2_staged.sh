#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_staged"
#SBATCH --output=teacher-src2_staged-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# STAGED source-teacher (branch source-teacher-staged): fixed source sub2 drives
# 13 real + 27 inhull synthetic bodies from stage 0, then folds the 12 EXTRAPOLATED
# synthetic bodies in ONE AT A TIME (13 stages), resuming each stage. A/B this
# against the no-stage checkpoints/smplx_teacher_src2_xf_aug run: same body set,
# same everything, the ONLY difference is the fold-in schedule (weight mask).
#
# One slurm job runs all 13 stages sequentially. --resume makes a requeue continue
# from the first unfinished stage (a full curriculum likely exceeds one window).
# Final policy = checkpoints/smplx_teacher_src2_staged_s12/nn/  (eval that one).

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

echo "[staged] host=$(hostname) job=$SLURM_JOB_ID source=sub2"
python -u scripts/staged_source_teacher_runner.py --source 2 --run-name src2_staged --resume

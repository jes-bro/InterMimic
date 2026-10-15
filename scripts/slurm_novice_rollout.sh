#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --job-name="novice-rollout"
#SBATCH --output=novice-rollout-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
# GPU wrapper for scripts/novice_rollout.py: every argument is passed through.
#
#   sbatch --exclude=simurgh6,simurgh2 scripts/slurm_novice_rollout.sh \
#       --clip InterAct/<tag>/sub204_bball_000.pt --bundle <seq>.pth --object-mesh <seq>_align.obj \
#       --arm student_g3_act_xf_ret_nvadlr__f0 --checkpoint checkpoints/.../mimic_00100000.pth \
#       --gender male --models /simurgh2/projects/ret-hoi/InterAct/models/smplx --out renders/novice_<seq>
#
# Another machine: INTERMIMIC_ENV=<its env>, and edit the #SBATCH header for its cluster.
# Run the same python line with --dry-run on a login node first: it validates every
# input and writes the derived cfg without touching the GPU.
source ~/.bashrc
conda deactivate
conda activate "${INTERMIMIC_ENV:-intermimic-gym2}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 2
echo "[novice-rollout] host=$(hostname) job=${SLURM_JOB_ID:-none} args: $*"
python3 -u scripts/novice_rollout.py "$@"

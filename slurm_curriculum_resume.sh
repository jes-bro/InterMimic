#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="c-resume"
#SBATCH --output=curriculum-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Generic curriculum RESUME wrapper. Runs curriculum_runner.py with the exact
# args passed in the CURRICULUM_ARGS env var -- which resubmit_curriculum.sh
# recovers verbatim from each run's '[curriculum] invocation:' log line. This
# avoids hand-translating flags into slurm_curriculum.sh's env-var knobs, so a
# resubmit reproduces the ORIGINAL run's config exactly and --resume continues
# it from curriculum_work/<run>/state.json.
#
# Submit like:
#   export CURRICULUM_ARGS="--run-name ist_flong --schedule ... --resume"
#   sbatch --export=ALL slurm_curriculum_resume.sh
set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

: "${CURRICULUM_ARGS:?CURRICULUM_ARGS not set -- use resubmit_curriculum.sh or export it before sbatch}"

# Rename the job so squeue shows which run this is.
run=$(printf '%s\n' "$CURRICULUM_ARGS" | grep -oE -- '--run-name [^ ]+' | awk '{print $2}')
scontrol update JobId="$SLURM_JOB_ID" JobName="c-${run:-resume}" 2>/dev/null || true

echo "[curriculum] resume invocation: scripts/curriculum_runner.py $CURRICULUM_ARGS"
python scripts/curriculum_runner.py $CURRICULUM_ARGS

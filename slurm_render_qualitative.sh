#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --job-name="render-qual"
#SBATCH --output=render-qual-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Run a render_qualitative_*.sh on a GPU COMPUTE node (never on the sc login
# node). The render scripts are plain bash; this wrapper just provides the
# slurm allocation + conda env + a virtual display (IsaacGym camera rendering
# needs one on a headless node).
#
# Usage (from repo root):
#   sbatch slurm_render_qualitative.sh                                  # defaults to sub4
#   sbatch slurm_render_qualitative.sh scripts/render_qualitative_sub16.sh
#
# Output videos land in /tmp/ on the COMPUTE node (per render_qualitative.sh).
# /tmp is node-local, so copy them somewhere shared at the end if you want them
# off the node -- see the trailing cp.

set -u
RENDER_SCRIPT="${1:-scripts/render_qualitative_sub4.sh}"

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

echo "[render] script=$RENDER_SCRIPT host=$(hostname) job=$SLURM_JOB_ID"
[ -f "$RENDER_SCRIPT" ] || { echo "[render] ERROR: $RENDER_SCRIPT not found (run from repo root)"; exit 1; }

# Headless IsaacGym camera capture needs a virtual X display; wrap in xvfb if present.
if command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a bash "$RENDER_SCRIPT"
else
    echo "[render] WARNING: xvfb-run not found; running without a virtual display (may fail to capture)"
    bash "$RENDER_SCRIPT"
fi

# Copy node-local /tmp videos to a shared results dir so they survive past the job.
DEST="render_results/${SLURM_JOB_ID}"
mkdir -p "$DEST"
cp -v /tmp/render_*.mp4 "$DEST"/ 2>/dev/null || echo "[render] no /tmp/render_*.mp4 to copy"
echo "[render] done; videos -> $DEST"

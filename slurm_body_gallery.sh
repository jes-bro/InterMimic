#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="body-gallery"
#SBATCH --output=body-gallery-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=END,FAIL

# Render every per-subject body side by side in ONE Isaac Gym env.
#   sbatch slurm_body_gallery.sh                                   # capsule MJCFs
#   MESH=1 sbatch slurm_body_gallery.sh                            # SMPL-X surfaces
#   MESH=1 SUBJECTS="sub4 sub10 sub13 sub16" OUT=g.png sbatch slurm_body_gallery.sh
# --mesh needs the SMPL-X models: set SMPLX_MODELS=/path/to/models/smplx (must hold
# SMPLX_MALE.npz / SMPLX_FEMALE.npz). Output PNG lands in the repo root (or OUT).

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SUBJECTS="${SUBJECTS:-}" \
  python3 -u scripts/render_body_gallery.py \
    ${MESH:+--mesh} \
    ${SUBJECTS:+--subjects $SUBJECTS} \
    --out "${OUT:-body_gallery.png}"

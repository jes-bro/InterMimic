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
# --mesh gets its surfaces from EITHER a prebaked npz (no models needed) or the
# SMPL-X models:
#   MESH=1 MESH_NPZ=/path/omomo_smplx_meshes.npz sbatch slurm_body_gallery.sh   # baked
#   MESH=1 SMPLX_MODELS=/path/models/smplx        sbatch slurm_body_gallery.sh   # from models
# The baked npz is produced on a machine WITH the models (scripts/smplx_mesh.py)
# and copied over -- use it since the cluster has no SMPL-X models.
# Output PNG lands in the repo root (or OUT).

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

python3 -u scripts/render_body_gallery.py \
    ${MESH:+--mesh} \
    ${MESH_NPZ:+--mesh-npz "$MESH_NPZ"} \
    ${SUBJECTS:+--subjects $SUBJECTS} \
    --out "${OUT:-body_gallery.png}"

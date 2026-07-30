#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="distill-src-no1no14"
#SBATCH --output=distill-source-no1no14-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# SOURCE-TEACHER DISTILLATION, variant 'no1no14': the in-distribution source set
# (as 'noheldout') with BOTH sub1 and sub14 dropped -> 11 sources
# {2,3,5,6,7,8,9,11,12,15,17}. Same 52-body target set as the other variants, so
# it is a clean single-knob comparison against 'noheldout' for what those two
# sources contribute.
#
# One body-conditioned transformer student imitates the per-source transformer
# teachers (InterMimic_All selects a teacher per env by source subid). This is the
# FIRST distill from TRANSFORMER teachers -- watch startup for obs-size / vmap
# errors (see generate_source_distill_cfgs.py header). Saves to
# checkpoints/smplx_student_source_xf_aug_no1no14/nn/.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# 1) gather the trained source-teacher checkpoints into the teacherPolicy dir
#    (fails loudly if any source teacher hasn't produced a checkpoint yet)
python -u scripts/collect_source_teachers.py --sources 2 3 5 6 7 8 9 11 12 15 17 \
    --out checkpoints/teachers/source_xf_aug_no1no14

# 2) distill
echo "[distill] no1no14 host=$(hostname) job=$SLURM_JOB_ID"
python -u -m intermimic.run_distill \
    --task InterMimic_All \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_all_source_xf_aug_no1no14.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all_source_xf_aug_no1no14.yaml \
    --headless \
    --output checkpoints

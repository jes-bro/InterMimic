#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

#SBATCH --job-name="retarget-shift"
#SBATCH --output=retarget-shift-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=FAIL

# NO --gres=gpu ON PURPOSE. This only torch.loads two .pt files and compares a
# column; asking for a GPU would put it behind every training job in the queue
# for work that takes seconds. A CPU-only job schedules almost immediately.
#
# Runs scripts/check_retarget_shift.py in the conda env -- the login node's
# python3 has no torch, which is why running it directly fails.
#
# USAGE (from repo root):
#   sbatch slurm_check_retarget_shift.sh
#   CLIP=sub2_largetable_000.pt sbatch slurm_check_retarget_shift.sh
#   BODIES="sub16 sub10 sub13 sub2" CLIP=... sbatch slurm_check_retarget_shift.sh
#
# Env:
#   CLIP          clip filename (default sub2_largetable_017.pt, the 260-frame one)
#   BODIES        space-separated bodies to check (default sub2 sub13 sub16 sub10)
#   MOTION_DIR    source clips     (default InterAct/OMOMO_new)
#   RETARGET_DIR  retargeted refs  (default InterAct/OMOMO_retarget_contact_src2)

set -u
# sbatch runs a COPY of this script from the job spool dir, so dirname "$0" is
# not the repo; slurm already sets cwd to the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}" || exit 2
if [ ! -f scripts/check_retarget_shift.py ]; then
    echo "ERROR: cwd $(pwd) is not the InterMimic repo root." >&2
    echo "       Submit from the repo root: cd <repo> && sbatch $(basename "$0")" >&2
    exit 2
fi

CLIP="${CLIP:-sub2_largetable_017.pt}"
BODIES="${BODIES:-sub2 sub13 sub16 sub10}"
MOTION_DIR="${MOTION_DIR:-InterAct/OMOMO_new}"
RETARGET_DIR="${RETARGET_DIR:-InterAct/OMOMO_retarget_contact_src2}"

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

echo "[shift] host=$(hostname) job=${SLURM_JOB_ID:-<none>}"
echo "[shift] clip=$CLIP  bodies=($BODIES)"
echo "[shift] motion=$MOTION_DIR  retarget=$RETARGET_DIR"
echo

python3 scripts/check_retarget_shift.py \
    --clip "$CLIP" \
    --bodies $BODIES \
    --motion-dir "$MOTION_DIR" \
    --retarget-dir "$RETARGET_DIR"
rc=$?

echo
echo "[shift] exit=$rc"
exit $rc

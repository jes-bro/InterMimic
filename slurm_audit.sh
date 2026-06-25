#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --job-name="audit"
#SBATCH --output=audit-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Data-quality audit of the motion clips -- NO GPU, NO sim. Flags per SUBJECT and
# per OBJECT: nonfinite / explosion / joint-limit / teleport / no-interaction, so
# a struggling subject (e.g. sub4) can be diagnosed as bad DATA vs a control/method
# failure -- the complement to the kinematic replay. Run from the repo root.
#
#   sbatch slurm_audit.sh                                 # ALL subjects (best -- outliers stand out vs the median)
#   SUBJECTS="sub4 sub10 sub16" sbatch slurm_audit.sh     # just these
#   MOTION_DIR=InterAct/OMOMO sbatch slurm_audit.sh       # a different dataset

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
# audit imports only torch (CPU) -- no isaacgym -- so no LD_LIBRARY_PATH needed.

MOTION_DIR="${MOTION_DIR:-InterAct/OMOMO_new}"
ARGS="--motion-dir $MOTION_DIR"
[ -n "${SUBJECTS:-}" ] && ARGS="$ARGS --subjects $SUBJECTS"

echo "========== self-test: prove the detector in this env =========="
python scripts/audit_motion_data.py --selftest

echo
echo "========== audit: motion-dir=$MOTION_DIR subjects=${SUBJECTS:-ALL} =========="
mkdir -p audit_results
OUT="audit_results/audit_${SLURM_JOB_ID:-local}.txt"
python scripts/audit_motion_data.py $ARGS 2>&1 | tee "$OUT"
echo
echo "[audit] full report saved to $OUT"

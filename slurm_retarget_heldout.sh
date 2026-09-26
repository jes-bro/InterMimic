#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --exclude=simurgh6,simurgh2

#SBATCH --job-name="rt-heldout"
#SBATCH --output=rt-heldout-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Cluster wrapper for scripts/retarget_heldout_bodies.sh: add bodies to an
# EXISTING reference tree, on a compute node.
#
# WHY. The BEHAVE bodies' references were built on the login node and stopped
# part way -- sub320 had 1444 of 3356 clips, sub327 had 1178, with nothing left
# running. A long CPU job belongs on a compute node with a walltime, not on a
# login shell that a dropped ssh kills silently.
#
# RESUMABLE: finished (body, clip) pairs are skipped, so re-running after an
# interruption costs a directory scan, not the solves. That is what makes this
# safe to fire at a half-built tree.
#
#   BODIES="sub320 sub321 sub322 sub323 sub324 sub325 sub326 sub327" \
#       sbatch slurm_retarget_heldout.sh
#
#   # activity sources instead of the OMOMO 13 (a different tree AND motion dir):
#   BODIES="sub320 ..." TREE=InterAct/behave_cari4d_act_f0_bodymajor \
#       MOTION_DIR=InterAct/behave_cari4d \
#       SOURCES="sub401 sub402 ..." sbatch slurm_retarget_heldout.sh
#
# Every env var of the underlying script passes through (BODIES, SOURCES, TREE,
# MOTION_DIR, ITERS, ALLOW_WORSE_CM). WORKERS defaults to this job's CPU
# allocation rather than the node's core count -- asking for 128 workers inside a
# 64-CPU allocation just makes them fight.
#
# --exclude=simurgh6,simurgh2: uncorrectable-ECC GPUs. This job is CPU-only, but
# the exclusion is on every sbatch here so nothing lands there by habit.
set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
# No Isaac Gym and no GPU: this is torch on CPU, so no LD_LIBRARY_PATH dance.

: "${BODIES:?set BODIES, e.g. BODIES=\"sub320 sub321\"}"
export WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

echo "[rt-heldout] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[rt-heldout] bodies : $BODIES"
echo "[rt-heldout] tree   : ${TREE:-InterAct/OMOMO_retarget_contact_srcall13}"
echo "[rt-heldout] workers: $WORKERS"
echo

sh scripts/retarget_heldout_bodies.sh
rc=$?

echo
echo "[rt-heldout] exit $rc. Clip counts per body (compare with a finished body):"
TREE_D="${TREE:-InterAct/OMOMO_retarget_contact_srcall13}"
for b in sub2 $BODIES; do
    [ -d "$TREE_D/$b" ] && echo "    $b: $(ls "$TREE_D/$b" | wc -l)"
done
exit $rc

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="bball7-retarget"
#SBATCH --output=bball7-retarget-%A_%a.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Retarget every bball7 clip of ONE subject onto every f0 body (contact-aware
# solve, CPU only, no Isaac Gym). One array task per subject:
#
#   sbatch --array=0-6 scripts/slurm_cari4d_bball7_retarget.sh
#
# Writes <OUT_PREFIX>_src<id>/<body>/<clip>.pt (body-major, what the task reads)
# for the 43 TRAINING bodies (from the arm's cfg) AND the 3 held-out bodies
# sub10 sub13 sub16 -- the eval scores every source on the held-out bodies and
# the task refuses a missing (body, clip) reference, so without them the
# documented eval step cannot run.
# When all seven have PASSED, merge them into one tree for the arm (no
# --bodies-from: the roster is every body common to all seven = 46):
#
#   python3 scripts/merge_retarget_trees.py \
#       --sources InterAct/behave_cari4d_bball7_retarget_src401 \
#                 InterAct/behave_cari4d_bball7_retarget_src402 \
#                 InterAct/behave_cari4d_bball7_retarget_src404 \
#                 InterAct/behave_cari4d_bball7_retarget_src409 \
#                 InterAct/behave_cari4d_bball7_retarget_src411 \
#                 InterAct/behave_cari4d_bball7_retarget_src412 \
#                 InterAct/behave_cari4d_bball7_retarget_src458 \
#       --out InterAct/behave_cari4d_bball7_f0_bodymajor
#   (expect: 46 body dirs, 46 x 52 = 2392 links; check_retarget_coverage.py
#    confirms the held-out bodies are covered before an eval)
#
# --source-mjcf IS LOAD-BEARING: the solver resolves a bare subject id to
# smplx_omomo_<id>.xml, and there is no sub401 there -- the CARI4D bodies are
# smplh_behave_sub4xx.xml. A missing flag fails loudly (no such file), which
# is the right failure, but it is the flag to know about.
#
# READ THE VERDICT: retarget_contact.py refuses to write a solve whose contact
# error got worse, and prints a FAILURES block. Raise ITERS and rerun for those
# (finished pairs are skipped, so a rerun only does the failures).
set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SUBJECTS=(401 402 404 409 411 412 458)
SID="${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}"
MOTION_DIR="${MOTION_DIR:-InterAct/behave_cari4d_bball7_cf2}"
OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_bball7_retarget}"
CFG="${CFG:-isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_bball7_geoall__f0.yaml}"
ITERS="${ITERS:-300}"
ASSETS=isaacgym/src/intermimic/data/assets

MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub"$SID".xml 2>/dev/null | head -1)
[ -n "$MJCF" ] || { echo "[bball7-retarget] ERROR: no MJCF for sub$SID under $ASSETS/smplx" >&2; exit 1; }
[ -d "$MOTION_DIR" ] || { echo "[bball7-retarget] ERROR: no motion dir $MOTION_DIR (run the convert job first)" >&2; exit 1; }
n=$(ls "$MOTION_DIR"/sub"$SID"_*.pt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "[bball7-retarget] ERROR: no sub${SID}_* clips in $MOTION_DIR" >&2; exit 1; }

echo "[bball7-retarget] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[bball7-retarget] source sub$SID ($n clips) via $MJCF -> ${OUT_PREFIX}_src$SID  iters=$ITERS"

python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets-from "$CFG" \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

# Held-out eval bodies, same tree (finished pairs are skipped, so this is
# additive and safe to rerun).
HELDOUT="${HELDOUT:-sub10 sub13 sub16}"
echo "[bball7-retarget] held-out bodies: $HELDOUT"
python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets $HELDOUT \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

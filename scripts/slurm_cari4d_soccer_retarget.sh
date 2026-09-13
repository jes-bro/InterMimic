#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="soccer-retarget"
#SBATCH --output=soccer-retarget-%A_%a.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Retarget every soccer clip of ONE subject onto every f0 body (contact-aware
# solve, CPU only). One array task per subject, 15 subjects:
#
#   sbatch --array=0-14 scripts/slurm_cari4d_soccer_retarget.sh
#
# Writes <OUT_PREFIX>_src<id>/<body>/<clip>.pt for the 43 TRAINING bodies (from
# the soccer15 arm's cfg) AND the 3 held-out eval bodies sub10 sub13 sub16.
# When all 15 have PASSED, merge into ONE tree; both soccer arms (15 people and
# the top-7) read it -- the 7-arm just names fewer sources in dataSub, and the
# task only opens the (body, clip) files its dataSub selects:
#
#   python3 scripts/merge_retarget_trees.py \
#       --sources InterAct/behave_cari4d_soccer_retarget_src405 \
#                 InterAct/behave_cari4d_soccer_retarget_src480 \
#                 InterAct/behave_cari4d_soccer_retarget_src482 \
#                 InterAct/behave_cari4d_soccer_retarget_src484 \
#                 InterAct/behave_cari4d_soccer_retarget_src485 \
#                 InterAct/behave_cari4d_soccer_retarget_src487 \
#                 InterAct/behave_cari4d_soccer_retarget_src488 \
#                 InterAct/behave_cari4d_soccer_retarget_src489 \
#                 InterAct/behave_cari4d_soccer_retarget_src490 \
#                 InterAct/behave_cari4d_soccer_retarget_src491 \
#                 InterAct/behave_cari4d_soccer_retarget_src493 \
#                 InterAct/behave_cari4d_soccer_retarget_src494 \
#                 InterAct/behave_cari4d_soccer_retarget_src495 \
#                 InterAct/behave_cari4d_soccer_retarget_src496 \
#                 InterAct/behave_cari4d_soccer_retarget_src497 \
#       --out InterAct/behave_cari4d_soccer_f0_bodymajor
#   (expect: 46 body dirs, 46 x 61 = 2806 links)
#
# --source-mjcf IS LOAD-BEARING: the solver resolves a bare subject id to
# smplx_omomo_<id>.xml; the CARI4D bodies are smplh_behave_sub4xx.xml.
# READ THE VERDICT: retarget_contact.py refuses to write a solve whose contact
# error got worse and prints a FAILURES block. Raise ITERS and rerun for those
# (finished pairs are skipped). NOTE: on the 16 allow-listed all-free clips the
# contact-aware solve has no contact targets and reduces to a body-only solve;
# their "contact error" lines are not meaningful.
set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SUBJECTS=(405 480 482 484 485 487 488 489 490 491 493 494 495 496 497)
SID="${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}"
MOTION_DIR="${MOTION_DIR:-InterAct/behave_cari4d_soccer_cf}"
OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_soccer_retarget}"
CFG="${CFG:-isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_soccer15_geoall__f0.yaml}"
ITERS="${ITERS:-300}"
ASSETS=isaacgym/src/intermimic/data/assets

MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub"$SID".xml 2>/dev/null | head -1)
[ -n "$MJCF" ] || { echo "[soccer-retarget] ERROR: no MJCF for sub$SID under $ASSETS/smplx" >&2; exit 1; }
[ -d "$MOTION_DIR" ] || { echo "[soccer-retarget] ERROR: no motion dir $MOTION_DIR (run the convert job + relabel first)" >&2; exit 1; }
n=$(ls "$MOTION_DIR"/sub"$SID"_*.pt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "[soccer-retarget] ERROR: no sub${SID}_* clips in $MOTION_DIR" >&2; exit 1; }

echo "[soccer-retarget] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[soccer-retarget] source sub$SID ($n clips) via $MJCF -> ${OUT_PREFIX}_src$SID  iters=$ITERS"

python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets-from "$CFG" \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

HELDOUT="${HELDOUT:-sub10 sub13 sub16}"
echo "[soccer-retarget] held-out bodies: $HELDOUT"
python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets $HELDOUT \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

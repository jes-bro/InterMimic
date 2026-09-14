#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="cpr-retarget"
#SBATCH --output=cpr-retarget-%A_%a.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Retarget every CPR clip of ONE subject onto every f0 body (contact-aware
# solve, CPU only). One array task per subject, 13 subjects:
#
#   sbatch --array=0-12 --exclude=simurgh6,simurgh2 scripts/slurm_cari4d_cpr_retarget.sh
#
# Reads the KNEE-PASSED set (InterAct/behave_cari4d_cpr_kn, see
# slurm_cari4d_cpr_convert.sh) so the retarget's contact targets are the
# corrected labels. Writes <OUT_PREFIX>_src<id>/<body>/<clip>.pt for the 43
# TRAINING bodies (from the cpr13 arm's cfg) AND the 3 held-out eval bodies
# sub10 sub13 sub16. When all 13 have PASSED, merge into ONE tree:
#
#   python3 scripts/merge_retarget_trees.py \
#       --sources InterAct/behave_cari4d_cpr_retarget_src509 \
#                 InterAct/behave_cari4d_cpr_retarget_src510 \
#                 InterAct/behave_cari4d_cpr_retarget_src512 \
#                 InterAct/behave_cari4d_cpr_retarget_src513 \
#                 InterAct/behave_cari4d_cpr_retarget_src517 \
#                 InterAct/behave_cari4d_cpr_retarget_src564 \
#                 InterAct/behave_cari4d_cpr_retarget_src565 \
#                 InterAct/behave_cari4d_cpr_retarget_src566 \
#                 InterAct/behave_cari4d_cpr_retarget_src567 \
#                 InterAct/behave_cari4d_cpr_retarget_src569 \
#                 InterAct/behave_cari4d_cpr_retarget_src577 \
#                 InterAct/behave_cari4d_cpr_retarget_src579 \
#                 InterAct/behave_cari4d_cpr_retarget_src580 \
#       --out InterAct/behave_cari4d_cpr_f0_bodymajor
#   (expect: 46 body dirs, 46 x 23 = 1058 links)
#
# --source-mjcf IS LOAD-BEARING: the solver resolves a bare subject id to
# smplx_omomo_<id>.xml; the CARI4D bodies are smplh_behave_sub5xx.xml.
# READ THE VERDICT: retarget_contact.py refuses to write a solve whose contact
# error got worse and prints a FAILURES block. Raise ITERS and rerun for those
# (finished pairs are skipped). NOTE the object is STATIC: the contact targets
# are the hands on the manikin's chest on nearly every frame, so the solve is
# a hands-on-surface fit throughout, not the intermittent-contact case of the
# ball datasets.
set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SUBJECTS=(509 510 512 513 517 564 565 566 567 569 577 579 580)
SID="${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}"
MOTION_DIR="${MOTION_DIR:-InterAct/behave_cari4d_cpr_kn}"
OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_cpr_retarget}"
CFG="${CFG:-isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_cpr13_geoall__f0.yaml}"
ITERS="${ITERS:-300}"
ASSETS=isaacgym/src/intermimic/data/assets

MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub"$SID".xml 2>/dev/null | head -1)
[ -n "$MJCF" ] || { echo "[cpr-retarget] ERROR: no MJCF for sub$SID under $ASSETS/smplx" >&2; exit 1; }
[ -d "$MOTION_DIR" ] || { echo "[cpr-retarget] ERROR: no motion dir $MOTION_DIR (run the convert job + the knee pass first)" >&2; exit 1; }
n=$(ls "$MOTION_DIR"/sub"$SID"_*.pt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "[cpr-retarget] ERROR: no sub${SID}_* clips in $MOTION_DIR" >&2; exit 1; }

echo "[cpr-retarget] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[cpr-retarget] source sub$SID ($n clips) via $MJCF -> ${OUT_PREFIX}_src$SID  iters=$ITERS"

python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets-from "$CFG" \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

HELDOUT="${HELDOUT:-sub10 sub13 sub16}"
echo "[cpr-retarget] held-out bodies: $HELDOUT"
python3 -u scripts/retarget_contact.py --batch \
    --motion-dir "$MOTION_DIR" \
    --source "sub$SID" --source-mjcf "$MJCF" \
    --targets $HELDOUT \
    --iters "$ITERS" \
    --out-dir "${OUT_PREFIX}_src$SID"

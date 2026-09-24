#!/bin/sh
# retarget_bball7_local.sh -- build a bball7 body-major reference tree on one
# machine (ikura), the local counterpart of scripts/slurm_cari4d_bball7_retarget.sh.
#
# WHY IT EXISTS: the contact-weighting ABLATION needs a second tree that differs
# from the real one in exactly one thing -- the weight on bodies in contact:
#
#   W_CONTACT=0 sh scripts/retarget_bball7_local.sh
#
# w=1+W_CONTACT*contact, so 10 (the default, = the real tree) weights a contact
# body 11x, and 0 weights it the same as every other body: plain kinematic
# retargeting. Everything else -- clips, bodies, iters, source MJCFs -- is
# identical, which is what makes the trained arms comparable.
#
# WRITES <OUT_PREFIX>_src<id>/<body>/<clip>.pt per source, then merges them into
# $MERGED. Never point OUT_PREFIX/MERGED at the real tree: an arm's references
# are what it trained against, and overwriting them silently rewrites history for
# every arm that already used them.
#
# Usage (repo root, conda env with torch):
#   W_CONTACT=0 MERGED=InterAct/behave_cari4d_bball7_f0_bodymajor_uniformret \
#     nohup sh scripts/retarget_bball7_local.sh > retarget-uniform.log 2>&1 &
#   DRY=1 W_CONTACT=0 sh scripts/retarget_bball7_local.sh      # print the plan
#
# Env: W_CONTACT (default 10), WORKERS (default nproc-2), ITERS (300),
#      ALLOW_WORSE_CM (0 for the real tree, 1000 = keep-everything for an
#      ablation -- see the note where it is set), MOTION_DIR, CFG, OUT_PREFIX,
#      MERGED, HELDOUT, DRY.
#
# CPU only, no Isaac Gym, no GPU -- it can run beside training and evals.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

W_CONTACT="${W_CONTACT:-10}"
ITERS="${ITERS:-300}"
WORKERS="${WORKERS:-$(( $(nproc) - 2 ))}"
MOTION_DIR="${MOTION_DIR:-InterAct/behave_cari4d_bball7_cf2}"
CFG="${CFG:-isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_bball7_geoall__f0.yaml}"
HELDOUT="${HELDOUT:-sub10 sub13 sub16}"
ASSETS=isaacgym/src/intermimic/data/assets
SUBJECTS="401 402 404 409 411 412 458"

# Default output names carry the weight, so an ablation tree cannot be mistaken
# for the real one by looking at the path.
if [ "$W_CONTACT" = "10" ]; then
  OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_bball7_retarget}"
  MERGED="${MERGED:-InterAct/behave_cari4d_bball7_f0_bodymajor}"
else
  OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_bball7_retarget_w${W_CONTACT}}"
  MERGED="${MERGED:-InterAct/behave_cari4d_bball7_f0_bodymajor_w${W_CONTACT}}"
fi

# A solve that ends up WORSE than not retargeting is refused by default -- right
# for the real tree, where a regression means an under-converged solve.
#
# WRONG for an ablation. With uniform weights the solve is SUPPOSED to do badly
# at contacts; refusing those pairs would keep only the ones where it happened to
# do well and leave holes elsewhere, so the arm would train on a cherry-picked
# subset and the ablation would understate its own cost. So when W_CONTACT is not
# the real 10, keep every solve and let the arm train on the bad references --
# that is the thing being measured.
if [ "$W_CONTACT" = "10" ]; then
  ALLOW_WORSE_CM="${ALLOW_WORSE_CM:-0}"
else
  ALLOW_WORSE_CM="${ALLOW_WORSE_CM:-1000}"      # cm: effectively "write them all"
fi

[ -d "$MOTION_DIR" ] || { echo "ERROR: no motion dir $MOTION_DIR" >&2; exit 2; }
[ -f "$CFG" ] || { echo "ERROR: no cfg $CFG" >&2; exit 2; }

echo "== bball7 retarget  w_contact=$W_CONTACT  iters=$ITERS  workers=$WORKERS"
echo "   motion : $MOTION_DIR"
echo "   bodies : subjectBodies of $(basename "$CFG") + held-out [$HELDOUT]"
echo "   per-src: ${OUT_PREFIX}_src<id>"
echo "   merged : $MERGED"
echo "   worse  : --allow-worse-cm $ALLOW_WORSE_CM"
[ "$W_CONTACT" = "0" ] && echo "   NOTE: w_contact=0 is the ABLATION tree (uniform weights, no contact awareness);
         regressed solves are KEPT (allow-worse $ALLOW_WORSE_CM cm) -- refusing them
         would cherry-pick the pairs uniform weighting happened to do well on"

# Refuse to write into a tree that already exists with DIFFERENT settings: the
# per-source dirs are resumable by design (finished pairs are skipped), which is
# only safe when the settings match.
if [ -d "$MERGED" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "ERROR: $MERGED already exists. Pick another MERGED, or FORCE=1 if you are" >&2
  echo "       certain it was built with w_contact=$W_CONTACT." >&2
  exit 2
fi

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

for SID in $SUBJECTS; do
  # --source-mjcf is load-bearing: the solver resolves a bare id to
  # smplx_omomo_<id>.xml and the CARI4D bodies are smplh_behave_sub4xx.xml.
  MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub"$SID".xml 2>/dev/null | head -1)
  [ -n "$MJCF" ] || { echo "ERROR: no MJCF for sub$SID" >&2; exit 2; }
  echo "== sub$SID via $(basename "$MJCF")"
  # training bodies (from the arm's cfg)
  python3 -u scripts/retarget_contact.py --batch \
      --motion-dir "$MOTION_DIR" --source "sub$SID" --source-mjcf "$MJCF" \
      --targets-from "$CFG" --iters "$ITERS" --workers "$WORKERS" \
      --w-contact "$W_CONTACT" --allow-worse-cm "$ALLOW_WORSE_CM" \
      --out-dir "${OUT_PREFIX}_src$SID"
  # held-out eval bodies, same tree (additive; finished pairs are skipped)
  python3 -u scripts/retarget_contact.py --batch \
      --motion-dir "$MOTION_DIR" --source "sub$SID" --source-mjcf "$MJCF" \
      --targets $HELDOUT --iters "$ITERS" --workers "$WORKERS" \
      --w-contact "$W_CONTACT" --allow-worse-cm "$ALLOW_WORSE_CM" \
      --out-dir "${OUT_PREFIX}_src$SID"
done

echo "== merging 7 per-source trees -> $MERGED"
python3 -u scripts/merge_retarget_trees.py \
    --sources $(for SID in $SUBJECTS; do printf '%s ' "${OUT_PREFIX}_src$SID"; done) \
    --out "$MERGED"

echo "== done. bodies: $(ls "$MERGED" | wc -l)  (the real tree has 46)"
echo "   clips per body: $(ls "$MERGED/$(ls "$MERGED" | head -1)" | wc -l)  (expect 48)"

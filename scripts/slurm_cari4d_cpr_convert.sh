#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="cpr-convert"
#SBATCH --output=cpr-convert-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Convert every clip in a CPR manifest (scripts/cari4d_bball7_manifest.py
# --activity cpr) into InterMimic motion files. The CPR twin of
# scripts/slurm_cari4d_soccer_convert.sh; the per-clip conversion is identical
# (same wrapper, same flags). What differs -- the object is a MANIKIN that
# never moves:
#
#   * NO hand/foot relabel and NO census. Those scripts model the object as a
#     sphere with a radius from the mesh bounding box (~0.9 m for a manikin) and
#     would mark every hand "inside". The converter's own labels are
#     vertex-distance and already right for a static object.
#   * The ONE label fix CPR needs is the knee pass: the converter marks a
#     kneeler's knees/ankles/toes "must not touch" (they are > 10 cm from the
#     manikin, resting on the FLOOR) and the contact reward would tax kneeling
#     itself. Run it BY HAND after this job, with the conda env active (the
#     driver's bare python3 has no torch -- same as the soccer census):
#       python3 scripts/relabel_contact_cpr.py \
#           --src-dir InterAct/<TAG> --dst-dir InterAct/<TAG>_kn \
#           --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub509.xml
#     (--census first to see the counts; it writes nothing)
#   * Floor seating: rotate_pt.py now seats on the LOWEST OF ALL BODIES (knees
#     for a kneeler), the driver's default since 2026-09-14.
#
#   MANIFEST=/simurgh2/projects/ret-hoi/CARI4D/bundles/cpr/manifest.csv \
#   BUNDLES_ROOT=/simurgh2/projects/ret-hoi/CARI4D/bundles/cpr \
#   sbatch scripts/slurm_cari4d_cpr_convert.sh
#
# Per clip: ROTATE_AXIS=x (upright flip; silently skipped when unset), REPLAY=0,
# BETAS_NPZ (one body per person). Judged by the wrapper's exit status; a
# partial file is removed on failure; finished clips are skipped on resubmit.
set -u

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.csv}"
BUNDLES_ROOT="${BUNDLES_ROOT:?set BUNDLES_ROOT=/path/to/extracted/bundles}"
DATASET_TAG="${DATASET_TAG:-behave_cari4d_cpr}"
BETAS_NPZ="${BETAS_NPZ:-scripts/cpr_subject_betas.npz}"
INTERMIMIC="${INTERMIMIC:-/simurgh2/projects/ret-hoi/InterMimic}"
ASSETS="$INTERMIMIC/isaacgym/src/intermimic/data/assets"

cd "$INTERMIMIC"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST" >&2; exit 1; }
[ -d "$BUNDLES_ROOT" ] || { echo "ERROR: no bundles dir at $BUNDLES_ROOT" >&2; exit 1; }
[ -f "$BETAS_NPZ" ] || { echo "ERROR: no shared betas at $BETAS_NPZ -- run scripts/cari4d_subject_betas.py first" >&2; exit 1; }

echo "[cpr] host=$(hostname) job=${SLURM_JOB_ID:-none} tag=$DATASET_TAG"
echo "[cpr] manifest=$MANIFEST bundles=$BUNDLES_ROOT betas=$BETAS_NPZ"

n_expected=$(tail -n +2 "$MANIFEST" | grep -c .)
n_done=0; n_skip=0
# CSV columns: clip,subject,subject_id,clip_idx,object,take,gender,n_frames,lo,hi,export,bundle,mesh
while IFS=, read -r clip subject subject_id clip_idx object take gender n_frames lo hi export bundle mesh; do
    [ "$clip" = "clip" ] && continue
    [ -z "$clip" ] && continue
    mesh="${mesh%$'\r'}"
    PT="InterAct/$DATASET_TAG/sub${subject_id}_${object}_${clip_idx}.pt"
    if [ -f "$PT" ]; then
        echo "[cpr] $clip -> $PT exists, skipping"; n_skip=$((n_skip + 1)); continue
    fi
    echo
    echo "=============================================================================="
    echo "[cpr] $clip -> sub${subject_id}_${object}_${clip_idx}  ($n_frames frames, $gender)"
    echo "=============================================================================="
    if ! BUNDLE="$BUNDLES_ROOT/$bundle" MESH="$BUNDLES_ROOT/$mesh" \
         SUBJECT_ID="$subject_id" OBJECT_NAME="$object" CLIP_IDX="$clip_idx" GENDER="$gender" \
         DATASET_TAG="$DATASET_TAG" BETAS_NPZ="$BETAS_NPZ" ROTATE_AXIS=x REPLAY=0 \
         INTERMIMIC="$INTERMIMIC" \
         bash scripts/slurm_cari4d_to_mimic.sh </dev/null; then
        echo "[cpr] ERROR: conversion of $clip FAILED -- removing partial $PT and stopping" >&2
        rm -f "$PT"; exit 1
    fi
    [ -f "$PT" ] || { echo "[cpr] ERROR: conversion of $clip produced no $PT -- stopping" >&2; exit 1; }
    n_done=$((n_done + 1))
done < "$MANIFEST"

echo
echo "[cpr] converted $n_done, skipped $n_skip, of $n_expected manifest rows"
n_pt=$(ls InterAct/"$DATASET_TAG"/sub5*_*.pt 2>/dev/null | wc -l)      # CPR ids are sub509..sub580
if [ "$n_pt" -ne "$n_expected" ]; then
    echo "[cpr] ERROR: $n_pt motion files for $n_expected manifest rows" >&2; exit 1
fi
for sid in $(cut -d, -f3 "$MANIFEST" | tail -n +2 | sort -u); do
    ls "$ASSETS"/smplx/smplh_*_sub"$sid".xml >/dev/null 2>&1 || {
        echo "[cpr] ERROR: no MJCF for sub$sid under $ASSETS/smplx" >&2; exit 1; }
done
MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub$(cut -d, -f3 "$MANIFEST" | tail -n +2 | sort -u | head -1).xml | head -1)

echo
echo "[cpr] done: $n_pt clips in InterAct/$DATASET_TAG. NOW, with the conda env active, the knee pass:"
echo "[cpr]   python3 scripts/relabel_contact_cpr.py --src-dir InterAct/$DATASET_TAG --mjcf $MJCF --census"
echo "[cpr]   python3 scripts/relabel_contact_cpr.py --src-dir InterAct/$DATASET_TAG --dst-dir InterAct/${DATASET_TAG}_kn --mjcf $MJCF"
echo "[cpr] Do NOT run relabel_contact_human.py / relabel_contact_flags.py / relabel_contact_soccer.py on this set."

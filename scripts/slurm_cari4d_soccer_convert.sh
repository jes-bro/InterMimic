#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="soccer-convert"
#SBATCH --output=soccer-convert-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Convert every clip in a soccer manifest (scripts/cari4d_bball7_manifest.py
# --activity soccer) into InterMimic motion files, then run the FOOT contact
# CENSUS. The soccer twin of scripts/slurm_cari4d_bball7_convert.sh; the
# per-clip conversion is identical (same wrapper, same flags). What differs:
#
#   * contact relabel is relabel_contact_soccer.py (feet), and this job only
#     runs its --census: the foot threshold is chosen from the census's sweep,
#     not copied from the hand script. Read the sweep at the end of the .out,
#     then write the relabelled set by hand:
#       python3 scripts/relabel_contact_soccer.py \
#           --src-dir InterAct/<TAG> --dst-dir InterAct/<TAG>_cf \
#           --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub405.xml \
#           --ball-radius-from-mesh isaacgym/src/intermimic/data/assets/objects/objects \
#           --threshold <chosen>
#   * no wrist-based first pass (relabel_contact_flags.py): the soccer script
#     re-derives contact_obj from the same foot criterion in one step.
#
#   MANIFEST=/simurgh2/projects/ret-hoi/CARI4D/bundles/soccer/manifest.csv \
#   BUNDLES_ROOT=/simurgh2/projects/ret-hoi/CARI4D/bundles/soccer \
#   sbatch scripts/slurm_cari4d_soccer_convert.sh
#
# Per clip: ROTATE_AXIS=x (upright flip; silently skipped when unset), REPLAY=0,
# BETAS_NPZ (one body per person). Judged by the wrapper's exit status; a
# partial file is removed on failure; finished clips are skipped on resubmit.
set -u

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.csv}"
BUNDLES_ROOT="${BUNDLES_ROOT:?set BUNDLES_ROOT=/path/to/extracted/bundles}"
DATASET_TAG="${DATASET_TAG:-behave_cari4d_soccer}"
BETAS_NPZ="${BETAS_NPZ:-scripts/soccer_subject_betas.npz}"
INTERMIMIC="${INTERMIMIC:-/simurgh2/projects/ret-hoi/InterMimic}"
ASSETS="$INTERMIMIC/isaacgym/src/intermimic/data/assets"

cd "$INTERMIMIC"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST" >&2; exit 1; }
[ -d "$BUNDLES_ROOT" ] || { echo "ERROR: no bundles dir at $BUNDLES_ROOT" >&2; exit 1; }
[ -f "$BETAS_NPZ" ] || { echo "ERROR: no shared betas at $BETAS_NPZ -- run scripts/cari4d_subject_betas.py first" >&2; exit 1; }

echo "[soccer] host=$(hostname) job=${SLURM_JOB_ID:-none} tag=$DATASET_TAG"
echo "[soccer] manifest=$MANIFEST bundles=$BUNDLES_ROOT betas=$BETAS_NPZ"

n_expected=$(tail -n +2 "$MANIFEST" | grep -c .)
n_done=0; n_skip=0
# CSV columns: clip,subject,subject_id,clip_idx,object,take,gender,n_frames,lo,hi,export,bundle,mesh
while IFS=, read -r clip subject subject_id clip_idx object take gender n_frames lo hi export bundle mesh; do
    [ "$clip" = "clip" ] && continue
    [ -z "$clip" ] && continue
    mesh="${mesh%$'\r'}"
    PT="InterAct/$DATASET_TAG/sub${subject_id}_${object}_${clip_idx}.pt"
    if [ -f "$PT" ]; then
        echo "[soccer] $clip -> $PT exists, skipping"; n_skip=$((n_skip + 1)); continue
    fi
    echo
    echo "=============================================================================="
    echo "[soccer] $clip -> sub${subject_id}_${object}_${clip_idx}  ($n_frames frames, $gender)"
    echo "=============================================================================="
    if ! BUNDLE="$BUNDLES_ROOT/$bundle" MESH="$BUNDLES_ROOT/$mesh" \
         SUBJECT_ID="$subject_id" OBJECT_NAME="$object" CLIP_IDX="$clip_idx" GENDER="$gender" \
         DATASET_TAG="$DATASET_TAG" BETAS_NPZ="$BETAS_NPZ" ROTATE_AXIS=x REPLAY=0 \
         INTERMIMIC="$INTERMIMIC" \
         bash scripts/slurm_cari4d_to_mimic.sh </dev/null; then
        echo "[soccer] ERROR: conversion of $clip FAILED -- removing partial $PT and stopping" >&2
        rm -f "$PT"; exit 1
    fi
    [ -f "$PT" ] || { echo "[soccer] ERROR: conversion of $clip produced no $PT -- stopping" >&2; exit 1; }
    n_done=$((n_done + 1))
done < "$MANIFEST"

echo
echo "[soccer] converted $n_done, skipped $n_skip, of $n_expected manifest rows"
n_pt=$(ls InterAct/"$DATASET_TAG"/sub4*_*.pt 2>/dev/null | wc -l)
if [ "$n_pt" -ne "$n_expected" ]; then
    echo "[soccer] ERROR: $n_pt motion files for $n_expected manifest rows" >&2; exit 1
fi
for sid in $(cut -d, -f3 "$MANIFEST" | tail -n +2 | sort -u); do
    ls "$ASSETS"/smplx/smplh_*_sub"$sid".xml >/dev/null 2>&1 || {
        echo "[soccer] ERROR: no MJCF for sub$sid under $ASSETS/smplx" >&2; exit 1; }
done
MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub$(cut -d, -f3 "$MANIFEST" | tail -n +2 | sort -u | head -1).xml | head -1)

echo
echo "[soccer] FOOT CONTACT CENSUS (nothing written) -- read the THRESHOLD SWEEP at the end:"
python3 scripts/relabel_contact_soccer.py \
    --src-dir "InterAct/$DATASET_TAG" --mjcf "$MJCF" \
    --ball-radius-from-mesh "$ASSETS/objects/objects" --census
echo
echo "[soccer] done: $n_pt clips in InterAct/$DATASET_TAG. Choose --threshold from the sweep, then"
echo "[soccer]   python3 scripts/relabel_contact_soccer.py --src-dir InterAct/$DATASET_TAG --dst-dir InterAct/${DATASET_TAG}_cf --mjcf $MJCF --ball-radius-from-mesh $ASSETS/objects/objects --threshold <chosen>"

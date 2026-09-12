#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="bball7-convert"
#SBATCH --output=bball7-convert-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
#
# Convert EVERY clip in a bball7 manifest (scripts/cari4d_bball7_manifest.py)
# into InterMimic motion files, then run the two contact relabels that make the
# cf2 lineage the g3 bball arm trains on. One job, clips in sequence: the
# conversions share the InterAct dataset dir and rewrite each subject's MJCF,
# which is safe in order and a race in parallel.
#
#   MANIFEST=/simurgh2/projects/ret-hoi/CARI4D/bundles/bball7/manifest.csv \
#   BUNDLES_ROOT=/simurgh2/projects/ret-hoi/CARI4D/bundles/bball7 \
#   sbatch scripts/slurm_cari4d_bball7_convert.sh
#
# Per clip this calls scripts/slurm_cari4d_to_mimic.sh with the manifest's
# subject id, clip index, object name, bundle, mesh and gender, plus:
#   ROTATE_AXIS=x   the flip that puts the clip upright (skipped SILENTLY when
#                   unset -- the 2026-08-17 lesson; never leave it off)
#   REPLAY=0        no per-clip video; render after training, not 52 times here
#   BETAS_NPZ       ONE body per subject (scripts/cari4d_subject_betas.py)
# A clip whose motion file already exists is skipped, so a resubmission resumes.
#
# Output:  InterAct/<TAG>/sub4xx_<object>_<idx>.pt      raw + rotated
#          InterAct/<TAG>_cf/                            contact_obj relabelled
#          InterAct/<TAG>_cf2/                            hand contacts relabelled  <- train on this
#          assets/smplx/smplh_behave_sub4xx.xml           one MJCF per subject
#          assets/objects/<object>.urdf + objects/<object>/<object>.obj   one ball per clip
set -u

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.csv}"
BUNDLES_ROOT="${BUNDLES_ROOT:?set BUNDLES_ROOT=/path/to/extracted/bundles}"
DATASET_TAG="${DATASET_TAG:-behave_cari4d_bball7}"
BETAS_NPZ="${BETAS_NPZ:-scripts/bball7_subject_betas.npz}"
INTERMIMIC="${INTERMIMIC:-/simurgh2/projects/ret-hoi/InterMimic}"
ASSETS="$INTERMIMIC/isaacgym/src/intermimic/data/assets"

cd "$INTERMIMIC"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST" >&2; exit 1; }
[ -d "$BUNDLES_ROOT" ] || { echo "ERROR: no bundles dir at $BUNDLES_ROOT" >&2; exit 1; }
[ -f "$BETAS_NPZ" ] || { echo "ERROR: no shared betas at $BETAS_NPZ -- run scripts/cari4d_subject_betas.py first" >&2; exit 1; }

echo "[bball7] host=$(hostname) job=${SLURM_JOB_ID:-none} tag=$DATASET_TAG"
echo "[bball7] manifest=$MANIFEST bundles=$BUNDLES_ROOT betas=$BETAS_NPZ"

# Expected clip count comes from the FILE, not from the loop: a child that
# reads stdin inside the loop would eat manifest rows, and a loop counter
# would then agree with the shortfall.
n_expected=$(tail -n +2 "$MANIFEST" | grep -c .)
n_done=0; n_skip=0
# CSV columns: clip,subject,subject_id,clip_idx,object,take,gender,n_frames,lo,hi,export,bundle,mesh
while IFS=, read -r clip subject subject_id clip_idx object take gender n_frames lo hi export bundle mesh; do
    [ "$clip" = "clip" ] && continue            # header
    [ -z "$clip" ] && continue
    # Python's csv module ends lines with \r\n; `read` leaves the \r on the LAST
    # field, which is the mesh path -- invisible in any log, and "missing input"
    # on every clip. Strip it.
    mesh="${mesh%$'\r'}"
    PT="InterAct/$DATASET_TAG/sub${subject_id}_${object}_${clip_idx}.pt"
    if [ -f "$PT" ]; then
        echo "[bball7] $clip -> $PT exists, skipping"; n_skip=$((n_skip + 1)); continue
    fi
    echo
    echo "=============================================================================="
    echo "[bball7] $clip -> sub${subject_id}_${object}_${clip_idx}  ($n_frames frames, $gender)"
    echo "=============================================================================="
    # The wrapper installs the .pt at step 3 and rotates it at step 3.5 (the
    # upright flip). Judge success by its EXIT STATUS, not by the file: a
    # failure between the two leaves an upside-down clip that a resubmission's
    # skip-if-exists would then keep forever. Remove the partial file on failure.
    # stdin is redirected so nothing inside can consume manifest rows.
    if ! BUNDLE="$BUNDLES_ROOT/$bundle" MESH="$BUNDLES_ROOT/$mesh" \
         SUBJECT_ID="$subject_id" OBJECT_NAME="$object" CLIP_IDX="$clip_idx" GENDER="$gender" \
         DATASET_TAG="$DATASET_TAG" BETAS_NPZ="$BETAS_NPZ" ROTATE_AXIS=x REPLAY=0 \
         INTERMIMIC="$INTERMIMIC" \
         bash scripts/slurm_cari4d_to_mimic.sh </dev/null; then
        echo "[bball7] ERROR: conversion of $clip FAILED -- removing partial $PT and stopping" >&2
        rm -f "$PT"; exit 1
    fi
    if [ ! -f "$PT" ]; then
        echo "[bball7] ERROR: conversion of $clip produced no $PT -- stopping" >&2; exit 1
    fi
    n_done=$((n_done + 1))
done < "$MANIFEST"

echo
echo "[bball7] converted $n_done, skipped $n_skip, of $n_expected manifest rows"
n_pt=$(ls InterAct/"$DATASET_TAG"/sub4*_*.pt 2>/dev/null | wc -l)
if [ "$n_pt" -ne "$n_expected" ]; then
    echo "[bball7] ERROR: $n_pt motion files for $n_expected manifest rows" >&2; exit 1
fi

# One MJCF per subject must exist and be the SAME file every clip was built on.
for sid in $(cut -d, -f3 "$MANIFEST" | tail -n +2 | sort -u); do
    ls "$ASSETS"/smplx/smplh_*_sub"$sid".xml >/dev/null 2>&1 || {
        echo "[bball7] ERROR: no MJCF for sub$sid under $ASSETS/smplx" >&2; exit 1; }
done
MJCF=$(ls "$ASSETS"/smplx/smplh_*_sub401.xml | head -1)    # body ORDER is all the relabels read

# A relabel dir is "done" only if it holds every clip: the relabel scripts write
# clip by clip and refuse an existing dir, so a killed run leaves a partial dir
# that must be rebuilt, not accepted.
complete_dir() { [ -d "$1" ] && [ "$(ls "$1"/*.pt 2>/dev/null | wc -l)" -eq "$n_expected" ]; }
redo_dir() { if [ -d "$1" ]; then echo "[bball7] $1 is partial -- rebuilding"; rm -rf "$1"; fi; }

# cf: contact_obj (channel 330) from wrist-to-ball-centre distance.
if ! complete_dir "InterAct/${DATASET_TAG}_cf"; then
    redo_dir "InterAct/${DATASET_TAG}_cf"
    python3 scripts/relabel_contact_flags.py \
        --src-dir "InterAct/$DATASET_TAG" --dst-dir "InterAct/${DATASET_TAG}_cf" --mjcf "$MJCF"
fi
# cf2: per-hand-body flags (331..382) from finger-to-SURFACE distance, with each
# clip's OWN ball radius (the balls differ by up to 2.5 cm in radius).
if ! complete_dir "InterAct/${DATASET_TAG}_cf2"; then
    redo_dir "InterAct/${DATASET_TAG}_cf2"
    python3 scripts/relabel_contact_human.py \
        --src-dir "InterAct/${DATASET_TAG}_cf" --dst-dir "InterAct/${DATASET_TAG}_cf2" \
        --mjcf "$MJCF" --ball-radius-from-mesh "$ASSETS/objects/objects"
fi
complete_dir "InterAct/${DATASET_TAG}_cf2" || { echo "[bball7] ERROR: ${DATASET_TAG}_cf2 incomplete" >&2; exit 1; }
echo "[bball7] done: train on InterAct/${DATASET_TAG}_cf2 ($(ls InterAct/${DATASET_TAG}_cf2/*.pt | wc -l) clips)"
echo "[bball7] next: sbatch --array=0-6 scripts/slurm_cari4d_bball7_retarget.sh"

#!/bin/bash
# gcp_stage.sh -- move teacher checkpoints and training data between machines
# through the GCS bucket, so the four g3 student arms can run on GCP.
#
# The bucket is the hub (gs://jesb-intermimic, us-central1; the VMs read it via
# --scopes=cloud-platform, simurgh has gcloud). Layout, mirroring the repo:
#     gs://jesb-intermimic/checkpoints/<exp>/nn/mimic[_N].pth
#     gs://jesb-intermimic/InterAct/<dir>/...
#     gs://jesb-intermimic/assets/objects/...
#
# Every subcommand is a thin, printed wrapper around `gcloud storage cp`; run
# it from the repo root of whichever machine holds the thing being pushed.
#
#   push-teacher <exp> [<exp>...]     latest numbered snapshot + mimic.pth of each
#                                     experiment under $ROOT (default checkpoints/)
#   push-omomo-data <sub>...          those sources' OMOMO_new clips + contact-retarget trees
#   push-act-data                     the three activity motion dirs + trees + assets/objects
#   pull-teachers <exp> [<exp>...]    bucket -> checkpoints/<exp>/nn/
#   pull-omomo-data                   all 13 sources' clips + trees (OMOMO student VM)
#   pull-act-data                     activity dirs + trees + assets/objects (activity VM)
#
# WHERE THE 13 OMOMO TEACHERS ARE (2026-09-15): laptop gdrive download = sub
# 1 2 3 5 9 11 12 14 (flat files -> scripts/install_flat_checkpoints.py first);
# GCP teacher VMs = 2 5 8 15 17 (each VM pushes its own); cluster (wormhole) =
# the rest. collect_g3_teachers.py on the student VM names anything missing.
set -eu
BUCKET="${BUCKET:-gs://jesb-intermimic}"
ROOT="${ROOT:-checkpoints}"
OMOMO_SOURCES="sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17"
ACT_MOTION_DIRS="behave_cari4d_bball7_cf2 behave_cari4d_soccer_cf behave_cari4d_cpr_kn"
ACT_TREES="behave_cari4d_bball7_f0_bodymajor behave_cari4d_soccer_f0_bodymajor behave_cari4d_cpr_f0_bodymajor"
OBJECTS_DIR="isaacgym/src/intermimic/data/assets/objects"

cp_() { echo "+ gcloud storage cp $*"; gcloud storage cp "$@"; }
need_dir() { [ -d "$1" ] || { echo "ERROR: missing directory $1 (run from the repo root on the machine that has it)" >&2; exit 1; }; }

cmd="${1:?usage: gcp_stage.sh <subcommand> [args]  (see header)}"; shift
case "$cmd" in
  push-teacher)
    [ $# -ge 1 ] || { echo "usage: push-teacher <exp> [<exp>...]" >&2; exit 1; }
    for exp in "$@"; do
        nn="$ROOT/$exp/nn"; need_dir "$nn"
        # latest numbered snapshot (the collect script prefers it) + mimic.pth if present
        latest=$(ls -1 "$nn"/mimic_[0-9]*.pth 2>/dev/null | sort | tail -1 || true)
        files=""
        [ -n "$latest" ] && files="$latest"
        [ -f "$nn/mimic.pth" ] && files="$files $nn/mimic.pth"
        [ -n "$files" ] || { echo "ERROR: no mimic*.pth in $nn" >&2; exit 1; }
        echo "[$exp] pushing:$(for f in $files; do printf ' %s (%.0f MB)' "$(basename "$f")" "$(( $(stat -c %s "$f") / 1048576 ))"; done)"
        cp_ $files "$BUCKET/checkpoints/$exp/nn/"
    done ;;
  push-omomo-data)
    [ $# -ge 1 ] || { echo "usage: push-omomo-data <sub>... e.g. sub1 sub3" >&2; exit 1; }
    need_dir InterAct/OMOMO_new
    for s in "$@"; do
        n=${s#sub}
        need_dir "InterAct/OMOMO_retarget_contact_src$n"
        cp_ InterAct/OMOMO_new/${s}_*.pt "$BUCKET/InterAct/OMOMO_new/"
        cp_ -r "InterAct/OMOMO_retarget_contact_src$n" "$BUCKET/InterAct/"
    done ;;
  push-act-data)
    for d in $ACT_MOTION_DIRS $ACT_TREES; do need_dir "InterAct/$d"; done
    need_dir "$OBJECTS_DIR"
    for d in $ACT_MOTION_DIRS $ACT_TREES; do cp_ -r "InterAct/$d" "$BUCKET/InterAct/"; done
    # every activity object's URDF + mesh (the repo ships only OMOMO's 19)
    cp_ -r "$OBJECTS_DIR" "$BUCKET/assets/" ;;
  pull-teachers)
    [ $# -ge 1 ] || { echo "usage: pull-teachers <exp> [<exp>...]" >&2; exit 1; }
    for exp in "$@"; do
        mkdir -p "$ROOT/$exp/nn"
        cp_ "$BUCKET/checkpoints/$exp/nn/*" "$ROOT/$exp/nn/"
    done ;;
  pull-omomo-data)
    mkdir -p InterAct/OMOMO_new
    for s in $OMOMO_SOURCES; do
        n=${s#sub}
        cp_ "$BUCKET/InterAct/OMOMO_new/${s}_*.pt" InterAct/OMOMO_new/
        cp_ -r "$BUCKET/InterAct/OMOMO_retarget_contact_src$n" InterAct/
    done
    echo "next: the merge_retarget_trees.py command in the srcall13 launcher header (builds OMOMO_retarget_contact_srcall13)" ;;
  pull-act-data)
    mkdir -p InterAct
    for d in $ACT_MOTION_DIRS $ACT_TREES; do cp_ -r "$BUCKET/InterAct/$d" InterAct/; done
    mkdir -p "$(dirname "$OBJECTS_DIR")"
    cp_ -r "$BUCKET/assets/objects" "$(dirname "$OBJECTS_DIR")/"
    echo "next: scripts/merge_activity_data.py (see the activity launcher header)" ;;
  *) echo "unknown subcommand: $cmd (see header)" >&2; exit 1 ;;
esac

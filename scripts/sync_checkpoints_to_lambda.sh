#!/bin/bash
# Sync v2 checkpoints + vanilla student.pth from laptop staging area to
# a Lambda Labs instance, recovering any that got nested by an earlier
# bad `rsync --mkpath` invocation.
#
# What this fixes:
#   - When you rsync `<file>.pth` to a non-existent destination path
#     with --mkpath, rsync treats the trailing `.pth` as a directory
#     component and nests the file inside. Result: mimic.pth is a dir,
#     not a file, and torch.load() fails. This script does it the right
#     way: mkdir -p the parent first, then rsync with trailing slash on
#     destination so rsync treats it as a target dir.
#
# What the script does:
#   1. For each checkpoint, SSH to Lambda and check whether the
#      destination is already a healthy regular file. If yes, skip.
#   2. If not, stage the file locally with the correct name (mimic.pth)
#      in /tmp, mkdir -p the parent dir on Lambda, rsync the staged
#      file into the dir (trailing slash, no --mkpath).
#   3. Clean up the /tmp staging file.
#
# Usage:
#   bash scripts/sync_checkpoints_to_lambda.sh                 # uses default IP
#   bash scripts/sync_checkpoints_to_lambda.sh ubuntu@1.2.3.4  # override host
#
# If your Lambda instance IP changes when you restart it, pass the new
# ubuntu@<ip> as the first argument.

set -u

# Default Lambda target — change here or pass as first arg to override.
LAMBDA=${1:-ubuntu@137.131.47.244}

# Local staging area on laptop — where the v2 checkpoints landed after
# the simurgh2 → laptop rsync hop. Each .pth here was renamed on the
# way down to disambiguate, so the v2 files have names like
# `smplx_distill_both_normreward_v2_mimic.pth`; the script renames them
# back to `mimic.pth` when pushing to Lambda.
LOCAL_STAGE=$HOME/Downloads/checkpoints/v2

# (local_file, remote_parent_dir, remote_filename) triples. Tab-separated.
# Remote_filename ALWAYS ends up as `mimic.pth` or `student.pth` on Lambda.
#
# Note: the v2 staging "files" are actually directories on the laptop
# (same nested-dir rsync bug that hit Lambda) — the actual rolling-latest
# checkpoint is at <staging-dir>/mimic.pth. Point sources there.
# Student.pth on the laptop sits directly in Downloads.
read -r -d '' MANIFEST <<EOF
$LOCAL_STAGE/smplx_distill_both_normreward_v2_mimic.pth/mimic.pth	/home/ubuntu/InterMimic/checkpoints/smplx_distill_both_normreward_v2/nn	mimic.pth
$LOCAL_STAGE/smplx_distill_both_v2_mimic.pth/mimic.pth	/home/ubuntu/InterMimic/checkpoints/smplx_distill_both_abl_v2/nn	mimic.pth
$LOCAL_STAGE/smplx_distill_both_nobetas_normreward_v2_mimic.pth/mimic.pth	/home/ubuntu/InterMimic/checkpoints/smplx_distill_both_nobetas_normreward_v2/nn	mimic.pth
$HOME/Downloads/student.pth	/home/ubuntu/InterMimic/checkpoints/smplx_student	student.pth
EOF

# sync_one — push a single checkpoint file to Lambda the safe way.
#   $1 = local source path (any filename)
#   $2 = remote parent directory
#   $3 = remote filename (what it should be called on Lambda)
sync_one() {
    local local_path=$1 remote_dir=$2 remote_name=$3
    local remote_full=$remote_dir/$remote_name

    echo "=== $remote_full ==="

    if [ ! -f "$local_path" ]; then
        echo "  SKIP: local source missing at $local_path"
        return
    fi

    # Check what's on Lambda. We want a regular file, not a dir, not absent.
    local status
    status=$(ssh "$LAMBDA" "if [ -f '$remote_full' ]; then echo file:\$(stat -c %s '$remote_full'); elif [ -d '$remote_full' ]; then echo dir; else echo missing; fi" 2>/dev/null)

    case "$status" in
        file:*)
            local sz=${status#file:}
            local lsz
            lsz=$(stat -c %s "$local_path")
            if [ "$sz" = "$lsz" ]; then
                echo "  OK: already present (matching size $sz)"
                return
            else
                echo "  size mismatch (remote=$sz local=$lsz); re-syncing"
            fi
            ;;
        dir)
            echo "  remote is a directory (bad nesting from prior rsync); removing"
            ssh "$LAMBDA" "rm -rf '$remote_full'"
            ;;
        missing)
            echo "  remote missing; syncing"
            ;;
        *)
            echo "  unexpected status '$status'; will attempt sync anyway"
            ;;
    esac

    # Stage with the correct destination filename so we can use the
    # rsync trailing-slash-on-dir form (the safest way to avoid the
    # leaf-as-dir nesting bug).
    local stage=/tmp/sync_ckpt_$$_$remote_name
    cp "$local_path" "$stage"

    # mkdir -p the remote parent (rsync without --mkpath needs this).
    ssh "$LAMBDA" "mkdir -p '$remote_dir'"

    # Push file into the dir. The TRAILING SLASH on the destination is
    # the magic — it forces rsync to treat $remote_dir as a directory
    # and put $stage's basename ($remote_name) inside it.
    rsync -avh --progress "$stage" "$LAMBDA:$remote_dir/"

    # Cleanup local staging.
    rm -f "$stage"

    # Verify it landed as a regular file.
    ssh "$LAMBDA" "ls -la '$remote_full'"
}

# Read the manifest line by line and sync each entry.
while IFS=$'\t' read -r local_path remote_dir remote_name; do
    [ -z "$local_path" ] && continue
    sync_one "$local_path" "$remote_dir" "$remote_name"
    echo ""
done <<< "$MANIFEST"

echo "=== Summary ==="
ssh "$LAMBDA" '
for p in /home/ubuntu/InterMimic/checkpoints/smplx_distill_both_normreward_v2/nn/mimic.pth \
         /home/ubuntu/InterMimic/checkpoints/smplx_distill_both_abl_v2/nn/mimic.pth \
         /home/ubuntu/InterMimic/checkpoints/smplx_distill_both_nobetas_normreward_v2/nn/mimic.pth \
         /home/ubuntu/InterMimic/checkpoints/smplx_student/student.pth; do
    if [ -f "$p" ]; then
        printf "  %-80s OK  (%s bytes)\n" "$p" "$(stat -c %s "$p")"
    else
        printf "  %-80s MISSING\n" "$p"
    fi
done'

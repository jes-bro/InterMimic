#!/bin/bash
# Fix checkpoint directories that got rsync'd with a doubly-nested
# directory structure instead of a flat .pth file. This happens when
# rsync's --mkpath flag treats the trailing filename as a directory to
# create, then drops the source file inside, sometimes nested twice
# depending on whether the source itself was renamed.
#
# Before (broken):
#   <checkpoint-dir>/nn/mimic.pth/<nested-dir>/mimic.pth        (real file)
#   <checkpoint-dir>/nn/mimic.pth/<nested-dir>/mimic_00002000.pth ...
#
# After (correct):
#   <checkpoint-dir>/nn/mimic.pth                               (real file)
#   <checkpoint-dir>/nn/mimic_00002000.pth                      ...
#
# Idempotent: skips checkpoints that are already in the correct shape.
# Preserves intermediate-epoch snapshots (mimic_NNNNNNNN.pth) by
# moving every file inside the nested dirs up to the parent nn/.
#
# Run on the machine that has the broken checkpoints (e.g. Lambda):
#   bash scripts/fix_nested_checkpoints.sh

set -u

# Edit this if your repo lives somewhere other than /home/ubuntu/InterMimic
REPO_ROOT=${REPO_ROOT:-/home/ubuntu/InterMimic}

# v2 distill student checkpoints — each lives at $REPO_ROOT/checkpoints/<name>/nn/mimic.pth
V2_DIRS=(
    smplx_distill_both_normreward_v2
    smplx_distill_both_abl_v2
    smplx_distill_both_nobetas_normreward_v2
)

# Used as the temp staging directory while we shuffle files out of the
# nested dirs, blow away the broken structure, and move them back.
STAGE=/tmp/fix_ckpt_recover_$$

fix_one() {
    local name=$1
    local nn=$REPO_ROOT/checkpoints/$name/nn
    echo "=== $name ==="

    if [ ! -d "$nn" ]; then
        echo "  SKIP: $nn does not exist"
        return
    fi

    if [ -f "$nn/mimic.pth" ]; then
        echo "  already a regular file — nothing to fix"
        ls -la "$nn/mimic.pth"
        return
    fi

    if [ ! -d "$nn/mimic.pth" ]; then
        echo "  SKIP: $nn/mimic.pth is neither file nor dir (?)"
        return
    fi

    # mimic.pth is a directory; recover the nested files.
    mkdir -p "$STAGE"

    # Try the doubly-nested layout first ($nn/mimic.pth/<dir>/<files>).
    # Fall back to singly-nested ($nn/mimic.pth/<files>) if no inner dir.
    if compgen -G "$nn/mimic.pth/*/*" > /dev/null; then
        echo "  recovering doubly-nested files"
        mv "$nn"/mimic.pth/*/* "$STAGE"/
    elif compgen -G "$nn/mimic.pth/*" > /dev/null; then
        echo "  recovering singly-nested files"
        mv "$nn"/mimic.pth/* "$STAGE"/
    else
        echo "  WARNING: mimic.pth is an empty directory; nothing to recover"
        rmdir "$nn/mimic.pth"
        return
    fi

    # Clear the broken nested directory and put files back at nn/ level.
    rm -rf "$nn/mimic.pth"
    mv "$STAGE"/* "$nn"/
    rmdir "$STAGE"

    echo "  recovered files now at $nn/:"
    ls -la "$nn"
}

fix_student() {
    local sp_dir=$REPO_ROOT/checkpoints/smplx_student
    local sp=$sp_dir/student.pth
    echo "=== smplx_student (student.pth baseline) ==="

    if [ ! -e "$sp" ] && [ ! -d "$sp" ]; then
        echo "  SKIP: $sp does not exist — needs to be rsync'd from laptop"
        return
    fi

    if [ -f "$sp" ]; then
        echo "  already a regular file — nothing to fix"
        ls -la "$sp"
        return
    fi

    # Same recovery pattern as fix_one but for the student dir.
    mkdir -p "$STAGE"
    if compgen -G "$sp/*/*" > /dev/null; then
        mv "$sp"/*/* "$STAGE"/
    elif compgen -G "$sp/*" > /dev/null; then
        mv "$sp"/* "$STAGE"/
    else
        echo "  WARNING: student.pth is an empty directory"
        rmdir "$sp"
        return
    fi
    rm -rf "$sp"
    # Restore — there should be a single student.pth in $STAGE, but if
    # the nesting copied other names, keep them all.
    mv "$STAGE"/* "$sp_dir"/
    rmdir "$STAGE"

    echo "  recovered:"
    ls -la "$sp_dir"
}

for d in "${V2_DIRS[@]}"; do
    fix_one "$d"
done
fix_student

echo ""
echo "=== Summary ==="
for d in "${V2_DIRS[@]}"; do
    p=$REPO_ROOT/checkpoints/$d/nn/mimic.pth
    if [ -f "$p" ]; then
        printf "  %-50s OK  (%s bytes)\n" "$d/nn/mimic.pth" "$(stat -c %s "$p")"
    else
        printf "  %-50s MISSING or still bad\n" "$d/nn/mimic.pth"
    fi
done
sp=$REPO_ROOT/checkpoints/smplx_student/student.pth
if [ -f "$sp" ]; then
    printf "  %-50s OK  (%s bytes)\n" "smplx_student/student.pth" "$(stat -c %s "$sp")"
else
    printf "  %-50s MISSING\n" "smplx_student/student.pth"
fi

#!/bin/bash
# Render qualitative comparisons for sub2 source motion, on held-out
# bodies sub1 and sub5, across both objects (woodchair and largetable).
# Loops through all 4 (body, object) combinations and calls the main
# render script for each. Outputs 5 videos × 4 combos = 20 mp4s.
#
# Sibling of render_qualitative_sub6.sh — identical loop, only difference
# is the SOURCE_SUB substitution. Use both scripts to get a full sub2/sub6
# qualitative comparison across the OOD body × object matrix.

set -u
cd "$(dirname "$0")/.."

# Backup the main render script's hard-coded vars so we can restore them.
SCRIPT=scripts/render_qualitative.sh
cp "$SCRIPT" "${SCRIPT}.bak"

# Force SOURCE_SUB=sub2 in the main script for these runs.
sed -i 's/^SOURCE_SUB=.*/SOURCE_SUB=sub2/' "$SCRIPT"

for BODY in sub1 sub5; do
    for OBJ in woodchair largetable; do
        echo ""
        echo "=================================================================="
        echo "=== Combo: TARGET_BODY=$BODY  OBJECT=$OBJ  (source = sub2) ==="
        echo "=================================================================="
        sed -i "s/^TARGET_BODY=.*/TARGET_BODY=$BODY/" "$SCRIPT"
        sed -i "s/^OBJECT=.*/OBJECT=$OBJ/" "$SCRIPT"
        bash "$SCRIPT"
    done
done

# Restore original script vars
mv "${SCRIPT}.bak" "$SCRIPT"
echo ""
echo "=== ALL 4 combos done. Restored render_qualitative.sh from backup. ==="
ls -lh /tmp/render_sub{1,5}_{woodchair,largetable}_*.mp4 2>/dev/null

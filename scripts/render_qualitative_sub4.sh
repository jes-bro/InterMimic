#!/bin/bash
# Render qualitative comparisons for sub4 SOURCE motion, on held-out
# bodies sub1 and sub5. Sibling of render_qualitative_sub2.sh /
# render_qualitative_sub6.sh -- same loop, only SOURCE_SUB differs.
#
# IMPORTANT: sub4 has NO woodchair/largetable clips (the siblings' objects).
# sub4's objects are monitor, whitechair, smallbox, suitcase -- so the object
# loop here uses sub4's ACTUAL objects. `monitor` is first because that's the
# clip whose source motion looked broken/snappy in replay; this is the run to
# eyeball for the snap (the per-combo render_sub*_monitor_source_sub4.mp4 is the
# KINEMATIC source playback -- if the snap is in the motion it shows there; the
# *_full_method.mp4 etc. show the trained student driving that same motion).
#
# Edit BODY / OBJ below to widen or narrow the matrix. Outputs 5 videos per
# (body, object) combo into /tmp/.

set -u
cd "$(dirname "$0")/.."

# Backup the main render script's hard-coded vars so we can restore them.
SCRIPT=scripts/render_qualitative.sh
cp "$SCRIPT" "${SCRIPT}.bak"

# Force SOURCE_SUB=sub4 in the main script for these runs.
sed -i 's/^SOURCE_SUB=.*/SOURCE_SUB=sub4/' "$SCRIPT"

for BODY in sub1 sub5; do
    for OBJ in monitor whitechair; do
        echo ""
        echo "=================================================================="
        echo "=== Combo: TARGET_BODY=$BODY  OBJECT=$OBJ  (source = sub4) ==="
        echo "=================================================================="
        sed -i "s/^TARGET_BODY=.*/TARGET_BODY=$BODY/" "$SCRIPT"
        sed -i "s/^OBJECT=.*/OBJECT=$OBJ/" "$SCRIPT"
        bash "$SCRIPT"
    done
done

# Restore original script vars
mv "${SCRIPT}.bak" "$SCRIPT"
echo ""
echo "=== ALL combos done. Restored render_qualitative.sh from backup. ==="
ls -lh /tmp/render_sub{1,5}_{monitor,whitechair}_*.mp4 2>/dev/null

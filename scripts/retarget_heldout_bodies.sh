#!/bin/sh
# retarget_heldout_bodies.sh -- add NEW BODIES to an existing per-body reference
# tree, so a trained policy can be scored on people it has never seen.
#
# WHY. A g3 task reads <tree>/<body>/<clip>.pt and REFUSES a missing (body, clip)
# pair. So evaluating on a body means that body must have a reference for every
# clip of every source it will be scored against -- for the OMOMO students that
# is 13 sources x ~3,356 clips per body. This runs those solves.
#
# ADDITIVE AND RESUMABLE. Output goes into the SAME tree the students already
# use; each solve writes a new <body>/ directory and touches no existing file.
# Finished pairs are skipped, so a re-run after an interruption costs a directory
# scan, and a run alongside a live eval is safe (it only adds files that eval is
# not reading).
#
# NOT AN ABLATION TREE: --allow-worse-cm stays 0, so a solve that came out worse
# than not retargeting is refused and reported. Here a regression means an
# under-converged solve (raise ITERS), not a property under test -- the opposite
# of scripts/retarget_bball7_local.sh --w-contact 0.
#
# Usage (repo root, conda env with torch; CPU only, no GPU):
#   BODIES="sub300 sub301" sh scripts/retarget_heldout_bodies.sh
#   DRY=1 BODIES="sub300" sh scripts/retarget_heldout_bodies.sh      # plan only
#   WORKERS=128 BODIES="sub300 ... sub309" nohup sh scripts/retarget_heldout_bodies.sh \
#       > retarget-hodome.log 2>&1 &
#
# Env: BODIES (required), SOURCES (default the 13 OMOMO sources), TREE, MOTION_DIR,
#      ITERS (300), WORKERS (default nproc-2, capped at 128), ALLOW_WORSE_CM (0), DRY.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

OMOMO13="sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17"
BODIES="${BODIES:?set BODIES, e.g. BODIES=\"sub300 sub301\"}"
SOURCES="${SOURCES:-$OMOMO13}"
TREE="${TREE:-InterAct/OMOMO_retarget_contact_srcall13}"
MOTION_DIR="${MOTION_DIR:-InterAct/OMOMO_new}"
ITERS="${ITERS:-300}"
ALLOW_WORSE_CM="${ALLOW_WORSE_CM:-0}"
# Each worker is a process with torch loaded (~1.5G). 128 x 1.5G = ~192G, fine on
# a 503G box; the cap stops `nproc` on a 255-core machine from spawning 253 and
# taking the machine down with it.
DEFAULT_W=$(( $(nproc) - 2 )); [ "$DEFAULT_W" -gt 128 ] && DEFAULT_W=128
WORKERS="${WORKERS:-$DEFAULT_W}"
ASSETS=isaacgym/src/intermimic/data/assets

[ -d "$TREE" ] || { echo "ERROR: no tree $TREE" >&2; exit 2; }
[ -d "$MOTION_DIR" ] || { echo "ERROR: no motion dir $MOTION_DIR" >&2; exit 2; }

# Every target body needs its MJCF, under the name humanoid.py builds
# (smplx/smplx_omomo_<body>.xml) regardless of which dataset the person is from.
# Checking all of them up front beats discovering it 9 sources in.
for b in $BODIES; do
  [ -f "$ASSETS/smplx/smplx_omomo_$b.xml" ] || {
    echo "ERROR: no MJCF $ASSETS/smplx/smplx_omomo_$b.xml for body $b" >&2
    echo "       generate it first: scripts/generate_per_subject_mjcfs.py (cluster only)" >&2
    exit 2; }
done

n_clips=$(ls "$MOTION_DIR"/*.pt 2>/dev/null | wc -l)
n_bodies=$(echo "$BODIES" | wc -w)
echo "== add bodies to $TREE"
echo "   bodies  : $BODIES  ($n_bodies)"
echo "   sources : $SOURCES"
echo "   motion  : $MOTION_DIR ($n_clips clips total)"
echo "   workers : $WORKERS   iters: $ITERS   allow-worse: $ALLOW_WORSE_CM cm"
for b in $BODIES; do
  have=$(ls "$TREE/$b" 2>/dev/null | wc -l)
  echo "   $b: $have reference file(s) already present"
done

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

for s in $SOURCES; do
  n=$(ls "$MOTION_DIR"/${s}_*.pt 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "ERROR: no ${s}_* clips in $MOTION_DIR" >&2; exit 2; }
  echo "== source $s ($n clips) -> $n_bodies bodies"
  python3 -u scripts/retarget_contact.py --batch \
      --motion-dir "$MOTION_DIR" --source "$s" \
      --targets $BODIES --iters "$ITERS" --workers "$WORKERS" \
      --allow-worse-cm "$ALLOW_WORSE_CM" --out-dir "$TREE"
done

echo "== done."
for b in $BODIES; do
  echo "   $b: $(ls "$TREE/$b" 2>/dev/null | wc -l) reference files"
done
echo "   Each body needs one file per clip of the sources it will be scored on."
echo "   A short count means solves were REFUSED (see the FAILURES blocks above):"
echo "   raise ITERS and re-run -- finished pairs are skipped."

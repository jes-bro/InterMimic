#!/bin/sh
# Emit the list of (subject, object) pairs that actually have motion clips, one
# "sub<N> <object>" per line -- this is the array-task list for slurm_render_clips.sh.
#
# Derived from the motion filenames themselves (sub<N>_<object>_<idx>.pt), so we
# never launch a job for a pair with no clips, and we never miss one that exists.
#
#   sh scripts/gen_render_pairs.sh                      # -> render_pairs.txt
#   MOTION_DIR=InterAct/OMOMO OUT_FILE=p.txt sh scripts/gen_render_pairs.sh
set -eu
REPO="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO"

# Must match motion_file in isaacgym/src/intermimic/data/cfg/omomo_render.yaml.
MOTION_DIR="${MOTION_DIR:-InterAct/OMOMO_new}"
OUT_FILE="${OUT_FILE:-render_pairs.txt}"

[ -d "$MOTION_DIR" ] || { echo "FATAL: motion dir '$MOTION_DIR' not found" >&2; exit 1; }

# sub2_largetable_000.pt -> "sub2 largetable". Sorted by subject number then object
# so the array order is stable and readable.
ls "$MOTION_DIR" \
  | sed -n 's/^\(sub[0-9]\{1,\}\)_\(.*\)_[0-9]\{1,\}\.pt$/\1 \2/p' \
  | sort -u -k1.4n -k2 \
  > "$OUT_FILE"

n=$(wc -l < "$OUT_FILE")
[ "$n" -gt 0 ] || { echo "FATAL: no sub<N>_<object>_<idx>.pt files in '$MOTION_DIR'" >&2; exit 1; }

clips=$(ls "$MOTION_DIR" | grep -c '\.pt$' || true)
echo "[pairs] $MOTION_DIR: $clips clips -> $n (subject, object) pairs -> $OUT_FILE"
echo "[pairs] submit with:"
echo "    sbatch --array=0-$((n - 1))%16 slurm_render_clips.sh"

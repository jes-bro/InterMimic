#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --job-name="bball-retarget"
#SBATCH --output=bball-retarget-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Retarget the CARI4D bball clip onto one or more OTHER bodies. CPU-only: no
# GPU, no Isaac Gym, pure torch -- so it queues fast and does not compete with
# the training arms for accelerators. Deliberately NOT run on the login node,
# where it contends with rclone/sftp and takes far longer than the solve needs.
#
#   sbatch slurm_cari4d_bball_retarget.sh                      # sub2
#   TARGETS="sub2 sub10" sbatch slurm_cari4d_bball_retarget.sh
#   TARGETS="sub10" ITERS=150 sbatch slurm_cari4d_bball_retarget.sh
#
# Output layout is one FLAT directory per target,
#   InterAct/behave_cari4d_optj3d_cf2_<target>/sub100_bball_000.pt
# which is what the single-body arms (r12_sub2_ret and friends) point
# motion_file at. The clip KEEPS its sub100_ prefix on purpose: dataSub matches
# on the filename, and robotType is what selects the body.
#
# --source-mjcf IS LOAD-BEARING. retarget_contact.py resolves a subject id to
# smplx_omomo_<id>.xml, and smplx_omomo_sub100.xml is a SYNTHETIC OMOMO body,
# not the CARI4D bball subject (smplh_behave_sub100.xml). Same number, different
# person. Without the flag this retargets from the wrong body and reports
# nothing amiss.
#
# READ THE VERDICT AT THE END. An under-converged solve can be WORSE than not
# retargeting at all, and training on that hands the policy a reference worse
# than the original. Any target whose contact error did not fall is reported as
# FAILED and its output should not be trained on -- raise ITERS and redo it.

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
# No LD_LIBRARY_PATH and no Isaac Gym: this is torch on CPU only.
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

TARGETS="${TARGETS:-sub2}"
ITERS="${ITERS:-300}"
CLIP="${CLIP:-InterAct/behave_cari4d_optj3d_cf2/sub100_bball_000.pt}"
SOURCE="${SOURCE:-sub100}"
SOURCE_MJCF="${SOURCE_MJCF:-isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml}"
OUT_PREFIX="${OUT_PREFIX:-InterAct/behave_cari4d_optj3d_cf2}"

[ -f "$CLIP" ] || { echo "[bball-retarget] ERROR: no clip at $CLIP" >&2; exit 1; }
[ -f "$SOURCE_MJCF" ] || { echo "[bball-retarget] ERROR: no source MJCF at $SOURCE_MJCF -- this is the CARI4D subject's body and is NOT in git" >&2; exit 1; }

echo "[bball-retarget] host=$(hostname) job=$SLURM_JOB_ID"
echo "[bball-retarget] clip=$CLIP"
echo "[bball-retarget] source=$SOURCE via $SOURCE_MJCF"
echo "[bball-retarget] targets='$TARGETS' iters=$ITERS"
echo

LOG=/tmp/bball_retarget_$$.log
: > "$LOG"
for T in $TARGETS; do
    MJCF="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${T}.xml"
    if [ ! -f "$MJCF" ]; then
        echo "[bball-retarget] SKIP $T: no MJCF at $MJCF" | tee -a "$LOG"
        continue
    fi
    OUT="${OUT_PREFIX}_${T}"
    echo "=== $SOURCE -> $T  ->  $OUT ==="
    python3 -u scripts/retarget_contact.py \
        --clip "$CLIP" \
        --source "$SOURCE" --source-mjcf "$SOURCE_MJCF" \
        --target "$T" --iters "$ITERS" \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG"
    echo
done

echo "================================ VERDICT ================================"
# The solve is only useful if it REDUCED contact error. Parse the numbers the
# script printed rather than trusting exit status: retarget_contact exits 0 even
# when the result is worse, and a worse reference is the failure that matters.
python3 - "$LOG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
pairs = re.findall(r"\[retarget\] \S+\s+(\S+) -> (\S+)", text)
errs = re.findall(r"contact err ([\d.]+) -> ([\d.]+) cm", text)
if not pairs or len(pairs) != len(errs):
    print("  could not parse results -- read the log above by hand")
    raise SystemExit(1)
ok = True
for (src, tgt), (before, after) in zip(pairs, errs):
    b, a = float(before), float(after)
    good = a < b
    ok &= good
    print(f"  {src} -> {tgt:>8}: {b:6.2f} -> {a:6.2f} cm   "
          f"[{'PASS' if good else 'FAILED: did not improve -- raise ITERS'}]")
print()
print("  ALL GOOD -- safe to train on" if ok else
      "  SOME FAILED -- do NOT train on those; a worse reference is worse than none")
raise SystemExit(0 if ok else 1)
PY
rm -f "$LOG"

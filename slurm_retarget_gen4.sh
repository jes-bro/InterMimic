#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-12
#SBATCH --job-name="rt-gen4"
#SBATCH --output=rt-gen4-%A_%a.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=FAIL,END

# ONE submission: contact-retargeted references for the gen4 bodies (sub600-629)
# plus sub4, for all 13 OMOMO teacher sources, as a slurm array (one task per
# source). Each task is slurm_retarget_gen.sh's job with SOURCE / TARGETS / OUT
# fixed: writes InterAct/OMOMO_retarget_contact_src<S>/<body>/<clip>.pt into the
# EXISTING per-source tree (additive, resumable) and prints the measured
# contact-error verdict per body. CPU only, no Isaac Gym. sub4 is retargeted so
# it can be EVALUATED on retargeted arms (it never trains).
#
#   sbatch --exclude=simurgh6,simurgh2 slurm_retarget_gen4.sh
#   sbatch --array=0,4 ... slurm_retarget_gen4.sh        # only sources index 0 and 4
# Then the gen4 teacher launchers' guard (every body dir present) passes.
set -u
SOURCES=(sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17)
SOURCE="${SOURCES[$SLURM_ARRAY_TASK_ID]}"
TARGETS="sub4 $(seq -f sub%g 600 629 | tr '\n' ' ')"
OUT="InterAct/OMOMO_retarget_contact_${SOURCE/sub/src}"
ITERS="${ITERS:-300}"
ALLOW_WORSE_CM="${ALLOW_WORSE_CM:-0}"

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 2
[ -d "$OUT" ] || { echo "[rt-gen4] ERROR: $OUT does not exist -- the source's original retarget tree must be there" >&2; exit 2; }
for b in $TARGETS; do
    [ -f "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${b}.xml" ] || { echo "[rt-gen4] ERROR: no MJCF for $b" >&2; exit 2; }
done
echo "[rt-gen4] task $SLURM_ARRAY_TASK_ID source=$SOURCE targets=$TARGETS iters=$ITERS -> $OUT  $(date)"

python3 scripts/retarget_contact.py --batch --source "$SOURCE" \
    --targets $TARGETS --iters "$ITERS" --allow-worse-cm "$ALLOW_WORSE_CM" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" --out-dir "$OUT"

echo
echo "================ VERDICT (from the ACTUAL measured errors) ================="
python3 - "$OUT/retarget_summary.json" "$SOURCE" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))["summary"]
src = sys.argv[2]
ok = bool(summary)
for body, s in sorted(summary.items()):
    b, a, n = s["before_cm"], s["after_cm"], s["n"]
    if body == src:
        passed, why = a < 0.05, "identity no-op (<0.05cm)"
    else:
        passed, why = (a < b and a < 1.0), "reduced & <1cm"
    ok &= passed
    print(f"  {body:>8}: {b:6.2f} -> {a:6.2f} cm  over {n} clips   [{'PASS' if passed else 'FAIL'}: {why}]")
if not summary:
    print("  FAIL: no results")
print(f"\n  {'GENERATION OK -- data ready' if ok else 'GENERATION FAILED -- do not train on this'}")
sys.exit(0 if ok else 1)
PY

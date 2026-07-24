#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

#SBATCH --job-name="rt-gen"
#SBATCH --output=rt-gen-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# CPU-ONLY generation of per-body contact-retargeted references (no GPU, no Isaac
# Gym). Writes OUT/<body>/<clip>.pt, resumable, then prints a per-body verdict from
# the ACTUAL measured contact error (identity source==target must be a no-op; every
# other body must actually drop). Read rt-gen-<jobid>.out.
#
# Defaults = the SMOKE set (source sub2 -> 3 training bodies). Override via env vars:
#   SMOKE  (default): sbatch slurm_retarget_gen.sh
#   FULL src2:  SOURCE=sub2 TARGETS_FROM=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug.yaml \
#               OUT=InterAct/OMOMO_retarget_contact_src2 sbatch slurm_retarget_gen.sh
#   FULL src6:  SOURCE=sub6 TARGETS_FROM=isaacgym/src/intermimic/data/cfg/omomo_teacher_src6_xf_aug.yaml \
#               OUT=InterAct/OMOMO_retarget_contact_src6 sbatch slurm_retarget_gen.sh
# TARGETS_FROM reads subjectBodies from a cfg (excludes held-out); or list TARGETS
# by hand. Runs from repo root.

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
# No LD_LIBRARY_PATH / no Isaac Gym -- pure torch CPU.
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="${SOURCE:-sub2}"
OUT="${OUT:-InterAct/OMOMO_retarget_contact_smoke}"
ITERS="${ITERS:-300}"

# Body set: TARGETS_FROM a cfg (preferred; excludes held-out) or a hand list.
if [ -n "${TARGETS_FROM:-}" ]; then
    TARG_ARGS=(--targets-from "$TARGETS_FROM")
    echo "[rt-gen] targets from cfg: $TARGETS_FROM"
else
    TARGETS="${TARGETS:-sub2 sub6 sub9}"
    TARG_ARGS=(--targets $TARGETS)
    echo "[rt-gen] targets: $TARGETS"
fi

echo "[rt-gen] host=$(hostname) job=$SLURM_JOB_ID  source=$SOURCE iters=$ITERS -> $OUT"
echo

python3 scripts/retarget_contact.py --batch --source "$SOURCE" \
    "${TARG_ARGS[@]}" --iters "$ITERS" \
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
    if body == src:                       # identity retarget: must be a no-op
        passed, why = a < 0.05, "identity no-op (<0.05cm)"
    else:                                 # cross-body: must reduce contact error
        passed, why = (a < b and a < 1.0), "reduced & <1cm"
    ok &= passed
    print(f"  {body:>8}: {b:6.2f} -> {a:6.2f} cm  over {n} clips   "
          f"[{'PASS' if passed else 'FAIL'}: {why}]")
if not summary:
    print("  FAIL: no results")
print(f"\n  {'GENERATION OK -- data ready' if ok else 'GENERATION FAILED -- do not train on this'}")
sys.exit(0 if ok else 1)
PY

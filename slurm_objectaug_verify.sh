#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="oa-verify"
#SBATCH --output=oa-verify-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Validation for the objectAug machinery -- runs WITHOUT the fold-in checkpoint
# (trains from scratch for a few minutes, then self-terminates). With
# OBJECTAUG_DEBUG=1 (set below) the env emits the exact verify numbers:
#   [masschk]  total object mass per env -- should be ~CONSTANT across differing
#              aug if hold-mass works (scales with aug**3 if the correction failed)
#   [posechk]  pose error for just-reset envs -- should be ~0 (sim is state-init'd
#              to the reference); large => sim/ref dof orderings are misaligned
# It also confirms the plumbing: config loads, env/network build, the
# "objectAug ON"/"reward structure" lines are right, and mean_rewards is finite
# (not NaN / not stuck at 0) with epoch_num advancing (no instant resets).
# Does NOT exercise warm-start loading (no checkpoint yet) -- that's checked when
# you launch the real runs against the finished fold-in policy.
#
# Submit from the repo root:
#   sbatch slurm_objectaug_verify.sh                 # smokes drop_both
#   VARIANT=keep_both sbatch slurm_objectaug_verify.sh
# numEnvs comes from the config (4096); edit the config's numEnvs to shrink it.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
export OBJECTAUG_DEBUG=1   # turns on [masschk]/[posechk]; no effect without it

VARIANT="${VARIANT:-drop_both}"
CFG_ENV="isaacgym/src/intermimic/data/cfg/omomo_objectaug_${VARIANT}.yaml"
RUN_SECONDS="${RUN_SECONDS:-300}"   # self-terminate after this; prints land in <1 min

WORK=objectaug_work/verify
mkdir -p "$WORK"

# Build a FROM-SCRATCH train config (resume_from None) so no checkpoint is
# needed. Reuses curriculum_runner.TRAIN_TMPL -- the same MLP arch as the real
# runs -- so this also proves the network builds for these observations.
python - "$WORK" <<'PY'
import sys
sys.path.insert(0, 'scripts')
from curriculum_runner import TRAIN_TMPL
work = sys.argv[1]
open(f"{work}/train_verify.yaml", "w").write(TRAIN_TMPL.format(
    stage="verify", active="verify", exp_name="smplx_objectaug_verify",
    save_frequency=100000, resume_from="'None'", mask_dead_envs="false"))
print(f"[verify] wrote {work}/train_verify.yaml (from scratch, no checkpoint)")
PY

LOG="$WORK/verify_run.log"
echo "[verify] variant=$VARIANT  env cfg=$CFG_ENV"
echo "[verify] running ${RUN_SECONDS}s from scratch with OBJECTAUG_DEBUG=1..."
timeout "${RUN_SECONDS}" python -u -m intermimic.run --task InterMimic \
    --cfg_env "$CFG_ENV" --cfg_train "$WORK/train_verify.yaml" \
    --headless --output checkpoints 2>&1 | tee "$LOG"

echo
echo "================ VERIFY SUMMARY ($VARIANT) ================"
echo "--- [1] mass-hold: total_mass should be ~equal across differing aug ---"
grep "\[masschk\]" "$LOG" | head -6 || echo "  NONE -> objectAug/debug not active"
echo "--- [2/3] dof align + lambda: err should be ~0 for fresh envs ---"
grep "\[posechk\]" "$LOG" | head -4 || echo "  NONE -> pose term off / debug not active"
echo "--- plumbing: objectAug + reward-structure lines ---"
grep -m1 "objectAug ON" "$LOG" || echo "  MISSING objectAug ON line!"
grep -m1 "reward structure" "$LOG" || echo "  MISSING reward structure line!"
echo "--- reward sanity: finite, non-zero, epoch_num advancing ---"
grep "epoch_num:" "$LOG" | tail -5 || echo "  NONE -> never reached an epoch (inspect $LOG)"
echo "=========================================================="

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-src2_nvadlr_me3"
#SBATCH --output=teacher-src2_xf_aug_normval_adlr_me3-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# mini_epochs=3 arm: warm-started from the _normval_adlr control's epoch-21400
# checkpoint, ONE knob changed (mini_epochs 6 -> 3). Same env cfg (_lowbuf), same
# normalize_value/lr_schedule/kl_threshold, so the two curves share a start point
# and differ in exactly one thing.
#
# The question: at epoch 21400 the control's adaptive-LR controller has throttled
# last_lr to 7.59e-6 (below its 2e-5 start, 7.6x below its 5.77e-5 peak) because
# kl is pinned at kl_threshold 0.06 with clip_frac 0.47-0.57. Do fewer inner
# epochs cut per-epoch policy drift enough that the controller RAISES lr instead
# of cutting it -- and does the reward slope steepen? Also cuts the update phase,
# which is 34% of wall clock.
#
# READ AFTER ~1500 EPOCHS: info/last_lr (climbing toward 5e-5?), info/clip_frac
# (falling?), reward slope vs the control over the same epoch range.
#
# Runs from repo root. Saves to
# checkpoints/smplx_teacher_src2_xf_aug_normval_adlr_me3/nn/ -- NOT the control's
# dir. The control run is never written to; its checkpoint is only read as a seed.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Env cfg is _lowbuf -- the SAME one the control arm is currently running, so the
# env side is held fixed and mini_epochs is the only difference between them.
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_lowbuf.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_normval_adlr_me3.yaml
echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"

echo "[teacher] NORMVAL+ADLR mini_epochs=3 (warm-start arm): source=sub2 x 13 real + 40 synthetic bodies, neutral betas + body-norm + pose, normalize_value=True, lr_schedule=adaptive"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_src2_xf_aug_normval_adlr_me3/nn/"

# --- Warm-start + auto-resume. Two DIFFERENT sources, in strict priority order:
#   1. this arm's OWN latest checkpoint, once it has one (normal requeue path)
#   2. otherwise the SEED: the control arm's epoch-21400 checkpoint, read-only
# The stock block below could not express this -- its sed only matches
# resume_from: 'None', so putting the seed path in the yaml would make EVERY
# requeue reload the seed and silently discard this arm's own progress.
#
# ISOLATION (this arm must not touch the control run):
#   * writes go to checkpoints/${EXP}/nn/ where EXP ends in _me3 -- a different
#     dir from the control's. Asserted below.
#   * the seed is only ever READ (rl_games loads it at agent-init; nothing in this
#     job writes to the control's dir).
#   * a missing seed is a hard ERROR, not a fresh start: silently training from
#     scratch would produce a curve that is not comparable to the control and
#     would waste days before anyone noticed.
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CTRL_EXP=smplx_teacher_src2_xf_aug_normval_adlr
SEED="checkpoints/${CTRL_EXP}/nn/mimic_00021400.pth"
CKPT="checkpoints/${EXP}/nn/mimic.pth"

case "$EXP" in
    *_me3) : ;;
    *) echo "[teacher] ERROR: full_experiment_name '$EXP' is not the _me3 arm -- refusing to run" >&2
       echo "[teacher]        (this script would then write into another arm's checkpoint dir)" >&2
       exit 2 ;;
esac
if [ "$EXP" = "$CTRL_EXP" ]; then
    echo "[teacher] ERROR: this arm resolves to the CONTROL's experiment name -- refusing to run" >&2
    exit 2
fi

if [ -f "$CKPT" ]; then
    START="$CKPT"
    echo "[teacher] RESUMING from this arm's own checkpoint: ${START}"
elif [ -f "$SEED" ]; then
    START="$SEED"
    echo "[teacher] WARM-START from the control arm's seed (read-only): ${START}"
else
    echo "[teacher] ERROR: no own checkpoint at ${CKPT} and no seed at ${SEED}" >&2
    echo "[teacher]        Refusing to start from scratch -- that would not be comparable" >&2
    echo "[teacher]        to the control. Check the seed path (the control saves every" >&2
    echo "[teacher]        100 epochs, so pick whichever mimic_000NNNNN.pth exists)." >&2
    exit 2
fi
RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
sed "s|resume_from: 'None'|resume_from: '${START}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
if ! grep -q "resume_from: '${START}'" "$RESUME_TRAIN"; then
    echo "[teacher] ERROR: resume_from rewrite did not take (expected the literal" >&2
    echo "[teacher]        \"resume_from: 'None'\" in $CFG_TRAIN). Refusing to run untethered." >&2
    exit 2
fi
CFG_TRAIN="$RESUME_TRAIN"

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

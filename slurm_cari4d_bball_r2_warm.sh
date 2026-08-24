#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r2_warm"
#SBATCH --output=cari4d-bball-r2_warm-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R2_WARM bball experiment (see f7826fb). Round 2 after the fully-resets-off
# noreset arm: keep the OBJECT-side divergence resets off (object / igRatio /
# contactSteps false -- they re-create the free-flight wall on imperfect releases)
# but RESTORE the human divergence reset at 0.5m, which is what executes the crawl
# exploit that resets-off enabled. Bundles the post-mortem fixes: initVel True (all
# 39 bball cfgs had inherited initVel False, zeroing frame-0 object velocity so
# every spawn dropped a dead ball) and the relabeled contact-flag data
# (InterAct/behave_cari4d_optj3d_cf). PSI key dropped -- proven inert on short clips.
#
# SEPARATE experiment: own cfgs, own checkpoint dir
# (checkpoints/smplx_cari4d_bball_r2_warm/nn/) -- writes NOTHING into the
# original run's directory.
#
# NOT a fresh start: the train cfg carries an EXPLICIT, Jess-approved warm start
# from checkpoints/smplx_teachers_new/sub2.pth (resume_from, read at
# intermimic_agent.py:177 regardless of load_checkpoint). Once this run has its own
# mimic.pth, resubmits resume from that instead -- never from another run.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r2_warm_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r2_warm_train.yaml

# Guard: this experiment IS the resets-off arm -- refuse to run without the block.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r2_warm] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
# r2 inverts the family default: the HUMAN divergence reset is restored (0.5m) to
# execute the crawl exploit that the fully-resets-off arms enabled. Object-side
# resets stay off, so a cloned 'human: false' would silently make this a repeat
# of noreset -- refuse to run unless the threshold is actually there.
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r2_warm] ERROR: human reset not set to 0.5 in $CFG_ENV -- restoring it IS the r2_warm knob" >&2; exit 1
fi
# The other half of the arm: the object-side resets must stay OFF. If one gets
# flipped back on, the free-flight wall returns on imperfect releases and the run
# is no longer this experiment -- fail loudly rather than train the wrong thing.
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r2_warm] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r2_warm] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r2_warm] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r2_warm/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting (Jess rule 2026-08-11: fresh start only; resuming is
# permitted ONLY from this run's own checkpoints, for walltime resubmits).
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r2_warm] RESUMING own run from ${CKPT}"
else
    echo "[bball-r2_warm] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved)"
fi
if [ -n "$RESUME_FROM" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    sed "s|resume_from:.*|resume_from: '${RESUME_FROM}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    CFG_TRAIN="$RESUME_TRAIN"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

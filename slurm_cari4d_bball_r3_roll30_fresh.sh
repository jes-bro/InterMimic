#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r3_roll30_fresh"
#SBATCH --output=cari4d-bball-r3_roll30_fresh-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R3_ROLL30_FRESH -- the WARM-START CONTROL for r3_roll30. Shares r3_roll30's env cfg
# verbatim; the ONLY difference is resume_from: 'None' in the rlg cfg.
#
# WHY. Every bball arm since r2 warm starts from
# checkpoints/smplx_teachers_new/sub2.pth. That is InterMimic's OWN default,
# inherited from omomo.yaml down the config lineage -- nobody chose it for this
# task. The teacher's entire experience is subject 2 doing ~0.5 m/s tabletop
# manipulation. This clip is 1.98 m/s median and 7.33 m/s peak, on a different
# body, with intermittent contact.
#
# THE EVIDENCE THAT MOTIVATES IT. rectinj3 and rectinj3_warm are a matched pair
# -- identical env cfgs, rlg differing only in resume_from. At comparable
# own-epochs:
#     rectinj3       (fresh)   9,944 own epochs   mean_rewards 1.48
#     rectinj3_warm            ~10,629 own        mean_rewards 0.57
# The warm run trained LONGER and scored 2.6x worse. But that pair ran on
# behave_cari4d_rectinj3, and warm r3 reached 1.47 on optj3d_cf while warm
# rectinj3 reached 0.57 -- so the dataset matters and the result does not
# transfer on its own. This arm tests it on the data actually in use.
#
# NOTE the epoch counter starts at 0 here, not ~13,005. The warm-started arms
# inherit the teacher's epoch number, so compare on SIM STEP or on own-epochs
# (absolute minus 13,005), never on the raw counter.
#
# HOW TO READ IT. Against r3_roll30 at the same sim step -- one knob, so a difference
# is attributable. These are progress reads, not a stopping rule.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r3_roll30_fresh-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS" cari4d-bball-r3_roll30_fresh-*.out | tail -16
#
#
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r3_roll30_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r3_roll30_fresh_train.yaml

# Guard: rolloutLength IS this arm. Cloned from r2_warm, so the one thing that
# can silently revert it is the inherited 300 -- which would make this an exact
# duplicate of a run we already know returns completed 0.0%.
if ! grep -qE '^\s*rolloutLength:\s*30\b' "$CFG_ENV"; then
    echo "[bball-r3_roll30_fresh] ERROR: rolloutLength not 30 in $CFG_ENV -- the short rollout IS the experiment" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 30 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 30 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r3_roll30_fresh] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 30 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r3_roll30_fresh] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 30 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r3_roll30_fresh] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r3_roll30_fresh] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r3_roll30_fresh] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r3_roll30_fresh] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r3_roll30_fresh] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r3_roll30/nn/"

# Guard: this arm IS the absence of a warm start. An inherited rlg cfg would
# silently duplicate r3_roll30 under a different name.
if grep -qE "^\\s*resume_from:\\s*checkpoints/" "$CFG_TRAIN"; then
    echo "[bball-r3_roll30_fresh] ERROR: $CFG_TRAIN sets a checkpoint warm start -- this arm is the CONTROL and must have resume_from: 'None' (otherwise it duplicates r3_roll30)" >&2; exit 1
fi
# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r3_roll30_fresh] RESUMING own run from ${CKPT}"
else
    echo "[bball-r3_roll30_fresh] first launch: FRESH START -- no warm start. This arm IS the control for the inherited sub2 teacher init."
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

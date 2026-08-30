#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r11_sub2_plain"
#SBATCH --output=cari4d-bball-r11_sub2_plain-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R11_SUB2_PLAIN -- CROSS-BODY, NO RETARGETING. r8_horiz's recipe with one
# knob: robotType. sub2's body is asked to track a reference shaped like the
# CARI4D bball subject.
#
# WHY. Every bball arm so far has been identity: the reference is sub100's body
# and the policy drives sub100's body. This is the first cross-body arm on the
# fast task, and it is the BASELINE half of a pair -- the condition retargeting
# is supposed to fix. sub2 is 1.669 m at rest; the bball subject is a different
# build entirely (smplh_behave_sub100.xml, not the synthetic
# smplx_omomo_sub100.xml, which is a DIFFERENT body with the same number).
#
# The skeletons are compatible -- 52 bodies, 154 joints, 153 actions -- so only
# the proportions differ. Nothing else changes: geometric reward, six obs
# horizons, _cf2 labels, rolloutLength 50, fresh start.
#
# READ IT AGAINST r12_sub2_ret at the same sim step. Those two differ in the
# reference alone, so a difference between them IS the retargeting.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r11_sub2_plain-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS" cari4d-bball-r11_sub2_plain-*.out | tail -16
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r11_sub2_plain_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r11_sub2_plain_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r11_sub2_plain] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: rewardShape MUST be geometric in the train cfg -- it IS the experiment,
# and an inherited r6 cfg would silently duplicate a run already in flight.
if ! grep -qE '^\s*rewardShape:\s*geometric\b' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: rewardShape is not 'geometric' in $CFG_ENV -- the reward shape IS this experiment (absent = a duplicate of r6_cf2)" >&2; exit 1
fi
# Guard: the BODY is the experiment. An inherited r8 cfg still names the bball
# subject and would silently duplicate a run already in flight.
if ! grep -qE '^\s*robotType:\s*"smplx/smplx_omomo_sub2\.xml"' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: robotType is not smplx_omomo_sub2.xml in $CFG_ENV -- the cross-body swap IS this experiment" >&2; exit 1
fi
# Guard: the reference must stay the ORIGINAL. Pointed at the retargeted dir this
# would be r12, not its baseline.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: motion_file is not the unretargeted behave_cari4d_optj3d_cf2 -- this arm is the NO-retargeting baseline" >&2; exit 1
fi
# Guard: obsHorizons IS the experiment. An inherited r7 cfg has none, and would
# silently duplicate a run already in flight under a different name.
if ! grep -qE '^\s*obsHorizons:\s*\[' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: no obsHorizons in $CFG_ENV -- the horizon set IS this experiment (absent = a duplicate of r7_geom)" >&2; exit 1
fi
# Guard: numObs must match the horizon count (1599 per horizon, no betas here).
# A mismatch is a silent shape bug that surfaces deep inside rl_games.
NH=$(grep -oE '^\s*obsHorizons:\s*\[[^]]*\]' "$CFG_ENV" | tr ',' '\n' | wc -l)
NOBS=$(grep -oE '^\s*numObs:\s*[0-9]+' "$CFG_ENV" | grep -oE '[0-9]+')
if [ "$NOBS" != "$((NH * 1599))" ]; then
    echo "[bball-r11_sub2_plain] ERROR: numObs=$NOBS but $NH horizons x 1599 = $((NH * 1599)) in $CFG_ENV" >&2; exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r11_sub2_plain] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r11_sub2_plain] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r11_sub2_plain] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r11_sub2_plain] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r11_sub2_plain/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r11_sub2_plain] RESUMING own run from ${CKPT}"
else
    echo "[bball-r11_sub2_plain] first launch: FRESH START (no warm start possible -- numObs 9594 vs the sub2 teacher's 3198)"
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

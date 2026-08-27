#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r8_horiz"
#SBATCH --output=cari4d-bball-r8_horiz-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R8_HORIZ -- the OBSERVATION-HORIZON arm. Same as bball-r7_geom except
# env.obsHorizons ([1,16] -> [1,4,7,10,13,16]) and the numObs that follows from
# it (3198 -> 9594).
#
# WHY. The failure is a TRANSITION, not a skill: the arms learn the dribble and
# the jump, and cannot stitch them. intermimic.py's MLP policy sees the
# reference at delta_t 1 and delta_t 16 and NOTHING BETWEEN -- a 15-frame blind
# gap. At 30fps, 16 frames is 0.53s. On OMOMO, which those defaults were tuned
# on, that is a small slice of a slow reach. On this layup it spans the whole
# crouch-to-release, so the entire countermovement happens where the policy
# cannot see it. A jump has to be preloaded; the observation never shows the
# preload.
#
# UNIFORM, NOT TUNED. The horizons keep r7's endpoints (1 and 16) and fill
# between them at a constant stride of 3. Geometric spacing would have fit this
# clip's timing better and would have been a guess about one motion; uniform is
# defensible applied unchanged to a whole dataset, which is where this has to go
# if it works.
#
# TWO DIFFERENCES FROM r7, ONE OF THEM FORCED. numObs changes 3198 -> 9594, so
# the sub2 TEACHER warm start every arm since r2 has used CANNOT load (shape
# mismatch on the first layer). r8 therefore starts FRESH. That is not a free
# choice and it is a real confound: a worse r8 could be the missing warm start
# rather than the horizons. If r8 underperforms, the control to run before
# concluding anything is r7 with resume_from None and nothing else changed.
#
# HOW TO READ IT. Compare against r7_geom at the SAME sim step -- but r8 starts
# FRESH (see the confound above) and a fresh start climbs more slowly, so early
# gaps are expected and mean little. These are progress reads, not a stopping
# rule.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r8_horiz-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS" cari4d-bball-r8_horiz-*.out | tail -16
#
# NOTE the eval twin DOES move here, unlike r7's. obsHorizons changes the
# observation the network consumes, so a policy trained on 6 horizons cannot be
# evaluated through a 2-horizon obs at all -- this is a shape requirement, not a
# comparability choice.
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r8_horiz_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r8_horiz_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r8_horiz] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: rewardShape MUST be geometric in the train cfg -- it IS the experiment,
# and an inherited r6 cfg would silently duplicate a run already in flight.
if ! grep -qE '^\s*rewardShape:\s*geometric\b' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: rewardShape is not 'geometric' in $CFG_ENV -- the reward shape IS this experiment (absent = a duplicate of r6_cf2)" >&2; exit 1
fi
# Guard: obsHorizons IS the experiment. An inherited r7 cfg has none, and would
# silently duplicate a run already in flight under a different name.
if ! grep -qE '^\s*obsHorizons:\s*\[' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: no obsHorizons in $CFG_ENV -- the horizon set IS this experiment (absent = a duplicate of r7_geom)" >&2; exit 1
fi
# Guard: numObs must match the horizon count (1599 per horizon, no betas here).
# A mismatch is a silent shape bug that surfaces deep inside rl_games.
NH=$(grep -oE '^\s*obsHorizons:\s*\[[^]]*\]' "$CFG_ENV" | tr ',' '\n' | wc -l)
NOBS=$(grep -oE '^\s*numObs:\s*[0-9]+' "$CFG_ENV" | grep -oE '[0-9]+')
if [ "$NOBS" != "$((NH * 1599))" ]; then
    echo "[bball-r8_horiz] ERROR: numObs=$NOBS but $NH horizons x 1599 = $((NH * 1599)) in $CFG_ENV" >&2; exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r8_horiz] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r8_horiz] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r8_horiz] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r8_horiz] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r8_horiz/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r8_horiz] RESUMING own run from ${CKPT}"
else
    echo "[bball-r8_horiz] first launch: FRESH START (no warm start possible -- numObs 9594 vs the sub2 teacher's 3198)"
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

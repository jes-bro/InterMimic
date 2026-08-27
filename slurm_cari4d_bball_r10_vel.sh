#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r10_vel"
#SBATCH --output=cari4d-bball-r10_vel-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R10_VEL -- the VELOCITY-REWARD arm. Same as bball-r7_geom except
# rewardWeights pv (0. -> 3.0) and rv (0. -> 0.25). Config-only; no code change.
#
# WHY. rpv = exp(-epv * w['pv']), so w['pv'] = 0. makes it exp(0) = 1.0 EXACTLY
# and rb = rp * rr * energy: body linear AND angular velocity are entirely
# ungraded. That is not a bball choice -- pv is 0. in all 384 configs in this
# repo, InterMimic's own included. On slow tabletop manipulation it costs
# nothing: hit the right position every frame at 30fps and velocity is pinned as
# a side effect. On a ballistic clip it is not the same thing. A jump is
# determined by TAKEOFF VELOCITY, and a position-only reward gives no direct
# signal for "launch harder now so you are in the right place in five frames" --
# which is close to the observed failure, reaching the takeoff pose without the
# momentum to leave the ground.
#
# WHY OFF r7 AND NOT r6. rb is a product: rp*rr*rpv*rrv*energy. Switching on two
# more sub-1.0 factors shrinks it further, which is the collapse r7's geometric
# mean exists to fix. Under the raw product these terms would plausibly hurt.
# That is also the likeliest reason they were zeroed upstream.
#
# WHERE THE NUMBERS CAME FROM (scripts/measure_reference_velocity_noise.py):
#   pv 3.0   at this clip's 1.98 m/s median body speed, a 20% velocity error is
#            0.4 m/s, squared 0.16, so rpv = exp(-0.48) ~ 0.62 -- graded, not
#            punishing.
#   rv 0.25  the same ratio to pv that w['r'] (2.5) has to w['p'] (30), so the
#            velocity pair inherits the position pair's balance rather than
#            introducing a second, unrelated scale.
# Object weights are deliberately untouched: opv is already nonzero at 0.1, and
# orv (ball spin) is a separate question that would make this two experiments.
#
# THE REFERENCE VELOCITY IS TRUSTWORTHY, WHICH IS WHY THIS IS WORTH RUNNING. The
# clips store positions, so anything pv grades is a finite difference, and a
# monocular reconstruction's jitter is invisible in the position reward and
# amplified in the derivative. Measured: this clip's body-velocity noise is
# 0.059 against an OMOMO mocap baseline of 0.038 -- 1.55x, not 5x. The
# derivative is real motion.
#
# GENERALISATION CAVEAT, stated up front. Absolute velocity error scales with
# speed, so a FIXED pv grades fast motions more harshly than slow ones: the same
# 20% error needs pv ~3 here and pv ~51 on OMOMO at 0.5 m/s. So a win here does
# NOT by itself license "pv 3.0" as an InterMimic default. Claiming that needs
# the same value checked on an OMOMO teacher cell. (w['p'] = 30 has exactly the
# same property and is accepted upstream, so this is not a novel objection --
# but it is the bar for calling it method rather than tuning.)
#
# HOW TO READ IT. Against r7_geom at the SAME sim step -- r10 differs from it in
# these two weights alone, and keeps the sub2 teacher warm start, so a
# difference at matched steps is attributable. These are progress reads, not a
# stopping rule.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r10_vel-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS" cari4d-bball-r10_vel-*.out | tail -16
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r10_vel_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r10_vel_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r10_vel] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: the velocity weights ARE the experiment. Inherited from r7 they are 0.,
# which makes rpv/rrv identically 1.0 and this an exact duplicate of r7_geom.
if ! grep -qE '^\s*pv:\s*[1-9]' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: rewardWeights.pv is not nonzero in $CFG_ENV -- pv 0 makes rpv = exp(0) = 1.0 and this a duplicate of r7_geom" >&2; exit 1
fi
if ! grep -qE '^\s*rv:\s*0?\.[0-9]|^\s*rv:\s*[1-9]' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: rewardWeights.rv is not nonzero in $CFG_ENV -- grading linear but not angular body velocity is arbitrary" >&2; exit 1
fi
# Guard: rewardShape MUST be geometric in the train cfg -- it IS the experiment,
# and an inherited r6 cfg would silently duplicate a run already in flight.
if ! grep -qE '^\s*rewardShape:\s*geometric\b' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: rewardShape is not 'geometric' in $CFG_ENV -- the reward shape IS this experiment (absent = a duplicate of r6_cf2)" >&2; exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r10_vel] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r10_vel] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r10_vel] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r10_vel] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r10_vel/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r10_vel] RESUMING own run from ${CKPT}"
else
    echo "[bball-r10_vel] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

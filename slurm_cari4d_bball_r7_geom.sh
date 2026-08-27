#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r7_geom"
#SBATCH --output=cari4d-bball-r7_geom-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R7_GEOM -- the REWARD-SHAPE arm. Identical to bball-r6_cf2 except
# env.rewardShape: geometric in the TRAIN cfg. One knob, so any difference from
# r6 is the reward shape alone.
#
# WHY (measured, and after three nulls). r4/r5/r6 each fixed a real, measured
# defect off r3 -- reset threshold, rollout coverage, unearnable contact labels
# -- and none beat r3. That pattern says the binding constraint is not in the
# space those knobs search.
#
# r6 at sim step 138000, held frames:
#     rb=0.304 ro=0.402 rig=0.422 rcg=0.240   product = 0.012
# The product is an AND gate, which is correct -- a policy that tracks the body
# and drops the object must not score. But it makes every term's gradient
# proportional to the OTHERS: dR/d(rcg) = rb*ro*rig = 0.052. Improving contact
# barely moves the reward BECAUSE the rest are bad, so nothing can be fixed
# first. Over 12h r6's held rcg_hand moved 0.409 -> 0.417.
#
# The geometric mean is a MONOTONE transform of the same product: the optimum
# and the AND property are unchanged (any zero still zeros it), but the value
# lands at 0.334 and dR/d(rcg) = R/(4*rcg) = 0.35, ~7x larger, and it stops
# collapsing when the other factors are weak.
#
# NOT a sum. Additive is what the multiplicative form exists to prevent, and
# r4_human1m already showed that failure with the reset instead: 28.7%
# completion at mean reward 0.14, 10.6% of episodes falling -- survival without
# imitation.
#
# HOW TO READ IT. The reward-shape change is visible in the held-frame factors
# and in how many episodes survive the clip. These are progress reads, not a
# stopping rule -- these runs take a long time and the interesting movement can
# come late.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r7_geom-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS" cari4d-bball-r7_geom-*.out | tail -16
# Compare against r6_cf2 at the SAME sim step: r7 differs from it in the reward
# shape alone, so a difference at matched steps is attributable.
#
# NOTE the eval twin deliberately does NOT set rewardShape: eval stays on the
# original product so r7's eval numbers stay comparable with r2..r6. The knob
# is a training-time change, not a correction to the grading rubric.
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r7_geom_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r7_geom_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r7_geom] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: rewardShape MUST be geometric in the train cfg -- it IS the experiment,
# and an inherited r6 cfg would silently duplicate a run already in flight.
if ! grep -qE '^\s*rewardShape:\s*geometric\b' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: rewardShape is not 'geometric' in $CFG_ENV -- the reward shape IS this experiment (absent = a duplicate of r6_cf2)" >&2; exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r7_geom] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r7_geom] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r7_geom] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r7_geom] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r7_geom/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r7_geom] RESUMING own run from ${CKPT}"
else
    echo "[bball-r7_geom] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

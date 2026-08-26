#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r6_cf2"
#SBATCH --output=cari4d-bball-r6_cf2-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R6_CF2 -- the CONTACT-LABEL arm. Identical to bball-r5_roll50 except
# motion_file (behave_cari4d_optj3d_cf -> _cf2), in both the train and eval
# cfgs. One knob, so any difference from r5 is the contact relabel alone.
#
# WHY (measured, not hypothesised). r3's eval reward breakdown split by
# reference contact state:
#     free (66%)  rb=0.482 ro=0.538 rig=0.594 rcg=0.686  reward=0.174
#     held (34%)  rb=0.342 ro=0.580 rig=0.596 rcg=0.141  reward=0.019
# ro is FLAT, which killed the free-flight-gating hypothesis outright. rcg is
# what collapses -- and since rcg_hand is pinned to 1.0 wherever the reference
# says no hand contact, the held row is the only one measuring grip at all.
#
# The cause was upstream, not in the policy. relabel_contact_flags.py rewrote
# channel 330 (contact_obj) only; rcg_hand grades contact_human (331..382),
# which the _cf build never touched. On _cf, 21 of 53 claimed-contact frames
# (40%) had NO hand body touching the ball -- worst +0.187 m -- so rcg_hand was
# unearnable there for ANY policy. The stale channel still carried the garbage
# 99-100 end contact the _cf changelog says it dropped.
#
# _cf2 re-derives the 32 hand flags from geometry (0.02 m finger-to-SURFACE,
# which matches this sim's PhysX contact_offset) and re-derives contact_obj from
# the same criterion. Worst gap +0.187 -> +0.012 m; channel disagreement 15/101
# -> 0/101; positions byte-identical.
#
# WHAT THIS ANSWERS. Does making rcg_hand earnable actually raise it? If held-
# frame rcg stays near 0.141 on satisfiable labels, the grip failure is the
# POLICY and the next lever is elsewhere. If it climbs, the contact term was
# starved and the dribble-vs-layup tradeoff should ease.
#
# READ IT WITH THE ref-contact SPLIT, not the clip average -- the free rows
# carry no grip information by construction. REWARD_BREAKDOWN=1 is exported
# below, so the split and the rcg-factor block print every 1000 steps.
#
# NOTE r6's eval twin also reads _cf2, unlike the rest of the ladder. That is
# deliberate: an arm trained on corrected labels must be graded on them, or it
# is penalised for exactly the frames the relabel fixed. It does mean r6's raw
# eval numbers are not directly comparable to r2/r3/r5's.
#
# _cf2 is NOT in git (datasets never are). Build it with
# scripts/relabel_contact_human.py before submitting; the guard below checks.
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r6_cf2_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r6_cf2_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r6_cf2] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r6_cf2] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r6_cf2] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r6_cf2] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r6_cf2] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r6_cf2/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r6_cf2] RESUMING own run from ${CKPT}"
else
    echo "[bball-r6_cf2] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

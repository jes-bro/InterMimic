#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r9_horiz_prod"
#SBATCH --output=cari4d-bball-r9_horiz_prod-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R9_HORIZ_PROD -- uniform observation horizons on the ORIGINAL product
# reward. Same as bball-r6_cf2 except env.obsHorizons ([1,16] ->
# [1,4,7,10,13,16]) and the numObs that follows (3198 -> 9594).
#
# WHY THIS CELL EXISTS. r7 and r8 change the reward shape; this one does not.
# Together the four arms are a 2x2, so the two effects can be separated instead
# of inferred:
#
#                    obsHorizons [1,16]      obsHorizons [1,4,7,10,13,16]
#     product              r6_cf2                    r9_horiz_prod
#     geometric            r7_geom                   r8_horiz
#
#   r9 vs r6  -> the horizons alone (confounded with the warm start, see below)
#   r8 vs r9  -> the reward shape alone, warm start held constant (neither has
#                one), which is the CLEANEST comparison in the set
#   r7 vs r6  -> the reward shape alone, warm start held constant (both have one)
#
# WHY THE HORIZONS. The MLP policy saw the reference at delta_t 1 and delta_t 16
# and nothing between -- a 15-frame blind gap. At 30fps that is 0.53s: a small
# slice of a slow OMOMO reach, but the layup's whole crouch-to-release, so the
# preload a jump requires happens where the policy cannot see it. That is the
# shape of the observed failure: the dribble and the jump are each learned, the
# transition between them is not. Filled UNIFORMLY between r6's own endpoints,
# not geometrically -- geometric would fit this clip's timing better and would
# be a guess about one motion, and this has to survive being applied unchanged
# to a whole dataset.
#
# FORCED CONFOUND. numObs 3198 -> 9594, so the sub2 TEACHER warm start every arm
# since r2 has used cannot load (first-layer shape mismatch). r9 starts FRESH.
# Against r6 that is a second difference. Against r8 it is not -- r8 is fresh
# too -- which is why r8-vs-r9 is the comparison to lean on.
#
# KILL CRITERION, written before launch. By sim step 100000, r9 must reach BOTH
#     held-frame rcg  >= 0.45
#     completed       >= 10%
# judged against r6 at the SAME step (same reward shape, so the numbers are
# comparable). A fresh start climbs more slowly, so read the trend too: still
# rising at 100k earns more time.
#     grep -h -A 4 "by ref-contact" cari4d-bball-r9_horiz_prod-*.out | tail -8
#     grep -h -A 6 "TERMINATION REASONS  (sim step 100000)" cari4d-bball-r9_horiz_prod-*.out
#
# NOTE the eval twin carries the same horizons. obsHorizons changes the obs
# SHAPE, so a 6-horizon policy cannot be run through a 2-horizon obs at all --
# a requirement, not a comparability choice.
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r9_horiz_prod_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r9_horiz_prod_train.yaml

# Guard: the RELABELLED data IS this arm. Cloned from r5_roll50, whose inherited
# _cf path is the silent-failure mode -- it would make this an exact duplicate of
# a run already in flight, under a different name. The trailing \b matters: _cf
# is a prefix of _cf2, so a loose match would accept the wrong dataset.
if ! grep -qE '^\s*motion_file:\s*InterAct/behave_cari4d_optj3d_cf2\s*$' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: motion_file is not behave_cari4d_optj3d_cf2 in $CFG_ENV -- the relabelled contact data IS the experiment (plain _cf would duplicate r5)" >&2; exit 1
fi
# Guard: the data must exist AND carry the fix. _cf2 is produced by
# scripts/relabel_contact_human.py and is NOT in git, so a fresh clone or a
# partner's machine will not have it -- fail loudly rather than let Isaac Gym
# report a confusing asset error 40 lines later.
MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -d "$MOTION_DIR" ]; then
    echo "[bball-r9_horiz_prod] ERROR: $MOTION_DIR not found. Build it first:" >&2
    echo "  python3 scripts/relabel_contact_human.py --src-dir InterAct/behave_cari4d_optj3d_cf --dst-dir $MOTION_DIR --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml --threshold 0.02" >&2
    exit 1
fi
# Guard: rewardShape must be ABSENT. Its absence IS this arm -- present, it
# would be an exact duplicate of r8_horiz under a different name.
if grep -qE '^\s*rewardShape:' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: rewardShape is set in $CFG_ENV -- this arm keeps the ORIGINAL product reward (setting it duplicates r8_horiz)" >&2; exit 1
fi
# Guard: obsHorizons IS the experiment. An inherited r7 cfg has none, and would
# silently duplicate a run already in flight under a different name.
if ! grep -qE '^\s*obsHorizons:\s*\[' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: no obsHorizons in $CFG_ENV -- the horizon set IS this experiment (absent = a duplicate of r7_geom)" >&2; exit 1
fi
# Guard: numObs must match the horizon count (1599 per horizon, no betas here).
# A mismatch is a silent shape bug that surfaces deep inside rl_games.
NH=$(grep -oE '^\s*obsHorizons:\s*\[[^]]*\]' "$CFG_ENV" | tr ',' '\n' | wc -l)
NOBS=$(grep -oE '^\s*numObs:\s*[0-9]+' "$CFG_ENV" | grep -oE '[0-9]+')
if [ "$NOBS" != "$((NH * 1599))" ]; then
    echo "[bball-r9_horiz_prod] ERROR: numObs=$NOBS but $NH horizons x 1599 = $((NH * 1599)) in $CFG_ENV" >&2; exit 1
fi
# Guard: rolloutLength stays at r5's 50 -- this arm changes the DATA, not the
# coverage, and a drifted value would confound the two.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: rolloutLength not 50 in $CFG_ENV -- r6 must match r5 here or the relabel is confounded with a coverage change" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r9_horiz_prod] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r9_horiz_prod] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r9_horiz_prod] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r9_horiz_prod] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r9_horiz_prod/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r9_horiz_prod] RESUMING own run from ${CKPT}"
else
    echo "[bball-r9_horiz_prod] first launch: FRESH START (no warm start possible -- numObs 9594 vs the sub2 teacher's 3198)"
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

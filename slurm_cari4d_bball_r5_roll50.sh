#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r5_roll50"
#SBATCH --output=cari4d-bball-r5_roll50-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R5_ROLL50 -- the COVERAGE-TUNING arm. Identical to bball-r3_roll30
# except rolloutLength (30 -> 50 in the train cfg). One knob, so the comparison
# is clean; and since r3 differs from r2_warm in the same knob alone, the three
# arms are a 300 / 30 / 50 ladder on one variable.
#
# WHY r3 was not the end of it. r3 fixed the outright BUG: at rolloutLength 300
# against a 101-frame clip, the start sampler randint(0, max(1, clip_len -
# rolloutLength)) (intermimic.py:1246) clamps to randint(0,1) = always frame 0,
# so stateInit Hybrid was silently a Start init and r2_warm logged completed
# 0 (0.0%) of 16,666,993 episodes -- the layup received zero gradient. At 30
# starts finally spread (0..70) and coverage went ~38% -> ~87% of the clip.
# But r3's own numbers stalled: rb unmoved at 0.117, 93.9% of episodes still
# ended on the human divergence reset.
#
# WHY 50 (Jess's read of r3's mid-clip render, 2026-08-25). Of ~20 sampled
# 30-frame windows only ONE opened before the takeoff; the rest began already
# airborne, where there is nothing left to learn about jumping. A window is
# useful only if it STARTS before the takeoff frame T and SURVIVES to reach it,
# and r3 measured a mean episode of ~17 steps, so the useful starts are the ~17
# frames before T drawn from 0..(100-L):
#     L=30 -> starts 0..70 -> ~18/71 = 25%
#     L=50 -> starts 0..50 -> ~18/51 = 35%     <-- this arm
#     L=60 -> starts 0..40 -> 44% if T<=40, 20% if T=50
# 50 is both higher-yield and INSENSITIVE to T across 40..50, where 60 swings by
# 2x on a frame nobody has measured. It also gives 1.67s windows instead of
# 1.0s, so crouch->extend->flight fits inside one episode.
#
# MEASURE T BEFORE READING THIS ARM. The takeoff frame is inferred from watching
# renders, not measured:
#   python3 scripts/inspect_bball_clip.py \
#       --clip InterAct/behave_cari4d_optj3d_cf/sub100_bball_000.pt --every 1
# Read "lowest body point per frame"; takeoff is where it leaves the grounded
# ~0.23m offset. If T > 55 the coverage argument above does not hold and 50 is
# not the right value.
#
# The human 0.5m reset is deliberately KEPT. It is not too tight -- it is the
# thing that kills the crawl exploit (see the noreset arm). r4_human1m is the
# arm that tests relaxing it; keeping it here is what makes r5-vs-r3 one knob.
#
# PSI stays OFF. physicalBufferSize is absent, so psi defaults to 1 and the PSI
# block is gated off (intermimic.py:1655). Note that rolloutLength 50 would
# UN-GATE it if the key were added (eligibility = mel - rollout_length + 1,
# intermimic.py:847) -- deliberately not done, so rolloutLength stays the only
# variable vs r3_roll30.
#
# SEPARATE experiment: own cfgs, own checkpoint dir
# (checkpoints/smplx_cari4d_bball_r5_roll50/nn/) -- writes NOTHING into r3's or r2_warm's.
#
# NOT a fresh start: the train cfg carries an EXPLICIT, Jess-approved warm start
# from checkpoints/smplx_teachers_new/sub2.pth (resume_from, read at
# intermimic_agent.py:177 regardless of load_checkpoint) -- the same init
# r2_warm used, which is what keeps rolloutLength the sole difference. Once this
# run has its own mimic.pth, resubmits resume from that instead.
#
# READING THE LOG -- `completed` changes meaning here. A rollout that hits its
# 50-frame cap resets WITHOUT terminating (humanoid.py:553), which TERM_REASON
# scores as completed. Expect nonzero completed% immediately; it is NOT
# comparable to r2_warm's 0.0%. The real read is the eval twin (full clip,
# Start init, rolloutLength kept at 300).

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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r5_roll50_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r5_roll50_train.yaml

# Guard: rolloutLength IS this arm. Cloned from r3_roll30, so BOTH inherited
# values are silent-failure modes: a 30 makes this a duplicate of r3, and a 300
# (r3's own ancestor) makes it a duplicate of r2_warm, which we know returns
# completed 0.0%. Only 50 is this experiment.
if ! grep -qE '^\s*rolloutLength:\s*50\b' "$CFG_ENV"; then
    echo "[bball-r5_roll50] ERROR: rolloutLength not 50 in $CFG_ENV -- the 50-frame rollout IS the experiment (30 would duplicate r3, 300 would duplicate r2_warm)" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 50 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 50 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r5_roll50] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 50 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r5_roll50] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 50 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r5_roll50] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r5_roll50] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r5_roll50] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r5_roll50] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r5_roll50] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r5_roll50/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r5_roll50] RESUMING own run from ${CKPT}"
else
    echo "[bball-r5_roll50] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

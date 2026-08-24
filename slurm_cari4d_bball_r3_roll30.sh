#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r3_roll30"
#SBATCH --output=cari4d-bball-r3_roll30-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R3_ROLL30 -- the COVERAGE arm. Identical to bball-r2_warm except
# rolloutLength (300 -> 30 in the train cfg). One knob, so the comparison is
# clean.
#
# WHY (measured on r2_warm at sim step 156000, not guessed):
#   completed 0 (0.0%) of 16,666,993 episodes -- the layup got ZERO gradient
#   mean episode ~38 control steps of a 101-frame clip (~1.3s)
#   ig_diverge 100% -- that column is the MERGED kinematic flag
#     (intermimic.py:1931); object/igRatio are off, so all of it is the human
#     0.5m divergence reset firing at the takeoff.
# The start sampler is randint(0, max(1, clip_len - rolloutLength))
# (intermimic.py:1246). At 300 vs a 101-frame clip that clamps to randint(0,1)
# = always frame 0, so stateInit Hybrid was silently a Start init: the ONLY way
# to reach the layup was to survive from frame 0, and nothing ever did. At 30,
# starts spread over frames 0-71 and envs begin inside the takeoff window.
#
# The human 0.5m reset is deliberately KEPT. It is not too tight -- it is the
# thing that kills the crawl exploit (see the noreset arm). The bug was that
# frame 38 was the only place the policy was ever asked to perform.
#
# PSI stays OFF. physicalBufferSize is absent, so psi defaults to 1 and the PSI
# block is gated off (intermimic.py:1655). Note that rolloutLength 30 would
# UN-GATE it if the key were added (eligibility = mel - rollout_length + 1,
# intermimic.py:847) -- deliberately not done, so rolloutLength stays the only
# variable vs r2_warm.
#
# SEPARATE experiment: own cfgs, own checkpoint dir
# (checkpoints/smplx_cari4d_bball_r3_roll30/nn/) -- writes NOTHING into r2_warm's.
#
# NOT a fresh start: the train cfg carries an EXPLICIT, Jess-approved warm start
# from checkpoints/smplx_teachers_new/sub2.pth (resume_from, read at
# intermimic_agent.py:177 regardless of load_checkpoint) -- the same init
# r2_warm used, which is what keeps rolloutLength the sole difference. Once this
# run has its own mimic.pth, resubmits resume from that instead.
#
# READING THE LOG -- `completed` changes meaning here. A rollout that hits its
# 30-frame cap resets WITHOUT terminating (humanoid.py:553), which TERM_REASON
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r3_roll30_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r3_roll30_train.yaml

# Guard: rolloutLength IS this arm. Cloned from r2_warm, so the one thing that
# can silently revert it is the inherited 300 -- which would make this an exact
# duplicate of a run we already know returns completed 0.0%.
if ! grep -qE '^\s*rolloutLength:\s*30\b' "$CFG_ENV"; then
    echo "[bball-r3_roll30] ERROR: rolloutLength not 30 in $CFG_ENV -- the short rollout IS the experiment" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 30 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 30 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r3_roll30] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 30 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r3_roll30] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 30 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r3_roll30] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*0\.5' "$CFG_ENV"; then
    echo "[bball-r3_roll30] ERROR: human reset not set to 0.5 in $CFG_ENV -- it is what keeps the crawl exploit dead" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r3_roll30] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r3_roll30] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r3_roll30] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r3_roll30/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r3_roll30] RESUMING own run from ${CKPT}"
else
    echo "[bball-r3_roll30] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

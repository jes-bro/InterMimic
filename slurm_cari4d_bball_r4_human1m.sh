#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-r4_human1m"
#SBATCH --output=cari4d-bball-r4_human1m-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# BBALL_R4_HUMAN1M -- the reset-relaxation arm. Identical to bball-r3_roll30
# except resetThresholds.human (0.5 -> 1.0). Runs ALONGSIDE r3, which is the
# control, so a difference in rb is attributable to this one knob.
#
# WHY (measured on r3_roll30, job 17037568, sim step 40000):
#   episodes 9,677,090 | completed 6.1% | ig_diverge 93.9%
#   mean episode 16.9 steps of a 30-frame window
#   rb 0.117 (r2_warm: 0.118) -- unmoved
# r3's coverage fix worked: starts now spread over frames 0-71, so the layup
# gets gradient for the first time and clip coverage went from ~38% to ~87%.
# But 93.9% of episodes are still executed by the human divergence reset ~17
# frames in (TERM_REASON's 'ig_diverge' column is the MERGED kinematic flag,
# intermimic.py:1931; object/igRatio are off, so all of it is the human reset).
# Coverage is no longer the binding constraint. This arm asks whether the reset
# now is: given 2x the rope, does the policy recover from a mis-timed takeoff
# instead of being killed for it?
#
# WHY 1.0 IS SAFER HERE THAN ON noreset: that arm ran 300-frame rollouts, which
# gave a crawler room to accumulate an advantage. At rolloutLength 30 an episode
# is one second. Object-side resets stay off, as in every arm since looseterm.
#
# THE EVAL TWIN KEEPS human: 0.5. It is the measuring instrument -- r3 and r4
# must be scored identically or their eval numbers do not compare. Nothing is
# hidden: the qualitative render runs NO_TERM=1, which disables early
# termination entirely.
#
# SEPARATE experiment: own cfgs, own checkpoint dir
# (checkpoints/smplx_cari4d_bball_r4_human1m/nn/) -- writes NOTHING into r3's.
#
# NOT a fresh start: the train cfg carries the same EXPLICIT, Jess-approved warm
# start from checkpoints/smplx_teachers_new/sub2.pth that r2_warm and r3 used --
# the same init is what keeps the r3-vs-r4 comparison one-knob. NOT warm-started
# from r3's checkpoint. Note that resume_from restores the checkpoint's epoch
# counter (intermimic_agent.py:186), so epoch_num starts inflated by the sub2
# teacher's own epochs -- divide [mem] step by horizon 32 for real progress.
#
# READING THE LOG -- `completed` is capped-rollout contaminated, exactly as in
# r3: a rollout hitting its 30-frame cap resets WITHOUT terminating and scores
# as completed. Compare r3 and r4 on the per-step breakdown terms (rb/ro/rig/
# rcg) and on mean episode length, NEVER on mean_rewards -- episode lengths
# differ between the arms, so episodic returns are not commensurable.

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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r4_human1m_train.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_r4_human1m_train.yaml

# Guard: rolloutLength IS this arm. Cloned from r2_warm, so the one thing that
# can silently revert it is the inherited 300 -- which would make this an exact
# duplicate of a run we already know returns completed 0.0%.
if ! grep -qE '^\s*rolloutLength:\s*30\b' "$CFG_ENV"; then
    echo "[bball-r4_human1m] ERROR: rolloutLength not 30 in $CFG_ENV -- the short rollout IS the experiment" >&2; exit 1
fi
# Guard: stateInit must be Hybrid. rolloutLength 30 only buys coverage because
# Hybrid samples a start frame; under Start the sampler is bypassed entirely
# (intermimic.py:1247) and 30 would just truncate every frame-0 episode.
if ! grep -qE '^\s*stateInit:\s*"Hybrid"' "$CFG_ENV"; then
    echo "[bball-r4_human1m] ERROR: stateInit not Hybrid in $CFG_ENV -- short rollout without Hybrid only truncates episodes" >&2; exit 1
fi
# Guard: PSI must stay absent. rolloutLength 30 un-gates it, and a stray
# physicalBufferSize would add a second variable to a one-knob experiment.
if grep -qE '^\s*physicalBufferSize:' "$CFG_ENV"; then
    echo "[bball-r4_human1m] ERROR: physicalBufferSize present in $CFG_ENV -- rolloutLength 30 un-gates PSI; that is a second variable" >&2; exit 1
fi
# Guards inherited from r2_warm: this arm keeps that termination regime exactly.
if ! grep -qE '^\s*resetThresholds:' "$CFG_ENV"; then
    echo "[bball-r4_human1m] ERROR: resetThresholds block missing from $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*human:\s*1\.0' "$CFG_ENV"; then
    echo "[bball-r4_human1m] ERROR: human reset not set to 1.0 in $CFG_ENV -- the relaxed threshold IS the experiment" >&2; exit 1
fi
for KNOB in object igRatio contactSteps; do
    if ! grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV"; then
        echo "[bball-r4_human1m] ERROR: resetThresholds.${KNOB} not false in $CFG_ENV -- object-side resets must stay off" >&2; exit 1
    fi
done

echo "[bball-r4_human1m] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[bball-r4_human1m] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_cari4d_bball_r4_human1m/nn/"

# --- resume resolution: own checkpoints only (walltime resubmits). ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
# NO warm-starting from another RUN (Jess rule 2026-08-11). The cfg's sub2
# TEACHER warm start is the explicit, approved exception, same as r2_warm.
RESUME_FROM=""
if [ -f "$CKPT" ]; then
    RESUME_FROM="$CKPT"; echo "[bball-r4_human1m] RESUMING own run from ${CKPT}"
else
    echo "[bball-r4_human1m] first launch: EXPLICIT warm start from smplx_teachers_new/sub2.pth (per cfg; Jess-approved, same init as r2_warm)"
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

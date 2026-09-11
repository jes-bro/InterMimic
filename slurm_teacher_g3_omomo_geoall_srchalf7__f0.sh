#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-g3_omomo_geoall_srchalf7__f0"
#SBATCH --output=teacher-g3_omomo_geoall_srchalf7__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# HALF-B g3 teacher: 7 sources (sub2 sub6 sub7 sub8 sub9 sub11 sub14, 1679 clips)
# x 43 bodies, RAGGED motion storage. Its twin srchalf6 holds the other 6
# sources; together they are srcall13, split for distillation (see the env cfg
# header for the split rationale). Hand-written from the srcall13 launcher.
# 7-day walltime (every g3 arm ran 7 days in practice, via override or bash);
# the auto-resume below is the fallback if the node dies, not the schedule.
# Eval when done:  HELDOUT="sub10 sub13 sub16" sh scripts/eval_one.sh g3_omomo_geoall_srchalf7__f0
#
# --mem=320G is ~30% over the RAGGED budget under the PADDED loader's 2.02x model
# (14.2 + 2.02 x 114 GiB = 244; scripts/motion_memory_budget.py). Padded would
# have needed ~796 GiB (sub11's 652-frame outlier sets the pad). If the srcall13
# run has reported MaxRSS by the time this is submitted, size from its measured
# factor instead.
#
# BEFORE THE FIRST SUBMISSION build the merged retarget tree (symlinks):
#   python3 scripts/merge_retarget_trees.py \
#       --sources InterAct/OMOMO_retarget_contact_src2 InterAct/OMOMO_retarget_contact_src6 \
#                 InterAct/OMOMO_retarget_contact_src7 InterAct/OMOMO_retarget_contact_src8 \
#                 InterAct/OMOMO_retarget_contact_src9 InterAct/OMOMO_retarget_contact_src11 \
#                 InterAct/OMOMO_retarget_contact_src14 \
#       --out InterAct/OMOMO_retarget_contact_srchalf7 \
#       --bodies-from isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_omomo_geoall_srchalf7__f0.yaml
# Expect: linked 72197 clips across 43 bodies.
# The task refuses to start on a missing (body, clip) file, so a partial merge
# fails at startup rather than training on a silent source fallback.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

# Reward diagnostics (print-only; none change training).
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
export POSE_REWARD_DEBUG=1

# UNIFORM env count across ALL 16 cells (batch = envs*horizon must not differ
# between compared arms). Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_omomo_geoall_srchalf7__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_omomo_geoall_srchalf7__f0.yaml

# Ragged guard: this arm does not fit padded (see header), so a cfg that lost
# the flag must not be allowed to try and OOM the node 20 minutes in.
if ! grep -qE '^\s*raggedMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: srchalf7 without raggedMotionData in $CFG_ENV" >&2; exit 1
fi
# The merged tree must exist and cover every source, or the task dies at startup
# anyway -- say so here, with the fix, instead of from a 320G job that got scheduled.
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for s in sub2 sub6 sub7 sub8 sub9 sub11 sub14; do
    if ! ls "$RT"/sub2/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $RT has no ${s}_* clips under body sub2 -- run the" \
             "merge_retarget_trees.py command in this script's header" >&2; exit 1
    fi
done

# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi

# Buffer guard: the cfg must carry exactly this cell's multiplier (12.0).
if ! grep -qE '^\s*default_buffer_size_multiplier:\s*12\.0' "$CFG_ENV"; then
    echo "[teacher] ERROR: buffer multiplier in $CFG_ENV is not 12.0" >&2; exit 1
fi

# Fold guard: this cell's TEST bodies must not be in its training list.
# Parsed exactly (yaml), not grepped -- sub1 vs sub10 substring traps.
for b in sub10 sub13 sub16; do
    if python3 -c "import yaml,sys; sys.exit(0 if '$b' in yaml.safe_load(open('$CFG_ENV'))['env']['subjectBodies'] else 1)"; then
        echo "[teacher] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2; exit 1
    fi
done

echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[teacher] G3 RECIPE g3_omomo_geoall_srchalf7__f0 (OMOMO data, half B = 7 sources, RAGGED motion): 43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_g3_omomo_geoall_srchalf7__f0/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    # Match the line whether the yaml says `resume_from: None` or `'None'`: the
    # committed g3 train cfgs are UNQUOTED and the fleet's sed only matched the
    # quoted form, so the rewrite was a no-op and every resubmission started fresh.
    sed -E "s|^(\s*resume_from:)\s*'?None'?\s*$|\1 '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    if ! grep -qF "resume_from: '${CKPT}'" "$RESUME_TRAIN"; then
        echo "[teacher] ERROR: could not rewrite resume_from in $CFG_TRAIN -- refusing to" \
             "start fresh over ${CKPT}" >&2; exit 1
    fi
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[teacher] RESUMING from ${CKPT}"
else
    echo "[teacher] fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --output checkpoints

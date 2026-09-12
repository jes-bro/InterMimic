#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-g3_bball7_geoall__f0"
#SBATCH --output=teacher-g3_bball7_geoall__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# g3 RECIPE ON THE MULTI-SUBJECT BASKETBALL RECONSTRUCTION: 52 clips from 7
# people (sub401 402 404 409 411 412 458) x f0's 43 bodies, every clip with its
# own reconstructed ball. Hand-written from slurm_teacher_g3_bball_geoall__f0.sh.
# 7-day walltime; the auto-resume below is the fallback if the node dies.
# Eval when done:  HELDOUT="sub10 sub13 sub16" sh scripts/eval_one.sh g3_bball7_geoall__f0
#
# --mem=64G: 2236 (body, clip) motions padded to the 189-frame longest clip is
# ~4 GB; the single-clip bball arm ran at 64G with the same 43 bodies.
#
# BEFORE THE FIRST SUBMISSION, in this order (each refuses to run on missing input):
#   1. rclone the extracted bundles + manifest + scripts/bball7_subject_betas.npz
#   2. MANIFEST=... BUNDLES_ROOT=... sbatch scripts/slurm_cari4d_bball7_convert.sh
#   3. sbatch --array=0-6 scripts/slurm_cari4d_bball7_retarget.sh   (read the verdicts)
#   4. the merge_retarget_trees.py command in that script's header
# Expect InterAct/behave_cari4d_bball7_cf2 with 52 clips and
# InterAct/behave_cari4d_bball7_f0_bodymajor with 46 bodies (43 training + the
# 3 held-out eval bodies) x 52 = 2392 files.

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

# UNIFORM env count across ALL cells (batch = envs*horizon must not differ
# between compared arms). Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_bball7_geoall__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_bball7_geoall__f0.yaml

# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi
# Per-clip balls need per-object mass, or they weigh 0.45-0.85 kg by recon size.
if ! grep -qE '^\s*objectMass:' "$CFG_ENV"; then
    echo "[teacher] ERROR: bball7 without objectMass in $CFG_ENV" >&2; exit 1
fi

# The data must be there for every source: the task dies at startup anyway,
# say so here with the fix instead of from a scheduled job.
MF=$(grep -oE '^\s*motion_file:\s*\S+' "$CFG_ENV" | awk '{print $2}')
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for s in sub401 sub402 sub404 sub409 sub411 sub412 sub458; do
    if ! ls "$MF"/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $MF has no ${s}_* clips -- run scripts/slurm_cari4d_bball7_convert.sh" >&2; exit 1
    fi
    if ! ls "$RT"/sub1/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $RT has no ${s}_* clips under body sub1 -- run the retarget" \
             "array + merge in scripts/slurm_cari4d_bball7_retarget.sh" >&2; exit 1
    fi
done

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
echo "[teacher] G3 RECIPE g3_bball7_geoall__f0 (EgoExo4D bball, 7 people, 52 clips, per-clip balls): 43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_g3_bball7_geoall__f0/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    # Match the line whether the yaml says `resume_from: None` or `'None'`.
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

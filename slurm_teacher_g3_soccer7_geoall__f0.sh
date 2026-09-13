#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-g3_soccer7_geoall__f0"
#SBATCH --output=teacher-g3_soccer7_geoall__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# g3 RECIPE ON THE EGOEXO4D SOCCER RECONSTRUCTION, THE 7 PEOPLE WITH THE MOST
# CLIPS: 47 clips from sub405 (15) 487 (8) 482 (6) 490 (6) 485 (5) 484 (4)
# 493 (3) x f0's 43 bodies, every clip with its own reconstructed ball (0.43 kg,
# restitution 0.65 ball + floor). Hand-written from
# slurm_teacher_g3_soccer15_geoall__f0.sh; SAME converted data and SAME merged
# retarget tree as that arm, only dataSub differs (the task opens only the
# (body, clip) files dataSub names). 7-day walltime; auto-resume below.
# Eval when done:  HELDOUT="sub10 sub13 sub16" sh scripts/eval_one.sh g3_soccer7_geoall__f0
#
# --mem=64G: 2021 (body, clip) motions padded to the 677-frame longest clip
# (Sub87 t002a) is ~12.1 GB of motion -> ~37 GiB RSS by the validated model
# (14.2 + 2.02 x motion GiB). bball7 runs at 64G with 3.4 GB of motion.
# TIE FOR 7TH PLACE: sub493 / sub495 / sub496 all have 3 clips; sub493 is in
# by lowest id (see the env cfg header).
#
# BEFORE THE FIRST SUBMISSION: the same five steps as in
# slurm_teacher_g3_soccer15_geoall__f0.sh -- the whole 15-subject retarget
# array + merge, not a 7-subject subset (one tree serves both arms).

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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_soccer7_geoall__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_soccer7_geoall__f0.yaml

# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi
# Per-clip balls need per-object mass, or they weigh by recon size.
if ! grep -qE '^\s*objectMass:' "$CFG_ENV"; then
    echo "[teacher] ERROR: soccer arm without objectMass in $CFG_ENV" >&2; exit 1
fi

# The data must be there for every source: the task dies at startup anyway,
# say so here with the fix instead of from a scheduled job.
MF=$(grep -oE '^\s*motion_file:\s*\S+' "$CFG_ENV" | awk '{print $2}')
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for s in sub405 sub482 sub484 sub485 sub487 sub490 sub493; do
    if ! ls "$MF"/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $MF has no ${s}_* clips -- run scripts/slurm_cari4d_soccer_convert.sh + the relabel" >&2; exit 1
    fi
    if ! ls "$RT"/sub1/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $RT has no ${s}_* clips under body sub1 -- run the retarget" \
             "array + merge in scripts/slurm_cari4d_soccer_retarget.sh" >&2; exit 1
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
echo "[teacher] G3 RECIPE g3_soccer7_geoall__f0 (EgoExo4D soccer, top-7 people, 47 clips, per-clip balls 0.43 kg): 43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_g3_soccer7_geoall__f0/nn/"

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

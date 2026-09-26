#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tch-g3_omomo_geoall_src15_xf_nvadlr_nopose__f0"
#SBATCH --output=teacher-g3_omomo_geoall_src15_xf_nvadlr_nopose__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# METHOD CANDIDATE xf_nvadlr_nopose of g3_omomo_geoall_src15__f0: the OMOMO source sub15, f0's 43 bodies
# teacher as a 6-TOKEN TRANSFORMER with normalize_value + adaptive LR (exact-KL
# 0.06) and rewardTerms.pose.enable false (no relative joint-angle reward
# factor). NOT an ablation: the teacher nopose arms beat the with-pose teachers
# (Jess 2026-09-25), so the pose term is being dropped from the method.
# Hand-written from slurm_teacher_g3_omomo_geoall_src15__f0.sh: names, the cfgs, the conda
# env line, the transformer guard, the recipe echo and the METHOD CANDIDATE
# echo differ, nothing else. --mem is the base's (storage = the base's).
# One of the 15-arm fleet: OMOMO sub1 2 3 5 6 7 8 9 11 12 14 15 17 + bball7 + soccer15.
# 7-day walltime; the auto-resume below is the fallback if the node dies.
# Collect for a student with:  python3 scripts/collect_g3_teachers.py --exp-suffix _xf_nvadlr_nopose ...
# Eval when done (hand-write omomo_eval_g3_omomo_geoall_src15_xf_nvadlr_nopose__f0.yaml first, mirroring the
# pose key):  HELDOUT="sub10 sub13 sub16" sh scripts/eval_one.sh g3_omomo_geoall_src15_xf_nvadlr_nopose__f0

source ~/.bashrc
conda deactivate
conda activate "${INTERMIMIC_ENV:-intermimic-gym2}"   # another machine: INTERMIMIC_ENV=<its env name>
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_omomo_geoall_src15_xf_nvadlr_nopose__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_omomo_geoall_src15_xf_nvadlr_nopose__f0.yaml

# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi

# Transformer guard: this arm's train cfg must carry the 6-token transformer
# (the env gives it six horizons; num_tokens = len(obsHorizons)), or a cfg that
# lost the block would train the MLP under a transformer arm's name.
if ! grep -qE '^\s*name:\s*intermimic_transformer\s*$' "$CFG_TRAIN"; then
    echo "[teacher] ERROR: XF arm without network intermimic_transformer in $CFG_TRAIN" >&2; exit 1
fi
if ! grep -qE '^\s*num_tokens:\s*6\s*$' "$CFG_TRAIN"; then
    echo "[teacher] ERROR: XF arm without transformer.num_tokens 6 in $CFG_TRAIN" >&2; exit 1
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
echo "[teacher] G3 RECIPE g3_omomo_geoall_src15_xf_nvadlr_nopose__f0 (OMOMO data, 6-TOKEN TRANSFORMER, normval+adaptive LR, NO POSE TERM): 43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_g3_omomo_geoall_src15_xf_nvadlr_nopose__f0/nn/"
echo "[teacher] METHOD CANDIDATE xf_nvadlr_nopose of g3_omomo_geoall_src15__f0: 6-token transformer + normalize_value + adaptive LR (exact-KL 0.06); rewardTerms.pose.enable true -> false (no relative joint-angle reward factor)"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID}.yaml"
    # Match the line whether the yaml says `resume_from: None` or `'None'`: the g3
    # train cfgs are UNQUOTED and the old pattern only matched the quoted form, so
    # the rewrite was a no-op and every resubmission started fresh over its own
    # checkpoints while printing RESUMING. Refuse to start if the rewrite fails.
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

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name="tch-gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0"
#SBATCH --output=teacher-gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
# gen4 EGOEXO TEACHER launcher for gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0: the g3 launcher
# (slurm_teacher_g3_bball7_geoall_xf_nvadlr_nopose__f0.sh) with the names/paths swapped to this arm, the eval hint
# holding out sub4 as well, and three guards (every body's *_inertial MJCF and
# retargeted reference tree must exist; the motionScaleReward block must be present).
# GENERALIST: basketball + soccer in one teacher (22 sources); objectPropsFile guard
# instead of objectMass, the 22-source data loop, --mem=96G.
# Eval when done (hand-write omomo_eval_gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0.yaml first, mirroring
# humanoidAssetSuffix + motionScaleReward):  HELDOUT="sub4 sub10 sub13 sub16" sh scripts/eval_one.sh gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0
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

# UNIFORM env count across ALL cells (batch = envs*horizon must not differ
# between compared arms). Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0.yaml

# Retarget arm: streamed motion -> fragmentation cap (job 16502149 post-mortem),
# and the retarget knobs must actually be on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[teacher] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi
# gen4 guards: every body must have its *_inertial MJCF and its retargeted
# reference tree, or the run would silently load stock files / miss bodies.
RT_DIR=$(grep -E '^\s*retargetedMotionDir:' "$CFG_ENV" | awk '{print $2}')
if ! grep -qE '^\s*humanoidAssetSuffix:\s*_inertial\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: gen4 arm without humanoidAssetSuffix: _inertial in $CFG_ENV" >&2; exit 1
fi
for b in $(grep -E '^\s*- sub[0-9]+\s*$' "$CFG_ENV" | awk '{print $2}'); do
    f="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${b}_inertial.xml"
    [ -f "$f" ] || { echo "[teacher] ERROR: missing $f (run scripts/mjcf_add_inertials.py --all)" >&2; exit 1; }
    [ -d "$RT_DIR/$b" ] || { echo "[teacher] ERROR: no retargeted refs $RT_DIR/$b (retarget the activity clips onto the gen4 bodies + re-merge; see scripts/make_gen4_egoexo_teacher_cfgs.py)" >&2; exit 1; }
done
# Motion-scale guard: this arm trains on the per-clip reward exponent; a cfg that
# lost the block would train the plain recipe under this arm's name.
if ! grep -qE '^\s*motionScaleReward:' "$CFG_ENV" || ! grep -qE '^\s*referenceStd:\s*0.5157\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: motion-scale arm without motionScaleReward.referenceStd 0.5157 in $CFG_ENV" >&2; exit 1
fi
# Generalist: a basketball and a soccer ball do not share a mass or a restitution,
# so the per-object props file must be there (written on the cluster by
# scripts/merge_activity_data.py; it is not tracked).
PROPS=$(grep -oE '^\s*objectPropsFile:\s*\S+' "$CFG_ENV" | awk '{print $2}')
[ -n "$PROPS" ] || { echo "[teacher] ERROR: generalist without objectPropsFile in $CFG_ENV" >&2; exit 1; }
[ -f "$PROPS" ] || { echo "[teacher] ERROR: $PROPS missing -- run scripts/merge_activity_data.py (see its docstring)" >&2; exit 1; }

# The data must be there for every source: the task dies at startup anyway,
# say so here with the fix instead of from a scheduled job.
MF=$(grep -oE '^\s*motion_file:\s*\S+' "$CFG_ENV" | awk '{print $2}')
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for s in sub401 sub402 sub404 sub405 sub409 sub411 sub412 sub458 sub480 sub482 sub484 sub485 sub487 sub488 sub489 sub490 sub491 sub493 sub494 sub495 sub496 sub497; do
    if ! ls "$MF"/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $MF has no ${s}_* clips -- build it with scripts/merge_activity_data.py (bball7 + soccer15)" >&2; exit 1
    fi
    if ! ls "$RT"/sub1/${s}_*.pt >/dev/null 2>&1; then
        echo "[teacher] ERROR: $RT has no ${s}_* clips under body sub1 -- run the retarget" \
             "array + merge in scripts/slurm_cari4d_bball7_retarget.sh" >&2; exit 1
    fi
done

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
echo "[teacher] GEN4 EGOEXO RECIPE gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0 (sources sub401 sub402 sub404 sub405 sub409 sub411 sub412 sub458 sub480 sub482 sub484 sub485 sub487 sub488 sub489 sub490 sub491 sub493 sub494 sub495 sub496 sub497; 43 gen4 bodies, _inertial MJCFs, 6-TOKEN TRANSFORMER, normval+adaptive LR, NO POSE TERM, MOTION-SCALE EXPONENT ref 0.5157 capped at product): no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0/nn/"
echo "[teacher] gen4 twin of g3_bball7_geoall_xf_nvadlr_nopose__f0: gen4 bodies + inertial MJCFs + motionScaleReward; GENERALIST over basketball + soccer (objectPropsFile, plane 0.7)"

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

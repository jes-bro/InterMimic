#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=208G
#SBATCH --gres=gpu:1
#SBATCH --job-name="tch-gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0"
#SBATCH --output=teacher-gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
# gen4 OBJECT-GROUP TEACHER launcher for gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0: the gen4 src1 launcher
# (slurm_teacher_gen4_omomo_geoall_src1_xf_nvadlr_nopose__f0.sh) with the names/paths swapped to this arm, --mem=208G
# (ragged budget, scripts/motion_memory_budget.py --objects largebox smallbox plasticbox suitcase trashcan monitor), and two
# more guards (raggedMotionData on; the merged retarget tree holds this group's clips
# for a real AND a gen4 body). Nothing else differs.
# Eval when done (hand-write omomo_eval_gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0.yaml first, mirroring
# humanoidAssetSuffix + dataObjects):  HELDOUT="sub4 sub10 sub13 sub16" sh scripts/eval_one.sh gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0
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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0.yaml

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
    [ -d "$RT_DIR/$b" ] || { echo "[teacher] ERROR: no retargeted refs $RT_DIR/$b (run slurm_retarget_gen.sh for this source)" >&2; exit 1; }
done
# Object-group guards: this arm does not fit padded (see header), so a cfg that
# lost the ragged flag must not get to OOM the node 20 minutes in; and the
# merged tree must hold this group's clips for a real body AND a gen4 body, or
# the task dies at startup on the first missing (body, clip) file anyway.
if ! grep -qE '^\s*raggedMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: object-group arm without raggedMotionData in $CFG_ENV" >&2; exit 1
fi
for o in largebox smallbox plasticbox suitcase trashcan monitor; do
    for b in sub2 sub600; do
        if ! ls "$RT_DIR"/$b/*_${o}_*.pt >/dev/null 2>&1; then
            echo "[teacher] ERROR: $RT_DIR has no ${o} clips under body $b -- re-run the" \
                 "merge_retarget_trees.py command (see scripts/make_gen4_activity_teacher_cfgs.py)" >&2; exit 1
        fi
    done
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
echo "[teacher] GEN4 OBJECT-GROUP RECIPE gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0 (OMOMO data, objects largebox smallbox plasticbox suitcase trashcan monitor, 13 sources, RAGGED motion, 6-TOKEN TRANSFORMER, normval+adaptive LR, NO POSE TERM): 43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0/nn/"
echo "[teacher] OBJECT GROUP boxes: the gen4 src1 nopose recipe with dataSub = 13 sources, dataObjects = largebox smallbox plasticbox suitcase trashcan monitor, ragged storage, merged retarget tree; sampling unchanged (env e owns object e % n_obj)"

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

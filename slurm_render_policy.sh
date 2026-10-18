#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="render-pol"
#SBATCH --output=render-pol-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Render ONE policy rollout (a trained checkpoint DRIVING the humanoid) to mp4 --
# the policy analog of slurm_replay.sh (which is kinematic, no policy).
#
# THE RENDER MUST MATCH THE EXPERIMENT. This script used to hardcode
# the old shared template (omomo_test_multibody.yaml), the same chunk-1 smoke-test template the evals used,
# and then sed the body/source/object into it. For a gen-2 retargeting arm that
# produced a video of the policy tracking the UN-retargeted reference; for a
# gen-3 arm it would have shown the wrong ball physics with the free-flight gate
# off (and a 3230-vs-9594 obs width that cannot even load). A wrong number in a
# CSV gets caught eventually; a wrong video goes in a talk.
#
# So the environment is resolved from the ARM, by the same resolver the eval uses
# (scripts/check_eval_cfg.py), and the config is passed to --cfg_env UNTOUCHED --
# body/source/object are CLI overrides now, not a sed over the file. A render and
# its eval therefore cannot disagree about what environment the arm ran in.
#
#   # an arm, its own environment, its latest checkpoint:
#   ARM=g3_bball__f0 sbatch slurm_render_policy.sh
#   # pick body / source / object / length:
#   ARM=g2_mlp_ret_stock__f0 BODY=sub16 SOURCE=sub2 OBJECT=largetable FRAMES=600 \
#       sbatch slurm_render_policy.sh
#   # a specific checkpoint of that arm:
#   ARM=g3_omomo__f0 CHECKPOINT=checkpoints/.../nn/mimic_00054600.pth sbatch slurm_render_policy.sh
#   # a STUDENT arm (cfg omomo_student_*, train/rlg/omomo_student_*): the eval cfg names the
#   # entry/task (intermimic.run_distill + InterMimicDistillG3/G4) and the student .pth runs
#   # alone, DISTILL_SKIP_TEACHERS=1 (no teacher set needed to render):
#   ARM=student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0 BODY=sub6 SOURCE=sub6 \
#       OBJECT=woodchair CLIP=sub6_woodchair_020 sbatch slurm_render_policy.sh
#   # the SAME clip on several bodies (comparable videos; camera auto-framed to the clip):
#   ARM=gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0 BODY=sub4 SOURCE=sub6 OBJECT=woodchair \
#       CLIP=sub6_woodchair_000 FRAMES=600 sbatch slurm_render_policy.sh
#
# For a pre-gen-2 checkpoint with no arm-specific config (v1 multibody, distill,
# crosspair), pass ENV_YAML/TRAIN explicitly -- e.g.
#   ENV_YAML=isaacgym/src/intermimic/data/cfg/omomo_eval_v1_multibody_mlp.yaml
#
# Video -> renders/policy_<exp>_<body>_<source>[_<object>]_<stamp>.mp4

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

ARM="${ARM:-}"
BODY="${BODY:-sub10}"          # body in sim (subjectBodies)
SOURCE="${SOURCE:-}"           # reference motion (dataSub); default = the arm's own
OBJECT="${OBJECT:-}"           # empty = all objects
FRAMES="${FRAMES:-300}"

# Environment: the ARM's own eval config, or an explicit ENV_YAML for a
# pre-gen-2 checkpoint. No default template -- a render of the wrong environment
# is indistinguishable from a render of a bad policy.
CFGDIR=isaacgym/src/intermimic/data/cfg
if [ -n "$ARM" ]; then
    ENV_YAML=$(python3 scripts/check_eval_cfg.py --arm "$ARM") || exit 2
    case "$ARM" in
        student_*) TRAIN="${TRAIN:-$CFGDIR/train/rlg/omomo_${ARM}.yaml}" ;;          # students: no teacher_ prefix
        *)         TRAIN="${TRAIN:-$CFGDIR/train/rlg/omomo_teacher_${ARM}.yaml}" ;;
    esac
    if [ -z "${CHECKPOINT:-}" ]; then
        EXPNAME=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$TRAIN" | awk '{print $2}')
        CHECKPOINT=$(ls -1 "checkpoints/$EXPNAME/nn"/mimic_0*.pth 2>/dev/null | sort | tail -1)
        [ -z "$CHECKPOINT" ] && CHECKPOINT="checkpoints/$EXPNAME/nn/mimic.pth"
    fi
    # Default the source to what the arm actually trained on, rather than assuming
    # sub2 -- the bball arm's only source is sub100, and sub2 selects zero clips.
    [ -z "$SOURCE" ] && { SOURCE=$(python3 scripts/check_eval_cfg.py --default-source "$ARM") || exit 2; }
elif [ -n "${ENV_YAML:-}" ]; then
    : "${TRAIN:?set TRAIN too when passing ENV_YAML (it carries the network arch)}"
    [ -z "$SOURCE" ] && SOURCE=sub2
else
    echo "[render] ERROR: set ARM=<arm> (resolves the arm's own eval config), or" >&2
    echo "         ENV_YAML=<cfg> TRAIN=<cfg> for a pre-gen-2 checkpoint." >&2
    echo "         There is deliberately no default environment: rendering a policy" >&2
    echo "         in the wrong environment produces a convincing, wrong video." >&2
    exit 2
fi
# NB: no apostrophe in a ${VAR:?word} message -- bash quote-matches the word even
# inside double quotes, and one stray ' makes the whole script a syntax error.
: "${CHECKPOINT:?set CHECKPOINT=<path>, or ARM=<arm> to take that latest checkpoint}"

[ -f "$CHECKPOINT" ] || { echo "[render] ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }
[ -f "$ENV_YAML" ]   || { echo "[render] ERROR: env cfg not found: $ENV_YAML"; exit 1; }
[ -f "$TRAIN" ]      || { echo "[render] ERROR: train cfg not found: $TRAIN"; exit 1; }
# Entrypoint / task: stated by the eval cfg (evalEntry / evalTask), never guessed from the
# name -- teachers: intermimic.run + InterMimic (the defaults); students: run_distill + the
# distill task that builds the STUDENT observation. Rendering a student through the teacher
# path would feed it the teacher obs. Students render from the .pth alone (teacher skip).
eval "$(python3 - "$ENV_YAML" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print(f"EVAL_ENTRY='{c.get('evalEntry', 'intermimic.run')}'")
print(f"EVAL_TASK='{c.get('evalTask', 'InterMimic')}'")
PY
)"
SKIP_TEACHERS=""
[ "$EVAL_ENTRY" = intermimic.run_distill ] && SKIP_TEACHERS="DISTILL_SKIP_TEACHERS=1"

# CLIP pin (optional): play exactly ONE clip so renders of different bodies /
# arms show the IDENTICAL motion. Without it the task samples a clip at random
# per run (intermimic.py _sample_motion_ids), so two renders are not comparable.
# Same mechanism as slurm_replay_xbody.sh: a one-clip motion dir passed as
# --motion_file. The retargeted reference is resolved by clip basename under
# retargetedMotionDir/<body>/, so the pin works for retargeting arms too.
# CLIP must start with SOURCE's sub<N> (the dataSub filter keeps it) and be a
# clip of OBJECT when OBJECT is set. The camera is auto-framed to the clip
# (scripts/cam_for_clip.py) unless RECORD_VIDEO_CAM_POS is already set.
CLIP="${CLIP:-}"
MOT_ARG=""
if [ -n "$CLIP" ]; then
    case "$CLIP" in
        "${SOURCE}"_*) : ;;
        *) echo "[render] ERROR: CLIP=$CLIP does not start with SOURCE=$SOURCE" >&2; exit 1 ;;
    esac
    MOTION_DIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$ENV_YAML'))['env']['motion_file'])")
    SRC_CLIP="$(pwd)/${MOTION_DIR}/${CLIP}.pt"
    [ -f "$SRC_CLIP" ] || { echo "[render] ERROR: clip not found: $SRC_CLIP" >&2; exit 1; }
    if [ -n "$OBJECT" ]; then
        case "$CLIP" in
            *_"${OBJECT}"_*) : ;;
            *) echo "[render] ERROR: CLIP=$CLIP is not a ${OBJECT} clip" >&2; exit 1 ;;
        esac
    fi
    CLIPDIR="/tmp/render_policy_clip_${CLIP}_$$"
    mkdir -p "$CLIPDIR"; ln -sf "$SRC_CLIP" "$CLIPDIR/${CLIP}.pt"
    MOT_ARG="--motion_file $CLIPDIR"
    echo "[render] CLIP pinned: only ${CLIP}.pt (motion_file -> $CLIPDIR)"
    if [ -z "${RECORD_VIDEO_CAM_POS:-}" ]; then
        eval "$(python3 scripts/cam_for_clip.py "$SRC_CLIP" | grep -E '^export ')" || { echo "[render] ERROR: cam_for_clip.py failed for $SRC_CLIP" >&2; exit 1; }
        echo "[render] camera auto-framed to the clip: pos=$RECORD_VIDEO_CAM_POS target=$RECORD_VIDEO_CAM_TARGET"
    fi
fi

TAG="${BODY}_${SOURCE}${OBJECT:+_$OBJECT}${CLIP:+_${CLIP##*_}}"
scontrol update JobId="$SLURM_JOB_ID" JobName="rp-$TAG" 2>/dev/null || true

mkdir -p renders
# No temp config and no sed. body/source/object are CLI overrides (config.py), so
# --cfg_env receives the committed, reviewed file byte-for-byte. The sed this
# replaces also had the block-style-list bug: it rewrote the `dataSub:` line and
# orphaned the `- sub2` beneath it, silently corrupting any per-arm config.
OBJ_ARG="--data_objects all"
[ -n "$OBJECT" ] && OBJ_ARG="--data_objects $OBJECT"

# Name by RUN (checkpoint's experiment dir) + tag + timestamp -- never a bare
# body/source tag: four arms rendered on the same body must not overwrite each
# other, and successive peeks at one arm must not either (Jess rule).
EXP=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="renders/policy_${EXP}_${TAG}_${STAMP}.mp4"
echo "[render] ckpt=$CHECKPOINT"
echo "[render] env_cfg=$ENV_YAML  train_cfg=$TRAIN"
echo "[render] entry=$EVAL_ENTRY task=$EVAL_TASK${SKIP_TEACHERS:+  ($SKIP_TEACHERS: student .pth alone)}"
echo "[render] body=$BODY source=$SOURCE object=${OBJECT:-ALL} frames=$FRAMES -> $OUT"
# Record WHAT ENVIRONMENT this video shows, in the job log, next to the video's
# own name. A render is judged by eye and carries no other provenance.
python3 - "$ENV_YAML" <<'EOPY'
import sys, yaml
e = (yaml.safe_load(open(sys.argv[1])) or {}).get("env", {})
g = (e.get("rewardTerms") or {}).get("freeFlightGate") or {}
print(f"[render] motion={e.get('motion_file')} retarget={e.get('retargetedMotionDir', '<none>')}")
print(f"[render] numObs={e.get('numObs')} betas={e.get('betas_file', '<none>')} "
      f"objectDensity={e.get('objectDensity')} ffgResets={g.get('resets', False)}")
EOPY
env $SKIP_TEACHERS RECORD_VIDEO="$OUT" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m "$EVAL_ENTRY" --task "$EVAL_TASK" \
        --cfg_env "$ENV_YAML" --cfg_train "$TRAIN" \
        --subject_bodies "$BODY" --data_sub "$SOURCE" $OBJ_ARG $MOT_ARG \
        --test --checkpoint "$CHECKPOINT" --headless --num_envs 1

echo
echo "[render] done:"
ls -lh "$OUT"

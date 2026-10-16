#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:1
#SBATCH --job-name="stu-gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0"
#SBATCH --output=student-gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
# STUDENT, gen4 OBJECT GROUPS, FULL EPISODES, XF / RET / NVADLR / NOPOSE / MSEXP: the four gen4 object-group
# teachers (boxes / stands / chairs / tables, motion-scale reward, LATEST snapshots) distilled
# into one 6-TOKEN TRANSFORMER student via InterMimicDistillG4 (routes each env by its clip's
# OBJECT; teachers.yaml is object-keyed). This launcher is the g3 OMOMO XF student's
# (slurm_student_g3_omomo_xf_ret_nvadlr__f0.sh) with the names/paths swapped, --task InterMimicDistillG4, the fold
# guard holding out sub4 too, the teacher-set guard checking the 13 OBJECTS, and the gen4 +
# motion-scale guards added. --mem=480G as the srcall13 / all-objects arms (same data).
# FULL EPISODES: rolloutLength 1000 + minEpisodeFrames 30 (a guard checks both) --
# starts anywhere, every episode runs to the clip's end, PSI live on every clip; the
# teachers are still the 50-frame-window ones. The startup line '[psi] ... eligible ...
# cells' must show every motion eligible. Matched-epoch control: the 50-frame student's
# checkpoints (job 17894921: mimic_00001000/2000/3000.pth).
#
# BEFORE THE FIRST SUBMISSION, on the machine that holds the teacher checkpoints:
#   1. the merged retarget tree must hold the gen4 bodies (sub600-629) -- see
#      scripts/make_gen4_activity_teacher_cfgs.py for the merge command
#   2. python3 scripts/collect_g3_teachers.py --gen4-objects boxes stands chairs tables \
#          --out checkpoints/teachers/gen4_omomo_groups_msexp
#      (latest snapshot of each; refuses a partial set or an object owned twice; records epochs)
# Startup prints one line per teacher (file, objects, epoch) and the object->teacher table, and
# refuses on a source-keyed manifest, an uncovered object, or any obs-width / token mismatch.
# Eval when done: hand-write omomo_eval_student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0.yaml (evalTask InterMimicDistillG4) first.
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

# Same env count as the teachers. Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0.yaml

# Ragged guard: 13 sources do not fit padded (~1.6 TB); a cfg that lost the flag
# must not be allowed to try and OOM the node 20 minutes in.
if ! grep -qE '^\s*raggedMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[student] ERROR: OMOMO student without raggedMotionData in $CFG_ENV" >&2; exit 1
fi
# The merged tree must exist and cover every source.
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for s in sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17; do
    if ! ls "$RT"/sub2/${s}_*.pt >/dev/null 2>&1; then
        echo "[student] ERROR: $RT has no ${s}_* clips under body sub2 -- run the" \
             "merge_retarget_trees.py command in the srcall13 launcher header" >&2; exit 1
    fi
done

# gen4 guards: every body must have its *_inertial MJCF and its retargeted reference
# dir in the merged tree, or the run would silently load stock files / miss bodies.
if ! grep -qE '^\s*humanoidAssetSuffix:\s*_inertial\s*$' "$CFG_ENV"; then
    echo "[student] ERROR: gen4 student without humanoidAssetSuffix: _inertial in $CFG_ENV" >&2; exit 1
fi
for b in $(grep -E '^\s*- sub[0-9]+\s*$' "$CFG_ENV" | awk '{print $2}'); do
    f="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${b}_inertial.xml"
    [ -f "$f" ] || { echo "[student] ERROR: missing $f (run scripts/mjcf_add_inertials.py --all)" >&2; exit 1; }
    [ -d "$RT/$b" ] || { echo "[student] ERROR: no retargeted refs $RT/$b in the merged tree (re-run merge_retarget_trees.py)" >&2; exit 1; }
done
# Motion-scale guard: the student is graded by the teachers' reward; a cfg that lost
# the block would train it against the old reward under this arm's name.
if ! grep -qE '^\s*motionScaleReward:' "$CFG_ENV" || ! grep -qE '^\s*referenceStd:\s*0.5157\s*$' "$CFG_ENV"; then
    echo "[student] ERROR: motion-scale student without motionScaleReward.referenceStd 0.5157 in $CFG_ENV" >&2; exit 1
fi
# Full-episode guard: this arm trains with starts anywhere and episodes to the clip's
# end; a cfg that lost either key would train the 50-frame-window arm under this name.
if ! grep -qE '^\s*rolloutLength:\s*1000\s*$' "$CFG_ENV" || ! grep -qE '^\s*minEpisodeFrames:\s*30\s*$' "$CFG_ENV"; then
    echo "[student] ERROR: full-episode arm without rolloutLength 1000 + minEpisodeFrames 30 in $CFG_ENV" >&2; exit 1
fi

# Retarget arm: streamed motion -> fragmentation cap, and the retarget knobs on.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[student] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*retargetedMotionDir:' "$CFG_ENV"; then
    echo "[student] ERROR: retarget arm without retargetedMotionDir in $CFG_ENV" >&2; exit 1
fi

# Buffer guard: the cfg must carry the recipe's multiplier (12.0).
if ! grep -qE '^\s*default_buffer_size_multiplier:\s*12\.0' "$CFG_ENV"; then
    echo "[student] ERROR: buffer multiplier in $CFG_ENV is not 12.0" >&2; exit 1
fi

# Fold guard: f0's TEST bodies must not be in the training list (parsed, not grepped).
for b in sub4 sub10 sub13 sub16; do
    if python3 -c "import yaml,sys; sys.exit(0 if '$b' in yaml.safe_load(open('$CFG_ENV'))['env']['subjectBodies'] else 1)"; then
        echo "[student] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2; exit 1
    fi
done

# Teacher-set guard: teachers.yaml must exist, be OBJECT-keyed, and cover all 13 objects, or the
# task refuses at startup anyway -- say so here, before a 480G job is scheduled.
# (one manifest routes one way: a per-source g3 manifest here would be a wrong-teacher run)
TP=$(grep -oE '^\s*teacherPolicy:\s*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -f "$TP/teachers.yaml" ]; then
    echo "[student] ERROR: no $TP/teachers.yaml -- run scripts/collect_g3_teachers.py --gen4-objects (see header)" >&2; exit 1
fi
if ! python3 -c "
import yaml,sys
t=yaml.safe_load(open('$TP/teachers.yaml'))['teachers']
if any('sources' in e for e in t):
    print('[student] ERROR: teachers.yaml is keyed by sources -- a g3 manifest; this student routes by OBJECT (collect with --gen4-objects)'); sys.exit(1)
have={o for e in t for o in e['objects']}
need={'clothesstand','floorlamp','largebox','largetable','monitor','plasticbox','smallbox','smalltable','suitcase','trashcan','tripod','whitechair','woodchair'}
miss=sorted(need-have)
print('[student] teachers:', len(t), 'files;', 'missing objects:', miss or 'none')
sys.exit(1 if miss else 0)"; then
    echo "[student] ERROR: $TP/teachers.yaml does not cover every OMOMO object" >&2; exit 1
fi
TC=$(grep -oE '^\s*teacherPolicyCFG:\s*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -f "isaacgym/src/$TC" ]; then
    echo "[student] ERROR: teacherPolicyCFG not found: isaacgym/src/$TC" >&2; exit 1
fi
# Token guard: the transformer's token count must equal the student horizon count.
if ! python3 -c "
import yaml,sys
e=yaml.safe_load(open('$CFG_ENV'))['env']; t=yaml.safe_load(open('$CFG_TRAIN'))['params']['network']
n=t.get('transformer',{}).get('num_tokens'); h=len(e['studentObsHorizons'])
sys.exit(0 if (t['name']=='intermimic_transformer' and n==h) else 1)"; then
    echo "[student] ERROR: transformer.num_tokens != len(studentObsHorizons) (or network is not the transformer)" >&2; exit 1
fi

echo "[student] invocation: python -u -m intermimic.run_distill --task InterMimicDistillG4 --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=${SLURM_JOB_ID:-none})"
echo "[student] GEN4 OBJECT-GROUP student: 4 gen4 msexp teachers (boxes stands chairs tables, latest, routed by OBJECT via InterMimicDistillG4) -> 6-token transformer (9594), normval + adaptive LR (KL 0.06), RAGGED, NO POSE TERM, MOTION-SCALE REWARD (ref 0.5157), FULL EPISODES (rolloutLength 1000 > every clip, minEpisodeFrames 30: starts anywhere, run to the clip end, PSI live; 50-frame-window teachers), 43 gen4 bodies, num_envs=$NUM_ENVS"
echo "[student] host=$(hostname) job=${SLURM_JOB_ID:-none} -> checkpoints/smplx_student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID:-$$}.yaml"
    # Match the line whether the yaml says `resume_from: None` or `'None'`.
    sed -E "s|^(\s*resume_from:)\s*'?None'?\s*$|\1 '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    if ! grep -qF "resume_from: '${CKPT}'" "$RESUME_TRAIN"; then
        echo "[student] ERROR: could not rewrite resume_from in $CFG_TRAIN -- refusing to" \
             "start fresh over ${CKPT}" >&2; exit 1
    fi
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[student] RESUMING from ${CKPT}"
else
    echo "[student] fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run_distill \
    --task InterMimicDistillG4 \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --output checkpoints

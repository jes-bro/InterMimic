#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="stu-g3_act_nocpr_xf_ret_nvadlr_nopose__f0"
#SBATCH --output=student-g3_act_nocpr_xf_ret_nvadlr_nopose__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# STUDENT, ACTIVITIES WITHOUT CPR (act_nocpr), XF / RET / NVADLR, METHOD CANDIDATE
# nopose (NOT an ablation: the teacher nopose arm beat the with-pose teacher, so
# the pose term is being dropped from the method and this is the candidate recipe):
# the bball7 + soccer15 g3 teachers (2 checkpoints, 22 sources = the act student's
# 35 minus cpr13's 13) distilled into one 6-TOKEN TRANSFORMER student
# (InterMimicDistillG3 via intermimic.run_distill), normalize_value + adaptive
# LR (exact-KL 0.06), on ITS OWN merged activity data with per-object
# mass/restitution, and rewardTerms.pose.enable false (no relative joint-angle
# reward factor). Hand-written from slurm_student_g3_act_xf_ret_nvadlr_tfinal__f0.sh:
# names, the cfgs, the two data-prep commands below and the METHOD CANDIDATE echo differ,
# nothing else (the guards read every path from the cfg). No with-pose twin of
# this source set exists (Jess 2026-09-25: nopose only), so its read against the
# act students confounds "no cpr" with "no pose" -- see the env cfg header.
# 7-day walltime; auto-resume below.
# On GCP (no Slurm): NUM_ENVS=1024 sh scripts/gcp_run_in_tmux.sh $0 act_nocpr_xf_nopose   -- fits a2-highgpu-1g
#
# --mem=64G: the act student's budget (132 clips x 43 bodies RAGGED); this set
# holds FEWER clips (no cpr), so it is an upper bound.
#
# BEFORE THE FIRST SUBMISSION, on the machine that holds the data + checkpoints
# (each step refuses to redo; the merge refuses a non-empty output dir):
#   1. python3 scripts/merge_activity_data.py --arms bball7 soccer15 \
#          --out-motion InterAct/behave_cari4d_act_nocpr \
#          --out-retarget InterAct/behave_cari4d_act_nocpr_f0_bodymajor \
#          --props-out isaacgym/src/intermimic/data/cfg/object_props_g3_act_nocpr.yaml \
#          --bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0.yaml \
#          --student-plane-restitution 0.7
#      (the props file it writes must be committed next to the cfg, as object_props_g3_act.yaml was)
#   2. python3 scripts/collect_g3_teachers.py --activities bball7 soccer15 \
#          --out checkpoints/teachers/g3_act_nocpr
#      (PRINTS the epoch it picked per teacher -- record those two numbers)
# Early-run check for the adaptive LR: info/last_lr should hover ~2e-5 after
# warm-up; railing to 1e-6 = threshold too low, steady climb = too high.

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

NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0.yaml

if ! grep -qE '^\s*raggedMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[student] ERROR: activity student without raggedMotionData in $CFG_ENV" >&2; exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if ! grep -qE '^\s*cpuMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[student] ERROR: retarget arm without cpuMotionData in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*default_buffer_size_multiplier:\s*12\.0' "$CFG_ENV"; then
    echo "[student] ERROR: buffer multiplier in $CFG_ENV is not 12.0" >&2; exit 1
fi
for b in sub10 sub13 sub16; do
    if python3 -c "import yaml,sys; sys.exit(0 if '$b' in yaml.safe_load(open('$CFG_ENV'))['env']['subjectBodies'] else 1)"; then
        echo "[student] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2; exit 1
    fi
done

# Merged data + per-object props: every source in the cfg's dataSub must have
# clips in the flat dir, the props file's dataSub must equal the cfg's, and the
# retarget tree must hold every clip under every body (the task refuses a
# partial tree at startup; say so here with the fix).
MD=$(grep -oE '^\s*motion_file:\s*\S+' "$CFG_ENV" | awk '{print $2}')
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
OP=$(grep -oE '^\s*objectPropsFile:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for d in "$MD" "$RT"; do
    [ -d "$d" ] || { echo "[student] ERROR: $d missing -- run merge_activity_data.py (see header)" >&2; exit 1; }
done
[ -f "$OP" ] || { echo "[student] ERROR: $OP missing -- run merge_activity_data.py (see header)" >&2; exit 1; }
if ! python3 -c "
import yaml,sys,os
e=yaml.safe_load(open('$CFG_ENV'))['env']; p=yaml.safe_load(open('$OP'))
subs=[str(s) for s in e['dataSub']]
if p.get('dataSub')!=subs: print('[student] props dataSub != cfg dataSub:', p.get('dataSub'), subs); sys.exit(1)
if abs(float(p['student_plane_restitution'])-float(e['plane']['restitution']))>1e-9:
    print('[student] props plane restitution', p['student_plane_restitution'], '!= cfg', e['plane']['restitution']); sys.exit(1)
clips=[f for f in os.listdir('$MD') if f.endswith('.pt')]
missing=[s for s in subs if not any(f.startswith(s+'_') for f in clips)]
if missing: print('[student] sources with no clips in $MD:', missing); sys.exit(1)
objs={f[:-3].split('_')[-2] for f in clips}
noprop=sorted(objs-set(p['objects']))
if noprop: print('[student] objects with no props entry:', noprop); sys.exit(1)
print('[student] data:', len(clips), 'clips,', len(objs), 'objects,', len(subs), 'sources; props ok')"; then
    exit 1
fi

# Teacher-set guard: teachers.yaml must cover every source in dataSub.
TP=$(grep -oE '^\s*teacherPolicy:\s*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -f "$TP/teachers.yaml" ]; then
    echo "[student] ERROR: no $TP/teachers.yaml -- run scripts/collect_g3_teachers.py (see header)" >&2; exit 1
fi
if ! python3 -c "
import yaml,sys
t=yaml.safe_load(open('$TP/teachers.yaml'))['teachers']
have={s for e in t for s in e['sources']}
need={int(str(s)[3:]) for s in yaml.safe_load(open('$CFG_ENV'))['env']['dataSub']}
miss=sorted(need-have)
print('[student] teachers:', len(t), 'files;', 'missing sources:', miss or 'none')
sys.exit(1 if miss else 0)"; then
    echo "[student] ERROR: $TP/teachers.yaml does not cover every source in dataSub" >&2; exit 1
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

echo "[student] invocation: python -u -m intermimic.run_distill --task InterMimicDistillG3 --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=${SLURM_JOB_ID:-none})"
echo "[student] ACTIVITY-NOCPR XF/RET/NVADLR student: bball7+soccer15 -> 6-token transformer (9594), normval + adaptive LR (KL 0.06), per-object physics, RAGGED, num_envs=$NUM_ENVS"
echo "[student] host=$(hostname) job=${SLURM_JOB_ID:-none} -> checkpoints/smplx_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0/nn/"
echo "[student] METHOD CANDIDATE nopose (on the act_nocpr source set): rewardTerms.pose.enable true -> false (no relative joint-angle reward factor)"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID:-$$}.yaml"
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
    --task InterMimicDistillG3 \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --output checkpoints

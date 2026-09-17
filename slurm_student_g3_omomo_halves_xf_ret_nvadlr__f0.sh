#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:1

#SBATCH --job-name="stu-g3_omomo_halves_xf_ret_nvadlr__f0"
#SBATCH --output=student-g3_omomo_halves_xf_ret_nvadlr__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# STUDENT, OMOMO, XF / RET / NVADLR: the two HALF teachers (srchalf6: sub1 3 5 12 15 17; srchalf7: sub2 6 7 8 9 11 14), each serving its own sources distilled
# into one 6-TOKEN TRANSFORMER student (InterMimicDistillG3 via
# intermimic.run_distill), normalize_value + adaptive LR (exact-KL 0.06). Same
# data and teachers as the MLP twin (slurm_student_g3_omomo_mlp_ret_stock__f0.sh);
# the two launchers differ only in names and the CFG_TRAIN. 7-day walltime; the
# auto-resume below is the fallback if the node dies.
# On GCP (no Slurm): sh scripts/gcp_run_in_tmux.sh $0 omomo_halves_xf
#
# --mem=480G is the srcall13 arm's RAGGED budget under the padded 2.02x model;
# that arm is the calibration run -- read its MaxRSS (sacct/sstat) and lower this.
#
# BEFORE THE FIRST SUBMISSION, on the machine that holds the checkpoints:
#   1. the merged retarget tree (see the srcall13 launcher header) must exist
#   2. python3 scripts/collect_g3_teachers.py \
#          --omomo-arms srchalf6 srchalf7 --out checkpoints/teachers/g3_omomo_halves
#      (shared with the MLP twin; refuses a partial set; records each teacher's epoch)
# Startup prints one line per teacher (file, sources, epoch) and refuses on any
# obs-width / routing / token-count mismatch -- read the first 3 minutes of the log.
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

# Same env count as the teachers. Override per submission: NUM_ENVS=4096 sbatch ...
NUM_ENVS="${NUM_ENVS:-2048}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_omomo_halves_xf_ret_nvadlr__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_omomo_halves_xf_ret_nvadlr__f0.yaml

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
for b in sub10 sub13 sub16; do
    if python3 -c "import yaml,sys; sys.exit(0 if '$b' in yaml.safe_load(open('$CFG_ENV'))['env']['subjectBodies'] else 1)"; then
        echo "[student] ERROR: test body $b found in subjectBodies of $CFG_ENV" >&2; exit 1
    fi
done

# Teacher-set guard: teachers.yaml must exist and cover all 13 sources, or the
# task refuses at startup anyway -- say so here, before a 480G job is scheduled.
TP=$(grep -oE '^\s*teacherPolicy:\s*\S+' "$CFG_ENV" | awk '{print $2}')
if [ ! -f "$TP/teachers.yaml" ]; then
    echo "[student] ERROR: no $TP/teachers.yaml -- run scripts/collect_g3_teachers.py (see header)" >&2; exit 1
fi
if ! python3 -c "
import yaml,sys
t=yaml.safe_load(open('$TP/teachers.yaml'))['teachers']
have={s for e in t for s in e['sources']}
need={1,2,3,5,6,7,8,9,11,12,14,15,17}
miss=sorted(need-have)
print('[student] teachers:', len(t), 'files;', 'missing sources:', miss or 'none')
sys.exit(1 if miss else 0)"; then
    echo "[student] ERROR: $TP/teachers.yaml does not cover every OMOMO source" >&2; exit 1
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
echo "[student] OMOMO XF/RET/NVADLR student: 2 half teachers -> 6-token transformer (9594), normval + adaptive LR (KL 0.06), RAGGED, num_envs=$NUM_ENVS"
echo "[student] host=$(hostname) job=${SLURM_JOB_ID:-none} -> checkpoints/smplx_student_g3_omomo_halves_xf_ret_nvadlr__f0/nn/"

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
    --task InterMimicDistillG3 \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --output checkpoints

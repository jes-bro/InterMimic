#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:1

#SBATCH --job-name="stu-g3_omomo7_mlp_ret_stock__f0"
#SBATCH --output=student-g3_omomo7_mlp_ret_stock__f0-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# STUDENT, OMOMO7, MLP / RET / STOCK: the 7 per-source g3 OMOMO teachers of the srchalf7 split (sub2 6 7 8 9 11 14) -- the student twin of the srchalf7 ablation base
# distilled into one MLP 1024/1024/512, 6 student horizons (9594), constant LR student (InterMimicDistillG3 via
# intermimic.run_distill). Data = the base teacher arm's verbatim (see the env
# cfg header). 7-day walltime; auto-resume below.
# On GCP (no Slurm): sh scripts/gcp_run_in_tmux.sh $0 omomo7_mlp   -- shape: a2-highgpu-4g
#
# BEFORE THE FIRST SUBMISSION, on the machine that holds the checkpoints + data:
#   python3 scripts/collect_g3_teachers.py --omomo-sources 2 6 7 8 9 11 14 --out checkpoints/teachers/g3_omomo7
# Startup prints one line per teacher (file, sources, epoch) and refuses on any
# obs-width / routing mismatch -- read the first 3 minutes of the log.

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

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_omomo7_mlp_ret_stock__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_omomo7_mlp_ret_stock__f0.yaml

if ! grep -qE '^\s*raggedMotionData:\s*[Tt]rue' "$CFG_ENV"; then
    echo "[student] ERROR: this student is sized for RAGGED storage; flag missing in $CFG_ENV" >&2; exit 1
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

# Data guard: the motion dir and the retarget tree must exist and cover every
# source in dataSub (the task refuses a partial tree at startup; say so here).
MD=$(grep -oE '^\s*motion_file:\s*\S+' "$CFG_ENV" | awk '{print $2}')
RT=$(grep -oE '^\s*retargetedMotionDir:\s*\S+' "$CFG_ENV" | awk '{print $2}')
for d in "$MD" "$RT"; do
    [ -d "$d" ] || { echo "[student] ERROR: $d missing (see the base teacher launcher header for how it is built)" >&2; exit 1; }
done
if ! python3 -c "
import yaml,sys,os
e=yaml.safe_load(open('$CFG_ENV'))['env']
subs=[str(s) for s in e['dataSub']]
clips=[f for f in os.listdir('$MD') if f.endswith('.pt')]
missing=[s for s in subs if not any(f.startswith(s+'_') for f in clips)]
if missing: print('[student] sources with no clips in $MD:', missing); sys.exit(1)
body=e['subjectBodies'][0]
tree=os.listdir(os.path.join('$RT', body)) if os.path.isdir(os.path.join('$RT', body)) else []
miss2=[s for s in subs if not any(f.startswith(s+'_') for f in tree)]
if miss2: print('[student] sources with no clips under $RT/'+body+':', miss2); sys.exit(1)
print('[student] data:', len(subs), 'sources; clips + tree ok')"; then
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

echo "[student] invocation: python -u -m intermimic.run_distill --task InterMimicDistillG3 --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=${SLURM_JOB_ID:-none})"
echo "[student] OMOMO7 MLP / RET / STOCK student: the 7 per-source g3 OMOMO teachers of the srchalf7 split (sub2 6 7 8 9 11 14) -- the student twin of the srchalf7 ablation base -> MLP 1024/1024/512, 6 student horizons (9594), constant LR, num_envs=$NUM_ENVS"
echo "[student] host=$(hostname) job=${SLURM_JOB_ID:-none} -> checkpoints/smplx_student_g3_omomo7_mlp_ret_stock__f0/nn/"

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

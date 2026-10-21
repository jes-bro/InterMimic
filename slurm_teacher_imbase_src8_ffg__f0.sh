#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="tch-imbase_src8_ffg__f0"
#SBATCH --output=teacher-imbase_src8_ffg__f0-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
# INTERMIMIC BASELINE imbase_src8_ffg__f0 (launcher): upstream's per-subject recipe on source sub8 + OUR
# free-flight gate (rewardTerms.freeFlightGate resets true / reward false), nothing else. DERIVED from
# omomo_upstream_train_env.yaml / omomo_upstream_train.yaml by scripts/make_intermimic_baseline_cfgs.py; the base body is
# upstream's own file smplx/omomo.xml (= Subject 9, the paper's base model). Pinned by
# tests/test_intermimic_baseline_cfgs.py. Never edit as-run cfgs.
# Upstream's recipe trains at 4096 envs; override per submission: NUM_ENVS=2048 sbatch ...
source ~/.bashrc
conda deactivate
conda activate "${INTERMIMIC_ENV:-intermimic-gym2}"   # another machine: INTERMIMIC_ENV=<its env name>
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
NUM_ENVS="${NUM_ENVS:-4096}"
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_imbase_src8_ffg__f0.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_imbase_src8_ffg__f0.yaml

# Baseline guards: ONE source, upstream's base body (no subjectBodies), no retargeting, and
# the gate block present -- a cfg that lost any of these would train something else under
# this arm's name.
if ! grep -qE "^\s*dataSub:\s*\['sub8'\]\s*$" "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm must train exactly source sub8 (dataSub) in $CFG_ENV" >&2; exit 1
fi
if grep -qE '^\s*(subjectBodies|retargetedMotionDir|humanoidAssetSuffix|motionScaleReward|objectFlightReward|rewardShape):' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm carries a non-upstream key (subjectBodies/retarget/exponent/shape) in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*robotType:\s*"smplx/omomo\.xml"\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm must use upstream's body file smplx/omomo.xml in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*freeFlightGate:' "$CFG_ENV" || ! grep -qE '^\s*resets:\s*true\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm without rewardTerms.freeFlightGate.resets true (OUR gate) in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\s*name:\s*intermimic\s*$' "$CFG_TRAIN" || grep -qE '^\s*name:\s*intermimic_transformer' "$CFG_TRAIN"; then
    echo "[teacher] ERROR: baseline arm must use upstream's MLP network in $CFG_TRAIN" >&2; exit 1
fi
[ -d "InterAct/OMOMO_new" ] || { echo "[teacher] ERROR: InterAct/OMOMO_new not found (run from the repo root)" >&2; exit 1; }
[ -f "isaacgym/src/intermimic/data/assets/smplx/omomo.xml" ] || { echo "[teacher] ERROR: upstream body file smplx/omomo.xml missing" >&2; exit 1; }

echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[teacher] INTERMIMIC BASELINE imbase_src8_ffg__f0 (upstream per-subject recipe on source sub8: base body smplx/omomo.xml = Subject 9, raw clips from InterAct/OMOMO_new, Hybrid 0.1 + PSI 3 at a 300-frame window, product reward at the paper's weights, MLP) + OUR free-flight gate (resets on no-contact frames OFF). num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_imbase_src8_ffg__f0/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID:-$$}.yaml"
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

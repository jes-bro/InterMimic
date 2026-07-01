#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="kin-obj"
#SBATCH --output=kin-obj-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# KINEMATIC object-perturbation test (NO physics, NO policy). Retargets SOURCE
# motion onto BODY (via subjectBodies + play_dataset) and perturbs the object
# (scale/translate/yaw), then measures how far the retargeted hands are from the
# perturbed object surface on the source-contact frames. gap~0 => the interaction
# survived kinematic retargeting; gap blows up => it broke.
#
# Sweeps YAW PAST the safe range (0..180deg) to find each object's break point --
# symmetric objects should stay flat, asymmetric ones blow up early.
#
#   BODY=sub4 SOURCE=sub2 OBJECT=largetable sbatch slurm_kinematic_objtest.sh
#   OBJECT=largebox SCALE=1.15 TRANSLATE=0.05 sbatch slurm_kinematic_objtest.sh
#   YAWS="0 10 20 30 45 90 180" OBJECT=woodchair sbatch slurm_kinematic_objtest.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

BODY="${BODY:-sub2}"           # target body (subject retarget)
SOURCE="${SOURCE:-sub2}"       # source motion
OBJECT="${OBJECT:-largetable}" # one object per run
SCALE="${SCALE:-1.0}"          # isotropic object scale
TRANSLATE="${TRANSLATE:-0.0}"  # +X object offset (m)
YAWS="${YAWS:-0 15 30 45 90 135 180}"   # degrees, swept PAST the safe range
NUM_ENVS="${NUM_ENVS:-64}"
BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml

scontrol update JobId="$SLURM_JOB_ID" JobName="kin-${OBJECT}-b${BODY#sub}" 2>/dev/null || true
echo "[kin-obj] body=$BODY source=$SOURCE object=$OBJECT scale=$SCALE translate=${TRANSLATE}m yaw-sweep=[$YAWS]deg"

for YAWDEG in $YAWS; do
    YAWRAD=$(python -c "import math;print(math.radians($YAWDEG))")
    CFG="/tmp/kinobj_${BODY}_${SOURCE}_${OBJECT}_y${YAWDEG}.yaml"
    # Robustly patch the test config: set source/body/object and add env.objectPerturb.
    python - "$SOURCE" "$BODY" "$OBJECT" "$SCALE" "$TRANSLATE" "$YAWRAD" "$BASE" "$CFG" <<'PY'
import sys, yaml
src, body, obj, scale, trans, yaw, base, out = sys.argv[1:9]
d = yaml.safe_load(open(base)); e = d['env']
e['dataSub'] = [src]; e['subjectBodies'] = [body]; e['dataObjects'] = [obj]
e['objectPerturb'] = {'enable': True, 'scale': float(scale),
                      'translateM': float(trans), 'yawRad': float(yaw)}
yaml.safe_dump(d, open(out, 'w'), default_flow_style=False, sort_keys=False)
PY
    echo ""
    echo "================ YAW=${YAWDEG}deg  scale=$SCALE  translate=${TRANSLATE}m ================"
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG" --cfg_train "$TRAIN" \
        --test --play_dataset --headless --num_envs "$NUM_ENVS"
done

echo ""
echo "=== sweep done. Read the [objperturb] lines: 'preserved(<5cm)=X%' per yaw ==="
echo "=== flat X% across yaw = symmetric/tolerant; X% collapsing = break point found ==="

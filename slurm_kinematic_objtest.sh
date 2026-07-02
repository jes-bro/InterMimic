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
# (scale/translate/yaw). For EACH yaw it now does two things:
#   1. RENDERS A VIDEO  (/tmp -> render_results/) so you can SEE whether the hand
#      stays on the object -- a headless gap number alone is uninterpretable.
#   2. Prints the [objperturb] hand<->surface gap metric to stdout.
#
# The video is driven by the SAME RECORD_VIDEO / MAX_VIDEO_FRAMES machinery the
# render_qualitative scripts use: the rl_games player's play_dataset branch
# records N frames then sys.exit(0)s. Setting these vars ALSO fixes the old hang
# -- without RECORD_VIDEO the player replayed the dataset n_games*10 times and
# blew past the wall clock (job 16055436 spent 2h stuck on yaw=0 and swept
# nothing). With the frame cap each yaw finishes and the for-loop advances.
#
# Sweeps YAW PAST the safe range to find each object's break point -- symmetric
# objects should stay flat, asymmetric ones blow up early.
#
#   sbatch slurm_kinematic_objtest.sh                                   # sub2->sub2, largetable sweep
#   BODY=sub4 SOURCE=sub2 OBJECT=largetable sbatch slurm_kinematic_objtest.sh
#   YAWS="0" sbatch slurm_kinematic_objtest.sh                          # NULL-CHECK first (gap should be small)
#   YAWS="0 45 90 180" OBJECT=woodchair sbatch slurm_kinematic_objtest.sh
#
# Videos land in render_results/objperturb_<jobid>/kinobj_<obj>_b<body>_s<src>_y<deg>.mp4

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

BODY="${BODY:-sub2}"             # target body (subject retarget)
SOURCE="${SOURCE:-sub2}"         # source motion
OBJECT="${OBJECT:-largetable}"   # one object per run
SCALE="${SCALE:-1.0}"            # isotropic object scale
TRANSLATE="${TRANSLATE:-0.0}"    # +X object offset (m)
YAWS="${YAWS:-0 45 90 180}"      # degrees, swept PAST the safe range
NUM_ENVS="${NUM_ENVS:-1}"        # 1 = clean single-body video (camera records env 0)
MAX_FRAMES="${MAX_FRAMES:-300}"  # frames per yaw video (~10s @ 30fps); also caps the run
# Camera: default is a close-up of the single env; override for other objects.
CAM_POS="${CAM_POS:-3.0,3.0,2.5}"
CAM_TARGET="${CAM_TARGET:-0.0,0.0,1.0}"

BASE=isaacgym/src/intermimic/data/cfg/omomo_test_multibody.yaml
TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml
DEST="render_results/objperturb_${SLURM_JOB_ID:-local}"
mkdir -p "$DEST"

scontrol update JobId="${SLURM_JOB_ID:-0}" JobName="kin-${OBJECT}-b${BODY#sub}" 2>/dev/null || true
echo "[kin-obj] body=$BODY source=$SOURCE object=$OBJECT scale=$SCALE translate=${TRANSLATE}m yaw-sweep=[$YAWS]deg num_envs=$NUM_ENVS frames/yaw=$MAX_FRAMES"
echo "[kin-obj] videos -> $DEST"

# Headless IsaacGym camera capture needs a virtual X display; wrap in xvfb.
if command -v xvfb-run >/dev/null 2>&1; then XVFB="xvfb-run -a"; else
    echo "[kin-obj] WARNING: xvfb-run not found; camera capture may fail on a headless node"; XVFB=""; fi

for YAWDEG in $YAWS; do
    YAWRAD=$(python -c "import math;print(math.radians($YAWDEG))")
    CFG="/tmp/kinobj_${BODY}_${SOURCE}_${OBJECT}_y${YAWDEG}.yaml"
    VID="/tmp/kinobj_${OBJECT}_b${BODY#sub}_s${SOURCE#sub}_y${YAWDEG}.mp4"
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
    # RECORD_VIDEO makes the play_dataset player record MAX_VIDEO_FRAMES frames
    # then exit(0) -- so this call terminates and the loop advances to the next yaw.
    RECORD_VIDEO="$VID" MAX_VIDEO_FRAMES="$MAX_FRAMES" \
    RECORD_VIDEO_CAM_POS="$CAM_POS" RECORD_VIDEO_CAM_TARGET="$CAM_TARGET" \
    $XVFB python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG" --cfg_train "$TRAIN" \
        --test --play_dataset --headless --num_envs "$NUM_ENVS"
    # Copy the node-local video to the shared results dir immediately (survives the job).
    cp -v "$VID" "$DEST"/ 2>/dev/null || echo "[kin-obj] WARN: no video produced for yaw=$YAWDEG (check imageio + xvfb)"
done

echo ""
echo "=== sweep done. Videos in $DEST ==="
ls -lh "$DEST"/ 2>/dev/null
echo "=== Read the [objperturb] lines above for the numeric gap; WATCH the videos for the truth. ==="
echo "=== null-check: BODY==SOURCE, yaw=0 -> gap should be small AND the hand visibly on the object. ==="

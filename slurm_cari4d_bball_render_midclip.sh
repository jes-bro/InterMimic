#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-midclip"
#SBATCH --output=bball-midclip-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# MID-CLIP RENDER -- "did this arm learn the layup, given a running start?"
#
# WHY THIS EXISTS. slurm_cari4d_bball_render.sh renders the EVAL cfg, which is
# stateInit "Start": every episode begins at frame 0. That is exactly the regime
# r3_roll30 did NOT change. r3's fix was to make TRAINING sample mid-clip starts
# (rolloutLength 300 -> 30, so randint(0, 101-L) stops clamping to frame 0), so
# its effect -- if any -- is on competence when the policy begins NEAR the
# takeoff. A frame-0 render never once puts it in that situation, so an arm that
# genuinely improved at the layup segment would produce an identical video.
#
# WHAT THIS DOES. Renders a COMMITTED diagnostic cfg,
# omomo_cari4d_bball_diag_midclip.yaml, which differs from the arms' eval twin in:
#   stateInit     -> "Hybrid"   so the start-frame sampler actually runs
#   rolloutLength -> 50         so randint(0, 101-50) = starts over frames 0-50
#   human         -> false      so a 50-frame window PLAYS THROUGH instead of
#                               being cut the moment divergence trips
#   enableEvaluation -> False   metrics are undefined outside Start init
# Each episode is then exactly 50 frames from a random start, so a 600-frame
# video is ~12 independent windows. Watch for windows that begin mid-air or just
# before it.
#
# The 50 tracks r5_roll50's TRAIN regime (it was 30 while r3_roll30 was the live
# arm). The instrument is only honest if it shows the policy the start
# distribution it actually practised, so this value follows the arm under test --
# and watching an r3 checkpoint through it is a generalization read rather than
# r3's own distribution. The guard below asserts the value so the cfg and this
# script cannot drift apart silently.
#
# WHY human -> false AND NOT NO_TERM=1. NO_TERM only flips
# enableEarlyTermination, and that flag does not reach the reset: `reset` is
# computed from a `terminated` that already includes the kinematic flag, and the
# enable_early_termination check on the next line only sanitizes the RETURNED
# terminated (humanoid.py:552-555). The kinematic reset is applied
# unconditionally in compute_hoi_reset (intermimic.py:1733). Disabling it at the
# cfg is the only thing that works. terminationHeight is not touched because the
# cfg key is never read (hardcoded 0.3, humanoid.py:217) and body-fall measured
# 0.0% on both r2 and r3 anyway.
#
# This is a MEASURING INSTRUMENT, not an experiment: nothing trains on the diag
# cfg and no arm's cfg or checkpoint is touched. r3's and r4's eval twins are
# byte-identical, so ONE diag cfg serves both arms -- the CHECKPOINT selects the
# arm. Do not point CFG_ENV at an arm's eval cfg; the guards below will refuse it,
# because that is the render that produced the uninformative dribble loop.
#
#   sbatch slurm_cari4d_bball_render_midclip.sh
#   CHECKPOINT=checkpoints/smplx_cari4d_bball_r4_human1m/nn/mimic.pth \
#       FRAMES=600 sbatch slurm_cari4d_bball_render_midclip.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_cari4d_bball_r3_roll30/nn/mimic.pth}"
CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_diag_midclip.yaml}"
# 600 frames = ~12 windows of 50. Fewer and you may not draw a start near the
# takeoff at all; the starts are uniform over frames 0-50. Raise FRAMES to 1000
# for ~20 windows, the sample size the 30-frame version used to give.
FRAMES="${FRAMES:-600}"
[ -f "$CHECKPOINT" ] || { echo "[midclip] ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[ -f "$CFG_ENV" ]    || { echo "[midclip] ERROR: cfg not found: $CFG_ENV" >&2; exit 2; }

# --- VERIFY BEGIN ---   (extracted verbatim by tests/test_render_diag_patches.py)
# The cfg is COMMITTED, not patched at runtime -- but it can still be pointed at
# the wrong file. Assert the instrument rather than assume it: rendering the
# stock eval cfg by mistake reproduces the dribble-reset loop this script exists
# to replace, and looks like it worked.
grep -qE '^\s*stateInit:\s*"Hybrid"'   "$CFG_ENV" || { echo "[midclip] ERROR: $CFG_ENV is not Hybrid-init -- the start sampler would not run" >&2; exit 2; }
grep -qE '^\s*rolloutLength:\s*50\b'   "$CFG_ENV" || { echo "[midclip] ERROR: $CFG_ENV rolloutLength is not 50 -- starts would not concentrate on the pre-takeoff half, and the window would not match r5_roll50's training regime" >&2; exit 2; }
grep -qE '^\s*human:\s*[Ff]alse'       "$CFG_ENV" || { echo "[midclip] ERROR: $CFG_ENV still has the human reset on -- windows would be cut" >&2; exit 2; }
for KNOB in object igRatio contactSteps; do
    grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV" || { echo "[midclip] ERROR: resetThresholds.${KNOB} is not false -- window would still be cut" >&2; exit 2; }
done
# --- VERIFY END ---
echo "[midclip] instrument cfg: $CFG_ENV"
echo "[midclip] start frames drawn uniformly from 0..50 (clip 101, rolloutLength 50)"

mkdir -p renders
EXP=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="renders/midclip_${EXP}_${STAMP}.mp4"
# These arms save only the ROLLING mimic.pth (save_intermediate: False); reading
# it while training rewrites it is a race. Freeze a private copy first and keep
# it beside the video as provenance.
if [ "$(basename "$CHECKPOINT")" = "mimic.pth" ]; then
    SNAP="renders/${EXP}_midclip_${STAMP}_frozen.pth"
    cp "$CHECKPOINT" "$SNAP"
    CHECKPOINT="$SNAP"
    echo "[midclip] rolling checkpoint frozen -> $SNAP"
fi
echo "[midclip] ckpt=$CHECKPOINT frames=$FRAMES -> $OUT  (job=$SLURM_JOB_ID host=$(hostname))"

# The recorder's camera is FIXED and aimed at the ORIGIN, which is right for
# OMOMO but wrong here: this recon lives in the EgoExo4D camera's world frame and
# the subject sits metres away. Aim at the clip's own mean root instead, reading
# the clip from THIS cfg's motion_file rather than a hardcoded path (the bball
# arms span several exports whose world frames need not agree).
if [ -z "${RECORD_VIDEO_CAM_TARGET:-}" ]; then
    MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
    [ -n "$MOTION_DIR" ] || { echo "[midclip] ERROR: no motion_file in $CFG_ENV" >&2; exit 2; }
    CAM_CLIP=$(ls "$MOTION_DIR"/*.pt 2>/dev/null | head -1)
    [ -n "$CAM_CLIP" ] || { echo "[midclip] ERROR: no .pt clips under $MOTION_DIR" >&2; exit 2; }
    read -r CX CY CZ <<< "$(CAM_CLIP="$CAM_CLIP" python3 -c "
import os, torch
c = torch.load(os.environ['CAM_CLIP'], map_location='cpu')
m = c[:, 0:3].mean(dim=0)          # mean root position over the clip
print(f'{m[0]:.2f} {m[1]:.2f} {m[2]:.2f}')")"
    export RECORD_VIDEO_CAM_TARGET="${CX},${CY},1.4"   # aim a little high: this is a jump
    export RECORD_VIDEO_CAM_POS="$(python3 -c "print(f'{${CX}+3.0},{${CY}+3.0},2.5')")"
    echo "[midclip] auto camera: pos=$RECORD_VIDEO_CAM_POS target=$RECORD_VIDEO_CAM_TARGET (clip=$CAM_CLIP mean root ${CX},${CY},${CZ})"
fi

RECORD_VIDEO="$OUT" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG_ENV" \
        --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_train.yaml \
        --test --checkpoint "$CHECKPOINT" --headless --num_envs 1

echo "[midclip] done:"
ls -lh "$OUT"

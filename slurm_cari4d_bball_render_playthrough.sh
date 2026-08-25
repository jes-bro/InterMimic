#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-playthru"
#SBATCH --output=bball-playthru-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# PLAYTHROUGH RENDER -- "what does the policy DO at the takeoff, and is the
# reference even physically executable there?"
#
# WHY THIS EXISTS. slurm_cari4d_bball_render.sh renders the eval cfg with
# NO_TERM=1, whose comment claims the episode "plays through the failure". It
# does not. NO_TERM flips enableEarlyTermination, but that flag never reaches
# the reset: `reset` is computed from a `terminated` that already includes the
# kinematic flag, and the enable_early_termination check on the NEXT line only
# sanitizes the RETURNED terminated (humanoid.py:552-555). Worse, the kinematic
# reset is OR'd in unconditionally afterwards (intermimic.py:1733). So the human
# 0.5m divergence reset fires during the render exactly as in training: the
# policy diverges around frame 17-38, resets to frame 0 (eval is stateInit
# Start), and dribbles again. A 300-frame video is ~8 repeats of that loop --
# which looks identical for every arm, because the censoring lands in the same
# place for all of them, BEFORE the takeoff.
#
# WHAT THIS DOES. Renders a COMMITTED diagnostic cfg,
# omomo_cari4d_bball_diag_playthru.yaml, which differs from the arms' eval twin
# only in `human -> false` (and enableEvaluation -> False, since success rate is
# meaningless with resets off), leaving stateInit Start and rolloutLength 300. The episode then runs the
# FULL 101-frame clip from frame 0 uninterrupted, divergence and all, so you can
# see what the humanoid actually does when it reaches the takeoff instead of
# being cut before it. 300 frames = ~3 complete passes.
#
# HOW TO READ IT. This is the video that decides where the next GPU-week goes:
#   * reaches the takeoff and visibly ATTEMPTS a jump that falls short
#       -> training problem. Keep going on trainer knobs (r4's threshold, PSI).
#   * reaches the takeoff and does something physically incoherent, or the
#     reference ball is somewhere the humanoid could never follow
#       -> the reconstruction is unexecutable. Stop spending GPU; fix it
#          upstream in CARI4D (the 24cm ball offset, the unreliable last ~8
#          frames of ball trajectory).
# Note the humanoid WILL look wrong after divergence -- that is the point. Do
# not read "it looks bad" as failure; read WHERE and HOW it departs.
#
# terminationHeight is deliberately not touched: the cfg key is never read
# (hardcoded 0.3, humanoid.py:217) and body-fall measured 0.0% on both r2 and
# r3, so it is not what cuts these episodes.
#
# This is a MEASURING INSTRUMENT, not an experiment: nothing trains on the diag
# cfg and no arm's cfg or checkpoint is touched. r3's and r4's eval twins are
# byte-identical, so ONE diag cfg serves both arms -- the CHECKPOINT selects the
# arm. Do not point CFG_ENV at an arm's eval cfg; the guards below will refuse it,
# because that is exactly the render this script exists to replace.
#
#   sbatch slurm_cari4d_bball_render_playthrough.sh
#   CHECKPOINT=checkpoints/smplx_cari4d_bball_r4_human1m/nn/mimic.pth \
#       FRAMES=300 sbatch slurm_cari4d_bball_render_playthrough.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_cari4d_bball_r3_roll30/nn/mimic.pth}"
CFG_ENV="${CFG_ENV:-isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_diag_playthru.yaml}"
# The clip is 101 frames, so 300 = ~3 uninterrupted passes.
FRAMES="${FRAMES:-300}"
[ -f "$CHECKPOINT" ] || { echo "[playthru] ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[ -f "$CFG_ENV" ]    || { echo "[playthru] ERROR: cfg not found: $CFG_ENV" >&2; exit 2; }

# --- VERIFY BEGIN ---   (extracted verbatim by tests/test_render_diag_patches.py)
# Committed cfg, but assert it anyway -- pointing this at the stock eval twin
# silently reproduces the reset loop it exists to replace.
grep -qE '^\s*human:\s*[Ff]alse'       "$CFG_ENV" || { echo "[playthru] ERROR: $CFG_ENV still has the human reset on -- the pass would be cut" >&2; exit 2; }
grep -qE '^\s*stateInit:\s*"Start"'    "$CFG_ENV" || { echo "[playthru] ERROR: $CFG_ENV is not Start-init -- this instrument renders the full clip from frame 0" >&2; exit 2; }
grep -qE '^\s*rolloutLength:\s*300\b' "$CFG_ENV" || { echo "[playthru] ERROR: $CFG_ENV rolloutLength is not 300 -- episodes would be truncated, not played through" >&2; exit 2; }
for KNOB in object igRatio contactSteps; do
    grep -qE "^\s*${KNOB}:\s*[Ff]alse" "$CFG_ENV" || { echo "[playthru] ERROR: resetThresholds.${KNOB} is not false -- pass would still be cut" >&2; exit 2; }
done
# --- VERIFY END ---
echo "[playthru] instrument cfg: $CFG_ENV"
echo "[playthru] episodes now run the full 101-frame clip; expect ~$((FRAMES / 101)) passes in $FRAMES frames"

mkdir -p renders
EXP=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="renders/playthru_${EXP}_${STAMP}.mp4"
if [ "$(basename "$CHECKPOINT")" = "mimic.pth" ]; then
    SNAP="renders/${EXP}_playthru_${STAMP}_frozen.pth"
    cp "$CHECKPOINT" "$SNAP"
    CHECKPOINT="$SNAP"
    echo "[playthru] rolling checkpoint frozen -> $SNAP"
fi
echo "[playthru] ckpt=$CHECKPOINT frames=$FRAMES -> $OUT  (job=$SLURM_JOB_ID host=$(hostname))"

# Fixed camera aimed at the clip's own mean root, read from THIS cfg's
# motion_file (the bball arms span several exports whose world frames need not
# agree, so a hardcoded clip path can frame the wrong spot).
if [ -z "${RECORD_VIDEO_CAM_TARGET:-}" ]; then
    MOTION_DIR=$(grep -oE '^[[:space:]]*motion_file:[[:space:]]*\S+' "$CFG_ENV" | awk '{print $2}')
    [ -n "$MOTION_DIR" ] || { echo "[playthru] ERROR: no motion_file in $CFG_ENV" >&2; exit 2; }
    CAM_CLIP=$(ls "$MOTION_DIR"/*.pt 2>/dev/null | head -1)
    [ -n "$CAM_CLIP" ] || { echo "[playthru] ERROR: no .pt clips under $MOTION_DIR" >&2; exit 2; }
    read -r CX CY CZ <<< "$(CAM_CLIP="$CAM_CLIP" python3 -c "
import os, torch
c = torch.load(os.environ['CAM_CLIP'], map_location='cpu')
m = c[:, 0:3].mean(dim=0)          # mean root position over the clip
print(f'{m[0]:.2f} {m[1]:.2f} {m[2]:.2f}')")"
    export RECORD_VIDEO_CAM_TARGET="${CX},${CY},1.4"   # aim a little high: this is a jump
    export RECORD_VIDEO_CAM_POS="$(python3 -c "print(f'{${CX}+3.0},{${CY}+3.0},2.5')")"
    echo "[playthru] auto camera: pos=$RECORD_VIDEO_CAM_POS target=$RECORD_VIDEO_CAM_TARGET (clip=$CAM_CLIP mean root ${CX},${CY},${CZ})"
fi

RECORD_VIDEO="$OUT" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env "$CFG_ENV" \
        --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_train.yaml \
        --test --checkpoint "$CHECKPOINT" --headless --num_envs 1

echo "[playthru] done:"
ls -lh "$OUT"

#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="render-arms"
#SBATCH --output=render-arms-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Qualitative counterpart to the eval CSVs: render several arms rolling out the
# SAME pinned reference clip, so a success_rate can be looked at rather than
# read. All arms in ONE job -- at low queue priority the wait dominates, and the
# whole point is comparing them side by side, so splitting them is worse.
#
# USAGE (from repo root):
#   RUNS="src2_xf_aug_retarget src2_xf_aug_retarget_cpumotion" \
#   BODY=sub16 OBJECT=largetable ATTEMPTS=4 sbatch slurm_render_arms.sh
#
#   DRY=1 RUNS=... BODY=... OBJECT=... bash slurm_render_arms.sh   # print, run nothing
#
# Env:
#   RUNS      space-separated run ids (required). run@ckpt pins a checkpoint,
#             same syntax as slurm_eval_multi.sh. PIN THE CHECKPOINT when
#             comparing arms that have trained different amounts -- otherwise the
#             video compares training length, not the arms.
#   BODY      target body (required), e.g. sub16
#   OBJECT    object (required) -- with maxClipsPerObject=1 this is what pins the
#             single reference clip. Must be an object the SOURCE actually has
#             clips for, or intermimic.py raises on an empty motion set.
#   SOURCE    default sub2
#   ATTEMPTS  SEQUENTIAL episodes to record (default 4). Each episode is one
#             attempt at the pinned clip.
#             success_rate is best-of-~385 attempts, so ONE rollout is a single
#             draw; several at once shows the spread the number is summarising.
#   FRAMES    max recorded frames; 0 (default) = ATTEMPTS * 300, i.e. exactly
#             that many complete attempts
#   CLIP      clip to pin BY NAME, e.g. sub2_largetable_017. Empty = the first
#             after sorting, which for sub2/largetable is the SHORTEST of 17
#             (153 frames). An unknown name prints the available list.
#   REFERENCE=1
#             also render the ground-truth mocap replay of the same clip on the
#             same body, so what the policy is IMITATING can be watched next to
#             what it did. Written as REFERENCE__<body>__clip<N>.mp4
#   CAM_POS / CAM_TARGET
#             camera "x,y,z" (defaults 2.5,2.5,1.8 -> 0,0,0.9). Move CAM_POS
#             closer to the origin to fill more of the frame. The camera attaches
#             to ONE env, so there is no view that shows several envs at once --
#             which is why attempts are sequential episodes, not parallel envs.
#   OUT       output dir (default render_out/<BODY>)
#   ALLOW_MIXED_EPOCHS=1
#             render each arm at its LATEST checkpoint even though they sit at
#             different epochs. That is a capability snapshot ("what can each arm
#             do at its best so far"), NOT a method comparison -- arms that have
#             trained longer are advantaged. Without this, differing epochs are a
#             hard error, since the accidental version of this mistake is common
#             and invisible in the resulting figure.
#
# Output: <OUT>/<run>__<BODY>.mp4, one per arm. The script cross-checks that every
# arm loaded the SAME clip and exits 3 if not -- videos of different clips are not
# a comparison.

set -u
# Under sbatch, $0 is slurm's COPY of this script in the job spool dir, so
# dirname "$0" is NOT the repo -- slurm already sets cwd to the submit dir.
# Use SLURM_SUBMIT_DIR when present, fall back to the script's dir for a plain
# `bash slurm_*.sh` invocation, then verify we can actually see the repo.
cd "${SLURM_SUBMIT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}" || exit 2
if [ ! -f scripts/eval_one.sh ]; then
    echo "ERROR: cwd $(pwd) is not the InterMimic repo root (no scripts/eval_one.sh)." >&2
    echo "       Submit from the repo root: cd <repo> && sbatch $(basename "$0")" >&2
    exit 2
fi

RUNS="${RUNS:?set RUNS='run1 run2 ...'}"
BODY="${BODY:?set BODY=sub16 (the target body to drive)}"
OBJECT="${OBJECT:?set OBJECT=largetable (pins the clip with maxClipsPerObject=1)}"
SOURCE="${SOURCE:-sub2}"
ATTEMPTS="${ATTEMPTS:-4}"
FRAMES="${FRAMES:-0}"          # 0 = ATTEMPTS * episode length
CAM_POS="${CAM_POS:-2.5,2.5,1.8}"
CAM_TARGET="${CAM_TARGET:-0,0,0.9}"
CLIP="${CLIP:-}"
REF=""
[ "${REFERENCE:-0}" = 1 ] && REF="--reference"
OUT="${OUT:-render_out/${BODY}}"
MIXED=""
[ "${ALLOW_MIXED_EPOCHS:-0}" = 1 ] && MIXED="--allow-mixed-epochs"

if [ "${DRY:-0}" != 1 ]; then
    source ~/.bashrc
    conda deactivate
    conda activate intermimic-gym2
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

echo "[render] host=$(hostname) job=${SLURM_JOB_ID:-<none>}"
echo "[render] runs    : $RUNS"
echo "[render] body    : $BODY   source: $SOURCE   object: $OBJECT"
echo "[render] attempts: $ATTEMPTS   frames: $FRAMES"
echo "[render] out     : $OUT"
[ -n "$MIXED" ] && echo "[render] MIXED EPOCHS ALLOWED -- capability snapshot, not a method comparison"

# imageio is what actually writes the mp4; intermimic_players.py raises if
# RECORD_VIDEO is set without it, but checking here fails in seconds instead of
# after the first policy has loaded.
if [ "${DRY:-0}" != 1 ]; then
    python -c "import imageio, imageio_ffmpeg" 2>/dev/null || {
        echo "[render] ERROR: imageio / imageio-ffmpeg missing in this env." >&2
        echo "[render]        pip install imageio imageio-ffmpeg" >&2
        exit 2; }
fi

if [ "${DRY:-0}" = 1 ]; then
    echo "[render] DRY=1 -- would run:"
    echo "  python3 scripts/render_arms.py --runs $RUNS --body $BODY \\"
    echo "      --source $SOURCE --object $OBJECT --attempts $ATTEMPTS \\"
    echo "      --frames $FRAMES --cam-pos $CAM_POS --cam-target $CAM_TARGET --clip $CLIP --out-dir $OUT $MIXED $REF"
    exit 0
fi

python3 scripts/render_arms.py \
    --runs $RUNS \
    --body "$BODY" \
    --source "$SOURCE" \
    --object "$OBJECT" \
    --attempts "$ATTEMPTS" \
    --frames "$FRAMES" \
    --cam-pos "$CAM_POS" \
    --cam-target "$CAM_TARGET" \
    --clip "$CLIP" \
    --out-dir "$OUT" $MIXED $REF
rc=$?

echo
echo "================ RENDER DONE (rc=$rc) ================"
ls -la "$OUT" 2>/dev/null
echo "Pull them down with:"
echo "  rclone copy -P scdt:$(pwd)/$OUT ~/Downloads/$(basename "$OUT")_videos"
exit $rc

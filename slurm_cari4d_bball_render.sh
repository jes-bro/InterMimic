#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="bball-render"
#SBATCH --output=bball-render-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Plain in-sim mp4 of the bball overfit policy -- the Isaac Gym camera view,
# same as slurm_render_policy.sh gives for OMOMO (that script is hardwired to
# the multibody cfgs, hence this twin). The CHEAP qualitative look: capsule
# humanoid + ball in the sim env, no CARI4D rasteriser, no camera compositing
# (scripts/slurm_sim_figure.sh is the figure-quality path).
#
# Start-mode init, 1 env: the rollout begins at the reference start, runs until
# termination or the frame cap, and the video shows exactly where/how it dies
# (expected: ball departs from the reference around the frame-43 wall).
#
#   sbatch slurm_cari4d_bball_render.sh
#   CHECKPOINT=.../mimic_00020000.pth FRAMES=300 sbatch slurm_cari4d_bball_render.sh

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-checkpoints/smplx_cari4d_bball_overfit/nn/mimic.pth}"
FRAMES="${FRAMES:-300}"
[ -f "$CHECKPOINT" ] || { echo "[bball-render] ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 2; }

mkdir -p renders
OUT="renders/policy_bball_$(basename "$CHECKPOINT" .pth).mp4"
echo "[bball-render] ckpt=$CHECKPOINT frames=$FRAMES -> $OUT  (job=$SLURM_JOB_ID host=$(hostname))"

# The recorder's camera is FIXED at (3,3,2.5) aimed at the ORIGIN -- right for
# OMOMO clips, but this recon lives in the EgoExo4D camera's world frame, so
# the subject can be metres from origin and out of shot. Aim the (still fixed)
# camera at the clip's own mean root position instead; explicit
# RECORD_VIDEO_CAM_POS/TARGET still win if exported by the caller.
if [ -z "${RECORD_VIDEO_CAM_TARGET:-}" ]; then
    read -r CX CY CZ <<< "$(python3 - <<'PY'
import torch
c = torch.load('InterAct/behave_cari4d/sub100_bball_000.pt', map_location='cpu')
r = c[:, 0:3]  # root_pos over the clip
m = r.mean(dim=0)
print(f"{m[0]:.2f} {m[1]:.2f} {m[2]:.2f}")
PY
)"
    export RECORD_VIDEO_CAM_TARGET="${CX},${CY},1.0"
    export RECORD_VIDEO_CAM_POS="$(python3 -c "print(f'{${CX}+3.0},{${CY}+3.0},2.5')")"
    echo "[bball-render] auto camera: pos=$RECORD_VIDEO_CAM_POS target=$RECORD_VIDEO_CAM_TARGET (clip mean root ${CX},${CY},${CZ})"
fi

RECORD_VIDEO="$OUT" MAX_VIDEO_FRAMES="$FRAMES" \
    python -u -m intermimic.run --task InterMimic \
        --cfg_env isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_eval.yaml \
        --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_cari4d_bball_train.yaml \
        --test --checkpoint "$CHECKPOINT" --headless --num_envs 1

echo "[bball-render] done:"
ls -lh "$OUT"

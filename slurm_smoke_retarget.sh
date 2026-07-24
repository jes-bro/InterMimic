#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="smoke-retarget"
#SBATCH --output=smoke-retarget-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# GPU SMOKE for the per-body reference retargeting LOADER (approach A). This is the
# ONE step that needs Isaac Gym: it proves an env reads ITS OWN body's retargeted
# reference. 3 TRAINING bodies (sub2 identity + sub6/sub9), 512 envs, ~2 min to the
# first reset. It does NOT need to converge -- KILL it once the checks below pass.
#
# PASS, in order (tail the .out):
#   1. [retarget] expanded N clips x 3 bodies = 3N per-body references   -> expansion ran
#   2. NO 'FileNotFoundError: [retarget] ... missing'                    -> all files present
#   3. NO 'body-block invariant violated' assertion                     -> right body's ref
#   4. [mem] line + num_envs: 512, then mean_rewards a real number (not NaN)
#
# PREREQ: generate the smoke data FIRST (training bodies only):
#   python3 scripts/retarget_contact.py --batch --source sub2 \
#       --targets sub2 sub6 sub9 --iters 300 --workers 16 \
#       --out-dir InterAct/OMOMO_retarget_contact_smoke
# (matches retargetedMotionDir in omomo_smoke_retarget.yaml). Run from repo root.

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_smoke_retarget.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_smoke_retarget.yaml

# Fail early + loud if the smoke data isn't there, rather than deep in Isaac Gym.
DIR=InterAct/OMOMO_retarget_contact_smoke
for b in sub2 sub6 sub9; do
    n=$(ls "$DIR/$b/"*.pt 2>/dev/null | wc -l)
    echo "[smoke] $DIR/$b : $n retargeted clips"
    if [ "$n" -eq 0 ]; then
        echo "[smoke] ERROR: no retargeted clips for $b. Generate first (see header)."; exit 2
    fi
done

echo "[smoke] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_smoke_retarget/nn/"
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output checkpoints

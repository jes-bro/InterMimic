#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="batch-replay"
#SBATCH --output=batch-replay-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Run SEVERAL kinematic xbody replays on ONE gpu, back-to-back, so a pile of
# num_envs=1 replays costs ONE gpu slot instead of one-per-replay. Each replay is
# tiny (num_envs=1), so sequential on a single gpu is plenty fast.
#
#   BODIES="sub16 sub10 sub4 sub13" SOURCE=sub2 sbatch slurm_batch_replays.sh
#   CLIP=sub2_largetable_005 BODIES="sub16 sub10" SOURCE=sub2 sbatch slurm_batch_replays.sh
#
# It reuses slurm_replay_xbody.sh per body (its #SBATCH lines are ignored under
# `bash`), so all the CLIP/SOURCE/BODY logic + guards are shared -- no duplication.

set -u
cd "$(dirname "$0")"

BODIES="${BODIES:-sub16 sub10 sub4 sub13}"   # bodies to sweep, one replay each
SOURCE="${SOURCE:-sub2}"                     # motion source (fixed across bodies)
export CLIP="${CLIP:-}"                       # optional: pin ONE clip so every body sees identical motion
export FRAMES="${FRAMES:-300}"

echo "[batch-replay] source=$SOURCE bodies=($BODIES) clip=${CLIP:-<all>} on ONE gpu host=$(hostname) job=$SLURM_JOB_ID"
for b in $BODIES; do
    echo "======== replay body=$b (source=$SOURCE) ========"
    BODY="$b" SOURCE="$SOURCE" bash slurm_replay_xbody.sh \
        || echo "[batch-replay] body=$b FAILED -- continuing to next"
done
echo "[batch-replay] all done. Videos in renders/replayxb_*.mp4"

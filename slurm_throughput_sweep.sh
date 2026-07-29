#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="tp-sweep"
#SBATCH --output=sweep-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# THROUGHPUT SWEEP -- every probe, SEQUENTIALLY, on ONE GPU.
#
# Running them back-to-back on the same card is not just cheaper than one job per
# probe, it is a better experiment: fps is sensitive to node and to what else is
# running, so probes spread across nodes are not comparable to each other. Here
# every probe sees identical hardware, and the only thing that differs is the knob.
#
# Each probe is its OWN python process. That matters -- Isaac Gym does not release
# the card cleanly within a process, so probes must not share one, or probe N+1
# inherits N's allocations and reads wrong.
#
# WHAT THIS ANSWERS. Job 16390586 runs at ~7100 fps step vs the src2_xf_aug
# baseline's ~9000, cause unknown. Two hypotheses, opposite predictions:
#   (a) memory pressure -- card pinned 43.7/44G  => cpumotion HELPS
#   (b) gather locality -- num_motions 52 -> 2704 scatters the per-step gather
#                          across 7.87G instead of ~0.15G  => cpumotion HURTS
# The prize is not settling that, though: it is numEnvs. Per-step Python/dispatch
# cost is per-BATCH not per-env, so more envs amortise it, and the 9k baseline was
# itself at 4096 envs -- so this is the route ABOVE baseline, not back to it.
# Envs need VRAM; cpumotion (frees 7.87G, does NOT scale with envs) and the
# untuned upstream PhysX reservations are where that VRAM comes from.
#
# TIME: ~20 epochs at ~24s = ~8min, plus ~4min startup (2704 .pt loads, PhysX
# init) = ~12min/probe. Six probes ~= 75min. The 3h limit is slack for the
# bigger-numEnvs probes, which step slower.
#
# OVERRIDES:
#   PROBE_EPOCHS=20   epochs per probe (must stay < save_best_after=100)
#   PROBES='...'      newline-separated knob sets, '-' for the control. Default
#                     sweep below. Example:
#                       PROBES=$'-\nCPU_MOTION=1\nNUM_ENVS=8192' sbatch slurm_throughput_sweep.sh
#
# Runs from repo root. Read the RESULTS TABLE at the end of sweep-<jobid>.out.

set -u
source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"

BASE_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_retarget.yaml
BASE_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_retarget.yaml
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
OUT_DIR="sweep_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# Default sweep, ordered so the cheap diagnostics run BEFORE the expensive plays.
# If the sweep dies partway (walltime, node failure) you still have the answers
# that matter most.
#   1 control      -- the arm as it now stands (cpuMotionData ON, since job 16391296
#                     settled that at +43% fps step). Should read ~10250 fps step;
#                     if it does not, the harness is lying and stop reading here.
#   2 env6144      -- spend the freed 7.87G on +50% envs. THE question now.
#   3 buf10        -- free more VRAM at zero learning cost (20.0 is upstream, untuned)
#   4 env6144+buf  -- both memory sources, same env count
#   5 env8192+buf  -- greedy. May OOM; that is a RESULT, not a failure.
#
# NOTE: cpuMotionData is no longer a probe -- it is in the base cfg, so probe 1
# already has it. Setting CPU_MOTION=1 here would be a no-op duplicate.
PROBES="${PROBES:-$(cat <<'EOF'
-
NUM_ENVS=6144
BUFFER_MULT=10.0
BUFFER_MULT=10.0 NUM_ENVS=6144
BUFFER_MULT=10.0 NUM_ENVS=8192
EOF
)}"

echo "[sweep] host=$(hostname) job=$SLURM_JOB_ID epochs=$PROBE_EPOCHS -> $OUT_DIR/"
echo "[sweep] REFERENCE: job 16390586 = 7071-7410 fps step, 5340-5610 fps total, 43.7/44G"
echo "[sweep] probes:"
echo "$PROBES" | nl -ba -w4
echo

n=0
while IFS= read -r knobs; do
    [ -z "$knobs" ] && continue
    n=$((n + 1))
    [ "$knobs" = "-" ] && knobs=""

    # Tag from the knobs, so the log name and the table row are self-describing.
    tag=$(echo "$knobs" | sed -e 's/CPU_MOTION=1/cpumotion/' -e 's/BUFFER_MULT=/buf/' \
                              -e 's/NUM_ENVS=/env/' -e 's/CONTACT_PAIRS=/cp/' \
                              -e 's/  */_/g' -e 's/^_//' -e 's/_$//')
    [ -z "$tag" ] && tag="control"
    log="$OUT_DIR/${n}_${tag}.log"

    echo "=============================================================="
    echo "[sweep] probe $n/$(echo "$PROBES" | grep -c .): $tag   (knobs: ${knobs:-none})"
    echo "[sweep] log -> $log"
    echo "=============================================================="

    WORK="/tmp/sweep_${SLURM_JOB_ID}_${n}"
    mkdir -p "$WORK"

    # Translate the knob string into make_probe_cfg.py flags.
    cfg_args=()
    for kv in $knobs; do
        case "$kv" in
            CPU_MOTION=1)     cfg_args+=(--cpu-motion) ;;
            NUM_ENVS=*)       cfg_args+=(--num-envs "${kv#*=}") ;;
            BUFFER_MULT=*)    cfg_args+=(--buffer-mult "${kv#*=}") ;;
            CONTACT_PAIRS=*)  cfg_args+=(--contact-pairs "${kv#*=}") ;;
            *) echo "[sweep] ERROR: unrecognised knob '$kv' -- refusing to run a" >&2
               echo "[sweep]        probe whose config I cannot express." >&2
               rm -rf "$WORK"; continue 2 ;;
        esac
    done

    if ! python3 scripts/make_probe_cfg.py \
            --base-env "$BASE_ENV" --base-train "$BASE_TRAIN" \
            --out-env "$WORK/env.yaml" --out-train "$WORK/train.yaml" \
            --tag "$tag" --epochs "$PROBE_EPOCHS" "${cfg_args[@]}"; then
        echo "[sweep] cfg generation FAILED for $tag; skipping" | tee "$log"
        rm -rf "$WORK"
        continue
    fi

    # Header the summariser reads back out of the log.
    { echo "[probe] tag=$tag host=$(hostname) job=$SLURM_JOB_ID"; } | tee "$log"

    # `|| true`: a probe that OOMs must NOT kill the sweep -- for the greedy
    # numEnvs probes, OOM is one of the outcomes we are trying to measure.
    python -u -m intermimic.run \
        --task InterMimic \
        --cfg_env "$WORK/env.yaml" \
        --cfg_train "$WORK/train.yaml" \
        --headless \
        --output "$WORK/checkpoints" 2>&1 | tee -a "$log" || true

    rm -rf "$WORK"
    echo "[sweep] probe $n done"
    echo
done <<< "$PROBES"

echo
echo "########################### RESULTS TABLE ###########################"
python3 scripts/summarize_throughput.py "$OUT_DIR"/*.log --warmup 5 \
    || echo "[sweep] summariser failed -- the per-probe logs are in $OUT_DIR/"

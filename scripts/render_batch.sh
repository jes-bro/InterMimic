#!/bin/sh
# render_batch.sh -- a whole batch of qualitative renders from one file, instead
# of pasting one long command per video.
#
# WHY. A demo batch is seven or eight near-identical commands differing in three
# words, which is how the first batch ended up with names like r2_rev_401.mp4
# that do not say which BODY they show. Here the batch is a file you edit, and
# the output name is generated from what was actually rendered.
#
# BATCH FILE. One render per line, whitespace separated:
#
#     <body> <source> <clip> <arm-key>
#     sub13  sub401   sub401_bballd03s01rev015at_002.pt  act
#     sub13  sub15    sub15_largetable_000.pt            omomo
#
# Blank lines and lines starting with # are ignored, so a batch can be commented
# and lines can be parked without deleting them.
#
# ARM KEYS map to the arm and checkpoint, so a line cannot pair an activity clip
# with the OMOMO policy by typo:
#     act    -> student_g3_act_xf_ret_nvadlr__f0    @ mimic_00100000.pth
#     omomo  -> student_g3_omomo_xf_ret_nvadlr__f0  @ mimic_00029000.pth
#
# OUTPUT NAMES are built, not chosen:
#     body-<body>__src-<clip stem>__<tag>.mp4     (tag: act100k / xf29k)
# so a video always states its own body, source clip and policy.
#
# GPUS: renders are dealt round-robin over GPUS and each card works through its
# share SEQUENTIALLY -- two Isaac Gym renders on one card is how you lose a whole
# batch to an OOM at minute nine.
#
# Usage (repo root, conda env active, LD_LIBRARY_PATH exported):
#   sh scripts/render_batch.sh scripts/batches/sub13_probe.txt
#   GPUS="1 2 3" sh scripts/render_batch.sh my_batch.txt
#   DRY=1 sh scripts/render_batch.sh my_batch.txt      # print the plan only
#   OUTDIR=$HOME/renders4 LOGDIR=$HOME/renderlogs sh scripts/render_batch.sh b.txt
#
# Run it under nohup to survive a dropped ssh:
#   nohup sh scripts/render_batch.sh batch.txt > batch.log 2>&1 &
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

BATCH="${1:?usage: sh scripts/render_batch.sh <batch-file>}"
[ -f "$BATCH" ] || { echo "ERROR: no batch file at $BATCH" >&2; exit 2; }

GPUS="${GPUS:-1 2 3 4 5 6 7}"          # card 0 is usually training
OUTDIR="${OUTDIR:-$HOME/renders}"
LOGDIR="${LOGDIR:-$HOME/renderlogs}"

# arm_key -> "<arm> <checkpoint> <tag>". Kept here rather than in the batch file
# so every line of a batch is three names and cannot mismatch policy to data.
arm_spec() {
  case "$1" in
    act)   echo "student_g3_act_xf_ret_nvadlr__f0 checkpoints/smplx_student_g3_act_xf_ret_nvadlr__f0/nn/mimic_00100000.pth act100k" ;;
    omomo) echo "student_g3_omomo_xf_ret_nvadlr__f0 checkpoints/smplx_student_g3_omomo_xf_ret_nvadlr__f0/nn/mimic_00029000.pth xf29k" ;;
    *)     return 1 ;;
  esac
}

# Parse first, launch second: a typo on the last line should not be discovered
# after six cards are already busy.
plan=""
n=0
n_gpu=$(echo "$GPUS" | wc -w)
while read -r body source clip arm rest; do
  case "${body:-}" in ''|\#*) continue ;; esac
  [ -n "${source:-}" ] && [ -n "${clip:-}" ] && [ -n "${arm:-}" ] || {
    echo "ERROR: $BATCH line '$body $source $clip $arm': want <body> <source> <clip> <arm-key>" >&2; exit 2; }
  [ -z "${rest:-}" ] || { echo "ERROR: $BATCH line for $body: extra field '$rest'" >&2; exit 2; }
  spec=$(arm_spec "$arm") || { echo "ERROR: $BATCH: unknown arm key '$arm' (want: act, omomo)" >&2; exit 2; }
  gpu=$(echo "$GPUS" | cut -d' ' -f$(( n % n_gpu + 1 )))
  stem=${clip%.pt}
  tag=$(echo "$spec" | awk '{print $3}')
  plan="$plan$gpu|$body|$source|$clip|$(echo "$spec" | awk '{print $1}')|$(echo "$spec" | awk '{print $2}')|body-${body}__src-${stem}__${tag}
"
  n=$((n + 1))
done < "$BATCH"

[ "$n" -gt 0 ] || { echo "ERROR: $BATCH has no render lines" >&2; exit 2; }

mkdir -p "$OUTDIR" "$LOGDIR"
echo "== render batch: $n render(s) over $n_gpu GPU(s) [$GPUS]"
echo "   videos -> $OUTDIR"
echo "   logs   -> $LOGDIR"
echo "$plan" | while IFS='|' read -r gpu body source clip arm ckpt name; do
  [ -n "${gpu:-}" ] || continue
  echo "   gpu $gpu  $body x $source  $clip  -> $name.mp4"
done

[ "${DRY:-0}" = 1 ] && { echo "   (DRY=1: not running)"; exit 0; }

for g in $GPUS; do
  # One subshell per card; its renders run one after another, never at once.
  (
    echo "$plan" | while IFS='|' read -r gpu body source clip arm ckpt name; do
      [ "${gpu:-}" = "$g" ] || continue
      env BODY="$body" SOURCE="$source" CLIP="$clip" ARM="$arm" CKPT="$ckpt" \
          GPU="$g" OUT="$OUTDIR/$name.mp4" \
          sh scripts/render_pair.sh > "$LOGDIR/$name.log" 2>&1 \
        || echo "[gpu $g] $name FAILED (see $LOGDIR/$name.log)" >&2
    done
  ) &
done
wait

echo "== done:"
ls -lh "$OUTDIR"

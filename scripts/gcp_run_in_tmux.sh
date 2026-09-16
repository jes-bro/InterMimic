#!/bin/bash
# gcp_run_in_tmux.sh -- start a slurm-style launcher on a GCP VM (no Slurm) in a
# detached tmux session, restarting on exit so the launcher's own auto-resume
# picks up the latest checkpoint after a crash or host maintenance.
#
#   sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_mlp_ret_stock__f0.sh omomo_mlp
#   tmux attach -t omomo_mlp        # Ctrl-b d to detach; NEVER Ctrl-c
#
# The `#SBATCH --output=` line in a launcher is a comment under bash, so without
# this wrapper a VM run has NO log file -- only tmux scrollback. Here stdout and
# stderr are tee'd to <name>-gcp.log in the repo root (append, so restarts keep
# one continuous history), and the tmux scrollback still shows it live.
#
# Run from the repo root. `until` retries with a 30 s pause; the launcher
# refuses to start fresh over an existing checkpoint (its resume block), so a
# restart is always a resume.
set -u
LAUNCHER="${1:?usage: gcp_run_in_tmux.sh <launcher.sh> <tmux-session-name>}"
SESSION="${2:?usage: gcp_run_in_tmux.sh <launcher.sh> <tmux-session-name>}"
[ -f "$LAUNCHER" ] || { echo "ERROR: no such launcher: $LAUNCHER" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (tmux attach -t $SESSION)" >&2; exit 1
fi
LOG="$(basename "$LAUNCHER" .sh)-gcp.log"
tmux new-session -d -s "$SESSION" \
    "until bash '$LAUNCHER' 2>&1 | tee -a '$LOG'; do echo '[gcp] launcher exited, restarting in 30 s' | tee -a '$LOG'; sleep 30; done"
echo "started tmux session '$SESSION' running $LAUNCHER; log -> $LOG"
echo "  watch:   tmux attach -t $SESSION      (detach: Ctrl-b d)"
echo "  or:      tail -f $LOG"

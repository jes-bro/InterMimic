#!/bin/bash
# gcp_run_in_tmux.sh -- start a slurm-style launcher on a GCP VM (no Slurm) in a
# detached tmux session, once. It does NOT restart on exit.
#
#   sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_mlp_ret_stock__f0.sh omomo_mlp
#   tmux attach -t omomo_mlp        # Ctrl-b d to detach; NEVER Ctrl-c
#
# The `#SBATCH --output=` line in a launcher is a comment under bash, so without
# this wrapper a VM run has NO log file -- only tmux scrollback. Here stdout and
# stderr are tee'd to <name>-gcp.log in the repo root (append, so restarts keep
# one continuous history), and the tmux scrollback still shows it live.
#
# Run from the repo root. To relaunch after an exit, run this script again: the
# launcher refuses to start fresh over an existing checkpoint (its resume
# block), so a manual relaunch is always a resume, never an overwrite.
set -u
LAUNCHER="${1:?usage: gcp_run_in_tmux.sh <launcher.sh> <tmux-session-name>}"
SESSION="${2:?usage: gcp_run_in_tmux.sh <launcher.sh> <tmux-session-name>}"
[ -f "$LAUNCHER" ] || { echo "ERROR: no such launcher: $LAUNCHER" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (tmux attach -t $SESSION)" >&2; exit 1
fi
LOG="$(basename "$LAUNCHER" .sh)-gcp.log"
# Runs the launcher ONCE. No automatic restart (Jess, 2026-09-16: never asked
# for auto-resume). If it exits, the exit status is logged and the tmux window
# stays open with a shell so the last screen can be read; relaunching is a
# deliberate act -- rerun this script, and the launcher's own resume block
# then continues from the run's latest checkpoint rather than starting fresh
# over it.
tmux new-session -d -s "$SESSION" bash -c \
    "set -o pipefail; bash '$LAUNCHER' 2>&1 | tee -a '$LOG'; echo \"[gcp] launcher exited with status \$? -- NOT restarting\" | tee -a '$LOG'; exec bash"
echo "started tmux session '$SESSION' running $LAUNCHER; log -> $LOG"
echo "  watch:   tmux attach -t $SESSION      (detach: Ctrl-b d)"
echo "  or:      tail -f $LOG"

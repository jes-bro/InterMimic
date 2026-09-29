#!/usr/bin/env python3
"""One-line verdict per REPLAY_TRACE npz: does the physics step leave the written
pose (a wound joint angle, an exploding velocity) on that body?

  python3 scripts/summarize_replay_traces.py ~/Downloads/sweep/trace_*.npz

Columns: max post-step joint velocity (rad/s), frames with any joint > 50 rad/s,
frames with any dof > 0.2 rad off the written value, dof-frames wound by > 3 rad,
and the worst dof. A clean body reads like sub2: ~10 rad/s, 0, 0, 0.
Background: sub4's stock MJCF winds its left knee by 2*pi and the +/-pi joint
limit then explodes the leg (2026-09-28); explicit <inertial> elements
(scripts/mjcf_add_inertials.py) fix it. This flags any other body with the
same trap.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--vel", type=float, default=50.0, help="velocity threshold (rad/s)")
    a = ap.parse_args()
    print(f"{'trace':40s} {'frames':>6s} {'max|v|':>7s} {'v>'+str(int(a.vel)):>6s} {'off>0.2':>7s} {'wound':>6s}  worst dof")
    flagged = []
    for f in a.traces:
        d = np.load(f, allow_pickle=True)
        names = [str(n) for n in d["dof_names"]]
        W, M, VM = d["written"], d["after_sim"], d["vafter_sim"]
        dev = np.abs(M - W)
        n_v = int((np.abs(VM).max(1) > a.vel).sum()); n_off = int((dev.max(1) > 0.2).sum()); n_w = int((dev > 3).sum())
        j = int(dev.max(0).argmax())
        tag = os.path.basename(f).replace("replay_trace_", "").replace(".npz", "")
        print(f"{tag:40s} {len(W):6d} {np.abs(VM).max():7.1f} {n_v:6d} {n_off:7d} {n_w:6d}  {names[j]} {dev[:, j].max():.2f}"
              + ("   <<< TRAP" if n_off else ""))
        if n_off:
            flagged.append(tag)
    print(f"\n{len(flagged)} of {len(a.traces)} traces show the trap: {flagged}")


if __name__ == "__main__":
    main()

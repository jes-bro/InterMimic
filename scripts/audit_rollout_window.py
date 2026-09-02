#!/usr/bin/env python3
"""What rolloutLength actually does to a motion dir: start coverage and PSI.

rolloutLength is ONE global number applied to clips of every length, and it
controls two things at once, in opposite directions:

  episode length   an episode ends rollout_length-1 steps after its start
                   (humanoid.py:553)
  start coverage   starts are drawn from [0, max(1, T - rollout_length))
                   (intermimic.py:1269 for Random/Hybrid; _reset_hybrid_state_init
                   builds its CDF over the same truncated range at :1297)

So a LARGER rolloutLength gives longer episodes but a NARROWER start window --
and once rollout_length >= T-1 the window collapses to a single frame and the
sampler can only ever return frame 0. stateInit: Hybrid then behaves exactly
like stateInit: Start, silently, with no warning anywhere. That is the bug the
bball r2_warm header documents ("0.0% of 16,666,993 episodes completed").

There is a third consequence: PSI only harvests states from a clip when
T >= rollout_length (psi_update.py:104). Set rolloutLength above your clip
lengths and physicalBufferSize does nothing at all, however it is configured.

  python3 scripts/audit_rollout_window.py InterAct/OMOMO_new
  python3 scripts/audit_rollout_window.py InterAct/OMOMO_new InterAct/behave_cari4d_optj3d_cf2
  python3 scripts/audit_rollout_window.py <dir> --rollouts 30,50,100,300 --per-clip

Reads only the tensors' shapes -- no sim, no GPU. Handles both flat dirs
(<dir>/*.pt) and the body-major retarget layout (<dir>/<body>/*.pt).
"""
import argparse
import glob
import os
import sys

import torch

DEFAULT_ROLLOUTS = (30, 50, 100, 300)


def clip_lengths(motion_dir):
    """-> {relative path: frame count}. Frame count is what the task uses as
    max_episode_length (intermimic.py:740 takes hoi_data.shape[0]; startk and
    initk are both 0 for the InterMimic task)."""
    paths = sorted(glob.glob(os.path.join(motion_dir, "*.pt"))
                   + glob.glob(os.path.join(motion_dir, "*", "*.pt")))
    out = {}
    for p in paths:
        t = torch.load(p, map_location="cpu", weights_only=False)
        out[os.path.relpath(p, motion_dir)] = int(t.shape[0])
    return out


def start_window(T, R):
    """Width of the start-sampling range, exactly as the task computes it."""
    return max(1, T - R)


def summarize(lengths, rollouts=DEFAULT_ROLLOUTS):
    """-> [{rollout, mean_window, n_pinned, n_psi, n}] for each rollout length.

    pinned = the start window collapsed to one frame, so Hybrid == Start
    psi    = clip is long enough for PSI to harvest from it (T >= R)
    """
    vals = list(lengths.values())
    rows = []
    for R in rollouts:
        widths = [start_window(T, R) for T in vals]
        rows.append(dict(rollout=R,
                         mean_window=sum(widths) / len(widths) if widths else 0.0,
                         n_pinned=sum(1 for w in widths if w == 1),
                         n_psi=sum(1 for T in vals if T >= R),
                         n=len(vals)))
    return rows


def max_safe_rollout(lengths):
    """Largest rolloutLength that pins NO clip to frame 0 and leaves every clip
    PSI-eligible.

    pinned when T - R <= 1, i.e. R >= T - 1  -> need R <= min(T) - 2
    PSI needs T >= R                         -> need R <= min(T)
    The first bound is the binding one. Returns None when even R=1 cannot
    satisfy it (a clip of 2 frames or fewer).
    """
    if not lengths:
        return None
    r = min(lengths.values()) - 2
    return r if r >= 1 else None


def percentiles(vals):
    s = sorted(vals)
    n = len(s)
    return dict(n=n, min=s[0], p25=s[n // 4], median=s[n // 2],
                p75=s[(3 * n) // 4], max=s[-1])


def report(motion_dir, lengths, rollouts, per_clip=False):
    lines = [f"== {motion_dir}"]
    if not lengths:
        lines.append("  no .pt clips found")
        return "\n".join(lines)

    p = percentiles(list(lengths.values()))
    lines.append(f"  {p['n']} clips: min={p['min']} p25={p['p25']} "
                 f"median={p['median']} p75={p['p75']} max={p['max']} frames")

    # A body-major dir holds the same clip retargeted per body; retargeting must
    # not change frame counts, so differing lengths for one clip name is a fault.
    by_name = {}
    for rel, T in lengths.items():
        by_name.setdefault(os.path.basename(rel), set()).add(T)
    ragged = {k: v for k, v in by_name.items() if len(v) > 1}
    if ragged:
        lines.append(f"  WARNING: {len(ragged)} clip name(s) have differing frame "
                     f"counts across bodies -- retargeting must preserve length:")
        for k, v in list(ragged.items())[:5]:
            lines.append(f"    {k}: {sorted(v)}")

    for row in summarize(lengths, rollouts):
        flag = ""
        if row["n_pinned"]:
            flag = "  <-- Hybrid silently == Start on these"
        lines.append(
            f"  R={row['rollout']:4d}: mean start window {row['mean_window']:7.1f} "
            f"frames | pinned to frame 0: {row['n_pinned']:3d}/{row['n']} | "
            f"PSI-eligible: {row['n_psi']:3d}/{row['n']}{flag}")

    safe = max_safe_rollout(lengths)
    lines.append(f"  max rolloutLength with no pinning and full PSI coverage: "
                 + (f"{safe}" if safe else "NONE (a clip is too short)"))

    if per_clip:
        lines.append("  per clip:")
        for rel, T in sorted(lengths.items(), key=lambda kv: kv[1]):
            windows = " ".join(f"R{R}:{start_window(T, R)}" for R in rollouts)
            lines.append(f"    {rel:>44}  T={T:5d}  {windows}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("motion_dirs", nargs="+", help="motion dir(s) to audit")
    ap.add_argument("--rollouts", default=",".join(str(r) for r in DEFAULT_ROLLOUTS),
                    help="comma-separated rolloutLength values to evaluate")
    ap.add_argument("--per-clip", action="store_true", help="list every clip")
    args = ap.parse_args(argv)

    try:
        rollouts = [int(r) for r in args.rollouts.split(",") if r.strip()]
    except ValueError:
        print(f"ERROR: --rollouts must be integers, got {args.rollouts!r}", file=sys.stderr)
        return 1
    if not rollouts:
        print("ERROR: --rollouts is empty", file=sys.stderr)
        return 1

    all_safe, missing = [], []
    for d in args.motion_dirs:
        if not os.path.isdir(d):
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            missing.append(d)
            continue
        lengths = clip_lengths(d)
        print(report(d, lengths, rollouts, per_clip=args.per_clip))
        print()
        safe = max_safe_rollout(lengths)
        if safe is not None:
            all_safe.append(safe)
    if missing:
        return 1

    # rolloutLength is global: one value has to serve every dir in a run.
    if len(args.motion_dirs) > 1 and all_safe:
        print(f"across all dirs, max rolloutLength that pins nothing and keeps "
              f"full PSI coverage: {min(all_safe)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

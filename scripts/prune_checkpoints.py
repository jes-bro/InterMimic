#!/usr/bin/env python3
"""Thin a run's mid-training checkpoints, keeping every Nth, to reclaim disk.

checkpoints/ reached 1.8 TB because the teacher runs save intermediates: a
single .pth is ~150 MB and a 94 GB run directory holds ~600 of them. The
training TRAJECTORY is worth keeping; 600 samples of it is not.

PROTECTED BY DEFAULT -- never pruned unless you name them explicitly:
    smplx_cari4d_bball*     the EgoExo4D arms (save_intermediate: False, so
                            they hold one checkpoint each and have nothing to
                            thin), several of them still training
    smplx_teacher_g2*       the gen-2 grid, kept whole

WHAT IT NEVER DELETES, regardless of flags:
  * the newest checkpoint in a run -- that is the run's result
  * the oldest -- the trajectory needs a left endpoint
  * any file named exactly mimic.pth -- the rolling checkpoint a resubmit
    resumes from, and what every render and eval command points at
  * anything in a run touched within --live-minutes (default 120), because that
    run is probably still training

It is a DRY RUN unless you pass --apply, and it reports reclaimed space as it
goes rather than only at the end.

    # see what would happen
    python3 scripts/prune_checkpoints.py --pattern 'smplx_curriculum*' --keep-every 10

    # ...then do it
    python3 scripts/prune_checkpoints.py --pattern 'smplx_curriculum*' --keep-every 10 --apply

Ordering is by the epoch number in the filename when there is one
(mimic_00012000.pth -> 12000), else by mtime. A run whose names parse
inconsistently is reported and SKIPPED rather than guessed at -- interleaving
the two orderings would produce a meaningless "every Nth", and a wrong keep-set
is unrecoverable.
"""
import argparse
import re
import sys
import time
from pathlib import Path

EPOCH_RE = re.compile(r"(\d+)(?=\.pth$)")
ROLLING = "mimic.pth"
# NOTE: --protect REPLACES this list, it does not extend it (see `protect = ...`
# below), so anything named on the CLI must repeat the patterns it still wants.
# That is why the live grid belongs HERE and not in a command line: a prune run
# that passed --protect for one family would otherwise silently un-protect the
# others.
# SUBSTRING patterns (*x*), not prefixes. The prefix forms smplx_teacher_g2* and
# smplx_cari4d_bball* only protected runs whose directory happened to follow the
# usual naming; a run named collab_g3_..., smplx_teacher_bball_..., or anything
# else off-convention was NOT protected and would have been thinned. Since the
# cost of a too-WIDE protect pattern is some disk left unreclaimed, and the cost
# of a too-NARROW one is destroying the current experiment's checkpoints, these
# are deliberately wide (Jess, 2026-09-06: "any directory with a g3 or g2 in it
# or a bball in it (in case your naming convention is wrong) is spared").
#
# NOTE: --protect REPLACES this list, it does not extend it (see `protect = ...`
# below), so anything named on the CLI must repeat the patterns it still wants.
DEFAULT_PROTECT = [
    "*bball*",   # EgoExo4D arms: save_intermediate False, one ckpt each
    "*g2*",      # the gen-2 grid, kept whole
    "*g3*",      # the gen-3 arms -- the CURRENT experiment; several were still
                 # training when /simurgh2/projects hit 100% (2026-09-06) and
                 # they resume from their own checkpoints, so thinning them
                 # costs real work.
]


def human(n):
    n = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}"
        n /= 1024


def run_dirs(root, patterns):
    """Every matching run's checkpoint directory, sorted."""
    out = []
    for pat in patterns:
        for d in sorted(root.glob(pat)):
            if not d.is_dir():
                continue
            nn = d / "nn"
            if nn.is_dir():
                out.append((d.name, nn))
            elif any(d.glob("*.pth")):
                out.append((d.name, d))       # some runs write straight into the run dir
    return out


def order_checkpoints(nn):
    """(ordered oldest-first, how) or (None, reason) when the ordering is ambiguous."""
    files = [p for p in nn.glob("*.pth") if p.name != ROLLING]
    if not files:
        return [], "no prunable checkpoints"
    nums = {p: EPOCH_RE.search(p.name) for p in files}
    have = [p for p, m in nums.items() if m]
    if len(have) == len(files):
        return sorted(files, key=lambda p: int(nums[p].group(1))), "epoch in filename"
    if not have:
        return sorted(files, key=lambda p: p.stat().st_mtime), "mtime"
    return None, (f"{len(have)}/{len(files)} names carry an epoch number -- "
                  f"refusing to guess an ordering")


def newest_mtime(nn):
    return max((p.stat().st_mtime for p in nn.glob("*")), default=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("checkpoints"),
                   help="checkpoints directory (default: ./checkpoints)")
    p.add_argument("--pattern", action="append", required=True,
                   help="glob for run directories; repeatable")
    p.add_argument("--keep-every", type=int, default=10,
                   help="keep 1 of every N checkpoints (default 10)")
    p.add_argument("--live-minutes", type=int, default=120,
                   help="skip runs touched within this many minutes (default 120)")
    p.add_argument("--protect", action="append", default=None,
                   help=f"glob for run dirs to never touch; repeatable. "
                        f"Default: {' '.join(DEFAULT_PROTECT)}")
    p.add_argument("--apply", action="store_true",
                   help="actually delete. Without it this only reports.")
    args = p.parse_args()

    if args.keep_every < 2:
        raise SystemExit("--keep-every must be >= 2 (1 would delete nothing)")
    protect = args.protect if args.protect is not None else DEFAULT_PROTECT
    if not args.root.is_dir():
        raise SystemExit(f"no checkpoints dir at {args.root} "
                         f"(run from the repo root, or pass --root)")

    protected = {d.name for pat in protect for d in args.root.glob(pat)}
    targets = run_dirs(args.root, args.pattern)
    if not targets:
        raise SystemExit(f"no run directories matched {args.pattern} under {args.root}")

    cutoff = time.time() - args.live_minutes * 60
    total_del = total_bytes = 0
    skipped, plan = [], []

    for name, nn in targets:
        if name in protected:
            skipped.append((name, "PROTECTED"))
            continue
        if newest_mtime(nn) > cutoff:
            skipped.append((name, f"touched in the last {args.live_minutes} min -- looks live"))
            continue
        ordered, how = order_checkpoints(nn)
        if ordered is None:
            skipped.append((name, how))
            continue
        if len(ordered) <= 2:
            skipped.append((name, f"only {len(ordered)} prunable -- nothing to thin"))
            continue

        # Keep every Nth counting BACK from the newest, so the most recent
        # checkpoint is always kept and the spacing is regular near the end --
        # the part anyone actually looks at.
        keep = {ordered[i] for i in range(len(ordered) - 1, -1, -args.keep_every)}
        keep.add(ordered[0])          # left endpoint
        keep.add(ordered[-1])         # the result

        doomed = [q for q in ordered if q not in keep]
        nbytes = sum(q.stat().st_size for q in doomed)
        plan.append((name, len(ordered), len(keep), doomed, nbytes, how))
        total_del += len(doomed)
        total_bytes += nbytes

    print(f"=== {'APPLYING' if args.apply else 'DRY RUN'} -- keep 1 of every "
          f"{args.keep_every} ===\n", flush=True)
    for name, n_all, n_keep, doomed, nbytes, how in plan:
        print(f"  {name}\n      {n_all} checkpoints ({how}) -> keep {n_keep}, "
              f"delete {len(doomed)}, reclaim {human(nbytes)}", flush=True)
    if skipped:
        print("\n  skipped:")
        for name, why in skipped:
            print(f"      {name}: {why}")
    print(f"\n  TOTAL: {total_del} files, {human(total_bytes)} across "
          f"{len(plan)} runs", flush=True)

    if not args.apply:
        print("\n  Dry run -- nothing deleted. Re-run with --apply to do it.")
        return 0
    if not plan:
        print("\n  Nothing to do.")
        return 0

    print("\n=== deleting ===", flush=True)
    freed = done = 0
    started = time.time()
    for name, _, _, doomed, nbytes, _ in plan:
        run_freed = 0
        for i, f in enumerate(doomed, 1):
            try:
                sz = f.stat().st_size
                f.unlink()
                run_freed += sz
                freed += sz
                done += 1
            except OSError as e:
                print(f"  WARNING could not delete {f}: {e}", flush=True)
            # Progress inside a big run -- 600 deletions over NFS is slow enough
            # that per-run reporting alone looks like a hang.
            if i % 25 == 0:
                pct = 100 * freed / total_bytes if total_bytes else 100
                print(f"    {name}: {i}/{len(doomed)} files, "
                      f"{human(run_freed)} here | total {human(freed)} "
                      f"({pct:.0f}%) in {time.time() - started:.0f}s", flush=True)
        pct = 100 * freed / total_bytes if total_bytes else 100
        print(f"  done {name}: freed {human(run_freed)} "
              f"| running total {human(freed)} ({pct:.0f}%)", flush=True)

    print(f"\n  deleted {done}/{total_del} files, reclaimed {human(freed)} "
          f"in {time.time() - started:.0f}s")
    print("  check the volume with:  df -h /simurgh2/projects")
    return 0


if __name__ == "__main__":
    sys.exit(main())

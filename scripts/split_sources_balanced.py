#!/usr/bin/env python3
"""Partition a set of OMOMO sources into two DISJOINT halves of given sizes so the
halves carry (as nearly as possible) the same number of clips, and both still
cover every object.

Why exhaustive: with 13 sources there are only C(13,6) = 1716 ways to choose the
6-source half, so the truly most balanced split costs nothing to find, and a
greedy pass (largest first, to the lighter half) is not guaranteed to find it.

    python3 scripts/split_sources_balanced.py --motion-dir ~/new_one/OMOMO_new \\
        --sources sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17 \\
        --size-a 6

Prints the top few candidates (by clip-count gap) with object coverage, so a
tie can be broken by hand rather than by iteration order.
"""
import argparse
import itertools
import os
import sys


def index_clips(motion_dir, sources):
    """{source: {object: n_clips}} from filenames sub<N>_<object>_<idx>.pt."""
    want = set(sources)
    out = {s: {} for s in sources}
    for f in os.listdir(motion_dir):
        if not f.endswith(".pt"):
            continue
        parts = f[:-3].split("_")
        s, obj = parts[0], parts[-2]
        if s in want:
            out[s][obj] = out[s].get(obj, 0) + 1
    missing = [s for s in sources if not out[s]]
    if missing:
        raise SystemExit(f"ERROR: no clips for {missing} under {motion_dir}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--motion-dir", required=True)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--size-a", type=int, required=True, help="sources in half A; B gets the rest")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--detail", type=int, default=0,
                   help="also print per-object clip counts for the top N candidates")
    a = p.parse_args(argv)

    clips = index_clips(os.path.expanduser(a.motion_dir), a.sources)
    n = {s: sum(v.values()) for s, v in clips.items()}
    all_objects = set(o for v in clips.values() for o in v)
    total = sum(n.values())
    print(f"{len(a.sources)} sources, {total} clips, {len(all_objects)} objects")
    for s in sorted(a.sources, key=lambda x: -n[x]):
        print(f"  {s:>6} {n[s]:>4}  {' '.join(sorted(clips[s]))}")

    cands = []
    for A in itertools.combinations(a.sources, a.size_a):
        B = [s for s in a.sources if s not in A]
        na = sum(n[s] for s in A)
        cov_a = set(o for s in A for o in clips[s])
        cov_b = set(o for s in B for o in clips[s])
        cands.append((abs(2 * na - total), na, total - na,
                      len(all_objects - cov_a), len(all_objects - cov_b), A, tuple(B)))
    cands.sort()

    def key(s):
        return int(s[3:])

    print(f"\ntop {a.top} of {len(cands)} splits by clip-count gap "
          f"(A has {a.size_a} sources, B has {len(a.sources) - a.size_a}):")
    for gap, na, nb, miss_a, miss_b, A, B in cands[:a.top]:
        cov = "both cover all objects" if not (miss_a or miss_b) else \
              f"A misses {miss_a}, B misses {miss_b} objects"
        print(f"  gap {gap:>3}  A {na:>4} = {' '.join(sorted(A, key=key))}")
        print(f"           B {nb:>4} = {' '.join(sorted(B, key=key))}   [{cov}]")

    if a.detail:
        # Per-object clip counts for the top candidates: an object with 5 clips
        # in a half is nearly as absent as one with 0, and the coverage flag
        # above cannot show that.
        objs = sorted(all_objects)
        for gap, na, nb, _, _, A, B in cands[:a.detail]:
            print(f"\n  gap {gap}: clips per object")
            print(f"    {'object':>13} {'A':>5} {'B':>5}")
            for o in objs:
                ca = sum(clips[s].get(o, 0) for s in A)
                cb = sum(clips[s].get(o, 0) for s in B)
                flag = "  <-- missing" if (ca == 0 or cb == 0) else ""
                print(f"    {o:>13} {ca:>5} {cb:>5}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

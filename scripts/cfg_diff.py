#!/usr/bin/env python3
"""Key-by-key diff of two InterMimic configs, so nobody has to hand-transcribe one.

A textual diff between a generated cfg (sorted keys, block-style lists) and a
hand-written one is unreadable, and reading either by eye is how a wrong row
ends up in a comparison table. This parses both and compares VALUES.

    python3 scripts/cfg_diff.py <a.yaml> <b.yaml>
    python3 scripts/cfg_diff.py <a.yaml> <b.yaml> --all
    python3 scripts/cfg_diff.py <a.yaml> <b.yaml> --keys physicalBufferSize,rewardTerms.pose.enable

Absent is printed explicitly as `<absent>` rather than skipped, because "the key
is missing" and "the key is set to the default" are different claims about a
config and only the second one is safe to state.

The summary line is not decoration: it reports how many keys were compared and
how many were identical. A key you expected to see in the diff and don't is
identical, not unexamined -- run with --all or --keys to see it stated.
"""
import argparse
import sys

import yaml

ABSENT = "<absent>"


def flatten(node, prefix=""):
    """Nested dict -> {'a.b.c': value}. Lists are leaves, compared by value."""
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(path):
    with open(path) as fh:
        return flatten(yaml.safe_load(fh))


def compare(a, b, keys=None):
    """-> (rows, n_compared, n_same) where rows = [(key, va, vb, differs)].

    `keys` restricts to an explicit list; a requested key absent from BOTH files
    is still returned, so asking about a typo'd key shows up rather than
    silently returning nothing.
    """
    names = list(keys) if keys else sorted(set(a) | set(b))
    rows, n_same = [], 0
    for k in names:
        va, vb = a.get(k, ABSENT), b.get(k, ABSENT)
        differs = va != vb
        if not differs:
            n_same += 1
        rows.append((k, va, vb, differs))
    return rows, len(names), n_same


def fmt(v, width):
    """Values can be 43-element body lists; keep one row to one line."""
    s = str(v)
    return s if len(s) <= width else s[:width - 1] + "…"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--label-a", help="name for the left file (default: its basename)")
    p.add_argument("--label-b", help="name for the right file")
    p.add_argument("--all", action="store_true",
                   help="also print keys whose values are identical")
    p.add_argument("--keys",
                   help="comma-separated dotted keys to report explicitly, "
                        "differing or not (e.g. rewardTerms.pose.enable)")
    p.add_argument("--width", type=int, default=60, help="max value width")
    args = p.parse_args(argv)

    import os
    la = args.label_a or os.path.basename(args.a)
    lb = args.label_b or os.path.basename(args.b)

    try:
        a, b = load(args.a), load(args.b)
    except (OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    rows, n_compared, n_same = compare(a, b, keys)

    shown = 0
    for key, va, vb, differs in rows:
        if not differs and not (args.all or keys):
            continue
        mark = " " if differs else "="
        print(f"{mark} {key}\n    {la}: {fmt(va, args.width)}\n    {lb}: {fmt(vb, args.width)}")
        shown += 1

    n_diff = n_compared - n_same
    print(f"\n{n_compared} keys compared: {n_diff} differ, {n_same} identical"
          + ("" if (args.all or keys) else f" (identical not shown; --all to see them)"))
    if shown == 0:
        print("(no rows printed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

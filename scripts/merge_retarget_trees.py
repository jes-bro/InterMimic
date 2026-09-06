#!/usr/bin/env python3
"""Merge per-source retarget trees into one, for a MULTI-SOURCE teacher.

Retargeting is solved per SOURCE subject, so each tree holds one source's clips
laid out by target body:

    InterAct/OMOMO_retarget_contact_src2/<body>/sub2_<obj>_<idx>.pt
    InterAct/OMOMO_retarget_contact_src6/<body>/sub6_<obj>_<idx>.pt

A single-source arm points retargetedMotionDir at one of those. But the task
reads exactly ONE directory (intermimic.py:302-334), so a teacher trained on
dataSub ['sub2','sub6'] needs BOTH sources' clips under each body -- otherwise it
raises FileNotFoundError on every clip of the second source. Deliberately: the
alternative is silently falling back to the un-retargeted reference for half the
data, which would run to completion and look fine.

This builds that combined tree out of SYMLINKS. No re-solving: the per-(body,
clip) files already exist and are identical either way, so copying would only
duplicate gigabytes and create a second thing to keep in sync.

    python3 scripts/merge_retarget_trees.py \\
        --sources InterAct/OMOMO_retarget_contact_src2 \\
                  InterAct/OMOMO_retarget_contact_src6 \\
        --out     InterAct/OMOMO_retarget_contact_src2src6 --dry-run

WHAT IT REFUSES TO DO. A body present in one source tree but missing from
another is an ERROR, not a partial merge. The arm's subjectBodies is a single
list applied to every source, so a body that lacks one source's clips fails at
startup anyway -- better to hear it here, with the list, than from a job that
died in 30 seconds. Pass --bodies-from to check against the arm's own roster
rather than against whatever happens to be on disk.
"""
import argparse
import os
import sys


def body_dirs(root):
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: no retarget tree at {root}")
    return {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}


def clips(root, body):
    d = os.path.join(root, body)
    return sorted(f for f in os.listdir(d) if f.endswith(".pt")) if os.path.isdir(d) else []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sources", nargs="+", required=True,
                   help="per-source retarget trees to combine, in dataSub order")
    p.add_argument("--out", required=True, help="combined tree to build")
    p.add_argument("--bodies-from", help="env yaml whose subjectBodies must ALL be "
                                         "covered; default = bodies common to every source")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    per_source = {s: body_dirs(s) for s in a.sources}
    for s, b in per_source.items():
        print(f"{s}: {len(b)} body dirs")

    if a.bodies_from:
        import yaml
        want = set(yaml.safe_load(open(a.bodies_from))["env"]["subjectBodies"])
        print(f"roster from {os.path.basename(a.bodies_from)}: {len(want)} bodies")
    else:
        want = set.intersection(*per_source.values())
        print(f"roster = bodies common to every source: {len(want)}")

    # A body missing from ANY source cannot be trained on, so report the whole
    # picture at once rather than failing on the first one.
    gaps = {s: sorted(want - b, key=lambda x: int(x[3:])) for s, b in per_source.items()}
    if any(gaps.values()):
        print("\nERROR: these bodies are missing from a source tree:", file=sys.stderr)
        for s, miss in gaps.items():
            if miss:
                print(f"  {s}: {len(miss)} missing -- {' '.join(miss[:8])}"
                      f"{' ...' if len(miss) > 8 else ''}", file=sys.stderr)
        print("\nSolve them before merging (slurm_retarget_gen.sh, CPU-only), or\n"
              "restrict the arm's subjectBodies to what every source covers.",
              file=sys.stderr)
        return 2

    n_links = n_bodies = 0
    collisions = []
    for body in sorted(want, key=lambda x: int(x[3:])):
        dst = os.path.join(a.out, body)
        seen = {}
        for s in a.sources:
            for c in clips(s, body):
                if c in seen:          # same filename from two sources: ambiguous
                    collisions.append((body, c, seen[c], s))
                    continue
                seen[c] = s
                if not a.dry_run:
                    os.makedirs(dst, exist_ok=True)
                    link = os.path.join(dst, c)
                    if not os.path.islink(link) and not os.path.exists(link):
                        os.symlink(os.path.abspath(os.path.join(s, body, c)), link)
                n_links += 1
        n_bodies += 1

    if collisions:
        print(f"\nERROR: {len(collisions)} clip name(s) appear in more than one "
              f"source tree, e.g. {collisions[0]}", file=sys.stderr)
        print("A clip must come from exactly one source or the merge is ambiguous.",
              file=sys.stderr)
        return 2

    print(f"\n{'would link' if a.dry_run else 'linked'} {n_links} clips "
          f"across {n_bodies} bodies -> {a.out}")
    if a.dry_run:
        print("(--dry-run: nothing written)")
    else:
        print(f"\nPoint the arm's retargetedMotionDir at: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

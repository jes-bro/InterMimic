#!/usr/bin/env python3
"""Pin scripts/prune_checkpoints.py -- it deletes checkpoints, so its refusals matter more than its successes.

The failure modes worth catching are all silent ones: pruning a run that is
still training, deleting a run's newest checkpoint, touching a protected family,
or inventing an ordering for filenames it cannot parse. Each would be
discovered only when something needed the file that is gone.

Run:  python3 tests/test_prune_checkpoints.py   (exit 0 = all green)
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts/prune_checkpoints.py")
failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def mk(root, run, names, age_s=86400, size=4096):
    d = root / "checkpoints" / run / "nn"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = d / n
        p.write_bytes(b"\0" * size)
        os.utime(p, (time.time() - age_s, time.time() - age_s))
    return d


def run(root, *extra):
    p = subprocess.run([sys.executable, SCRIPT, "--pattern", "smplx_*", *extra],
                       cwd=root, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def build(root):
    numbered = [f"mimic_{i*500:08d}.pth" for i in range(1, 44)]
    mk(root, "smplx_curriculum_a", numbered + ["mimic.pth"])
    mk(root, "smplx_teacher_g2_mlp_plain_stock__f0", numbered)   # protected family
    mk(root, "smplx_cari4d_bball_r6_cf2", ["mimic.pth"])         # protected family
    mk(root, "smplx_curriculum_mixed", ["mimic_00001000.pth", "alpha.pth", "beta.pth"])
    mk(root, "smplx_curriculum_live", numbered[:9], age_s=60)    # still training
    mk(root, "smplx_curriculum_tiny", ["mimic_00000500.pth", "mimic_00001000.pth"])


def test_dry_run_is_default():
    print("1. it does not delete unless told to:")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        before = sum(1 for _ in (root / "checkpoints").rglob("*.pth"))
        code, out = run(root, "--keep-every", "10")
        after = sum(1 for _ in (root / "checkpoints").rglob("*.pth"))
        check("exits 0", code == 0, f"(exit {code})")
        check("says DRY RUN", "DRY RUN" in out)
        check("deleted NOTHING without --apply", before == after,
              f"({before} -> {after})")
        check("still reports what it would reclaim", "TOTAL:" in out)


def test_refusals():
    print("\n2. the refusals -- each of these would be a silent loss:")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        run(root, "--keep-every", "10", "--apply")
        ck = root / "checkpoints"
        check("PROTECTED: the g2 family is untouched",
              len(list((ck / "smplx_teacher_g2_mlp_plain_stock__f0/nn").glob("*.pth"))) == 43)
        check("PROTECTED: the bball family is untouched",
              len(list((ck / "smplx_cari4d_bball_r6_cf2/nn").glob("*.pth"))) == 1)
        check("LIVE: a run touched minutes ago is skipped",
              len(list((ck / "smplx_curriculum_live/nn").glob("*.pth"))) == 9)
        check("AMBIGUOUS: unparseable names are skipped, not guessed",
              len(list((ck / "smplx_curriculum_mixed/nn").glob("*.pth"))) == 3)
        check("TOO SMALL: a 2-checkpoint run is left alone",
              len(list((ck / "smplx_curriculum_tiny/nn").glob("*.pth"))) == 2)


def test_what_survives():
    print("\n3. what a pruned run keeps:")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        run(root, "--keep-every", "10", "--apply")
        d = root / "checkpoints/smplx_curriculum_a/nn"
        kept = sorted(p.name for p in d.glob("*.pth"))
        check("the ROLLING mimic.pth survives", "mimic.pth" in kept)
        check("the NEWEST checkpoint survives", "mimic_00021500.pth" in kept,
              f"(kept {kept})")
        check("the OLDEST survives (left endpoint)", "mimic_00000500.pth" in kept,
              f"(kept {kept})")
        numbered = [k for k in kept if k != "mimic.pth"]
        check("roughly 1 in 10 kept", 4 <= len(numbered) <= 7,
              f"(kept {len(numbered)} of 43)")
        check("it actually deleted most of them", len(numbered) < 43)


def test_guards():
    print("\n4. argument guards:")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        code, out = run(root, "--keep-every", "1")
        check("--keep-every 1 is refused (would delete nothing)", code != 0)
        code, out = run(root, "--keep-every", "10", "--root", "nope")
        check("a missing checkpoints dir is refused", code != 0)
        p = subprocess.run([sys.executable, SCRIPT, "--pattern", "nomatch*"],
                           cwd=root, capture_output=True, text=True)
        check("a pattern matching nothing is refused", p.returncode != 0)


def test_progress_output():
    """Jess asked to see space reclaimed as it goes, not only at the end."""
    print("\n5. it reports reclaimed space as it goes:")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        _, out = run(root, "--keep-every", "10", "--apply")
        check("prints a running total during deletion", "running total" in out)
        check("names the run as each finishes", "done smplx_curriculum_a" in out)
        check("ends with the reclaimed figure", "reclaimed" in out)


def main():
    test_dry_run_is_default()
    test_refusals()
    test_what_survives()
    test_guards()
    test_progress_output()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())


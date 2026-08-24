#!/usr/bin/env python3
"""Prove the bball-r2_warm slurm guards assert the arm that f7826fb actually
describes -- and that they fire on every way the arm could silently decay.

Background: slurm_cari4d_bball_r2_warm.sh was cloned from a sibling bball arm and
inherited a `human: false` assertion. But r2 is the ONE arm that deliberately
RESTORES the human divergence reset (0.5m) to execute the crawl exploit, keeping
only the object-side resets off. The stale guard therefore refused to launch the
very config it was meant to protect (job 16957484 died on it). This test pins the
corrected intent so the clone-drift cannot come back.

Deliberately extracts and runs the guard block STRAIGHT OUT of the shell script
rather than reimplementing the greps -- a reimplementation would happily agree
with a stale guard, which is the exact failure being defended against.

Covers:
  1. positive: the committed r2 env cfg passes all guards (the launch works).
  2. negative: human reset flipped back to false FAILS (the clone-drift bug).
  3. negative: each object-side reset (object / igRatio / contactSteps) turned
     back on FAILS -- that would rebuild the free-flight wall.
  4. negative: resetThresholds block removed entirely FAILS.

Run:  python tests/test_r2_warm_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r2_warm.sh")
CFG = os.path.join(REPO,
                   "isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_r2_warm_train.yaml")

failures = []


def extract_guard_block():
    """Pull the guard section verbatim out of the slurm script.

    Spans the first '# Guard:' comment through the last line before the
    '[bball-r2_warm] invocation' echo -- i.e. every check the real job runs
    before it would burn a GPU.
    """
    lines = open(SCRIPT).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    end = next(i for i, l in enumerate(lines) if "invocation:" in l)
    block = "\n".join(lines[start:end])
    assert "resetThresholds" in block, "guard block did not survive extraction"
    return block


def run_guards(cfg_path, block):
    """Run the extracted guards against one cfg. Returns (exit_code, stderr)."""
    # The block reads $CFG_ENV; supply it and nothing else, so we exercise the
    # real assertions without touching conda/slurm/python.
    script = 'CFG_ENV="$1"\n' + block + "\nexit 0\n"
    proc = subprocess.run(["bash", "-c", script, "bash", cfg_path],
                          capture_output=True, text=True)
    return proc.returncode, proc.stderr.strip()


def mutated_cfg(pattern, replacement, tmpdir, label):
    """Write a copy of the real cfg with one knob sabotaged."""
    src = open(CFG).read()
    out, n = re.subn(pattern, replacement, src, count=1, flags=re.MULTILINE)
    assert n == 1, f"mutation '{label}' matched {n} times -- test fixture is stale"
    path = os.path.join(tmpdir, f"mutant_{label}.yaml")
    with open(path, "w") as f:
        f.write(out)
    return path


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def main():
    block = extract_guard_block()
    print(f"extracted {len(block.splitlines())} guard lines from {os.path.basename(SCRIPT)}\n")

    print("1. positive control -- the committed r2 cfg must launch:")
    code, err = run_guards(CFG, block)
    check("real cfg passes all guards", code == 0, f"(exit {code}: {err})")

    print("\n2-4. negative controls -- each must be REFUSED:")
    # (label, regex to find in the real cfg, sabotaged replacement)
    mutations = [
        # The original clone-drift bug: human reset back off.
        ("human_reset_off", r"^    human: 0\.5.*$", "    human: false"),
        # Object-side resets back on -> free-flight wall returns.
        ("object_reset_on", r"^    object: false.*$", "    object: 0.3"),
        ("igratio_reset_on", r"^    igRatio: false.*$", "    igRatio: 0.5"),
        ("contactsteps_reset_on", r"^    contactSteps: false.*$", "    contactSteps: 50"),
        # Whole block gone.
        ("resetthresholds_removed", r"^  resetThresholds:.*$", "  # resetThresholds removed"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        for label, pattern, replacement in mutations:
            path = mutated_cfg(pattern, replacement, tmpdir, label)
            code, err = run_guards(path, block)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

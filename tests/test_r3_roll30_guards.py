#!/usr/bin/env python3
"""Pin the bball-r3_roll30 arm: its slurm guards, and the one-knob claim itself.

r3_roll30 exists because r2_warm logged `completed 0 (0.0%)` of 16,666,993
episodes -- the layup never received a single gradient. Cause: the start-frame
sampler is randint(0, max(1, clip_len - rolloutLength)) (intermimic.py:1246), and
rolloutLength 300 against a 101-frame clip clamps that to randint(0,1) = always
frame 0. stateInit "Hybrid" was silently a Start init.

The arm's whole value is that rolloutLength is the ONLY difference from r2_warm.
Two ways that can rot, both covered here:
  A. the cfgs drift apart (a second knob sneaks in)     -> test_one_knob
  B. the guards fail to catch a reverted/cloned cfg     -> test_guards

Guard tests run the block STRAIGHT OUT of the shell script rather than
reimplementing the greps -- a reimplementation would happily agree with a stale
guard, which is the failure mode being defended against (see the r2 clone-drift
that killed job 16957484).

Also pins the arithmetic that justifies the value chosen (30), so a future edit
to rolloutLength has to confront what it does to start coverage.

Run:  python tests/test_r3_roll30_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r3_roll30.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_eval.yaml")
R2 = os.path.join(CFGDIR, "omomo_cari4d_bball_r2_warm_train.yaml")
R2_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r2_warm_eval.yaml")

# The clip this arm trains on: sub100_bball_000.pt, 101 frames @30fps.
CLIP_FRAMES = 101

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def flat(d, p=""):
    """Flatten nested cfg dicts to dotted keys, so a comparison can't be fooled
    by reordering or by comments."""
    if not isinstance(d, dict):
        return {p: d}
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else str(k)
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def extract_guard_block():
    """Pull the guard section verbatim out of the slurm script: the first
    '# Guard:' comment through the line before the invocation echo."""
    lines = open(SCRIPT).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    end = next(i for i, l in enumerate(lines) if "invocation:" in l)
    block = "\n".join(lines[start:end])
    assert "rolloutLength" in block, "guard block did not survive extraction"
    return block


def run_guards(cfg_path, block):
    script = 'CFG_ENV="$1"\n' + block + "\nexit 0\n"
    proc = subprocess.run(["bash", "-c", script, "bash", cfg_path],
                          capture_output=True, text=True)
    return proc.returncode, proc.stderr.strip()


def mutated_cfg(pattern, replacement, tmpdir, label):
    src = open(CFG).read()
    out, n = re.subn(pattern, replacement, src, count=1, flags=re.MULTILINE)
    assert n == 1, f"mutation '{label}' matched {n} times -- fixture is stale"
    path = os.path.join(tmpdir, f"mutant_{label}.yaml")
    with open(path, "w") as f:
        f.write(out)
    return path


def start_range(clip_frames, rollout_length):
    """Reimplements intermimic.py:1246 -- the exclusive upper bound of the
    uniform start-frame draw. 1 means 'frame 0 only'."""
    return max(1, clip_frames - rollout_length)


def test_one_knob():
    """The cfgs must differ from r2_warm in rolloutLength and nothing else."""
    print("1. one-knob claim (parsed yaml, not text):")

    train = flat(yaml.safe_load(open(CFG)) or {})
    r2 = flat(yaml.safe_load(open(R2)) or {})
    diffs = {k: (r2.get(k, "<absent>"), train.get(k, "<absent>"))
             for k in set(r2) | set(train)
             if r2.get(k, "<absent>") != train.get(k, "<absent>")}
    check("train cfg differs from r2_warm ONLY in env.rolloutLength",
          set(diffs) == {"env.rolloutLength"}, f"(differs in: {sorted(diffs)})")
    check("train rolloutLength is 30", train.get("env.rolloutLength") == 30,
          f"(got {train.get('env.rolloutLength')})")

    ev = flat(yaml.safe_load(open(CFG_EVAL)) or {})
    r2e = flat(yaml.safe_load(open(R2_EVAL)) or {})
    ev_diffs = {k for k in set(r2e) | set(ev)
                if r2e.get(k, "<absent>") != ev.get(k, "<absent>")}
    # The eval twin must NOT inherit the short rollout: rolloutLength also ends
    # the episode (humanoid.py:553), so shrinking it would redefine success and
    # break comparability with r2_warm's eval.
    check("eval cfg is unchanged from r2_warm's eval", not ev_diffs,
          f"(differs in: {sorted(ev_diffs)})")
    check("eval keeps rolloutLength 300 (full-clip success)",
          ev.get("env.rolloutLength") == 300, f"(got {ev.get('env.rolloutLength')})")

    # PSI must stay off: rolloutLength 30 makes motions PSI-eligible
    # (intermimic.py:847), so a stray key would add a second variable.
    check("no physicalBufferSize key (PSI stays gated off)",
          "env.physicalBufferSize" not in train)


def test_coverage_arithmetic():
    """Pin why 30, in terms of the sampler that caused the failure."""
    print("\n2. start-frame coverage (intermimic.py:1246 arithmetic):")
    check("r2_warm's 300 collapses to frame 0 only",
          start_range(CLIP_FRAMES, 300) == 1,
          f"(bound {start_range(CLIP_FRAMES, 300)})")
    bound = start_range(CLIP_FRAMES, 30)
    check("30 samples starts across frames 0-71", bound == 71, f"(bound {bound})")
    # The takeoff sits ~frame 35-70; a start must be able to land inside it or
    # the airborne frames get no gradient no matter how long the run trains.
    check("takeoff window (35-70) is reachable by a start draw", bound > 70,
          f"(bound {bound} does not reach frame 70)")
    # Guard the boundary: anything >= 41 stops covering the takeoff.
    check("41 would NOT cover the takeoff (why not a larger value)",
          start_range(CLIP_FRAMES, 41) <= 60, f"(bound {start_range(CLIP_FRAMES, 41)})")


def test_guards():
    print("\n3. slurm guards -- positive control:")
    block = extract_guard_block()
    code, err = run_guards(CFG, block)
    check("committed r3 cfg passes all guards", code == 0, f"(exit {code}: {err})")

    print("\n4. slurm guards -- each sabotage must be REFUSED:")
    mutations = [
        # The clone-drift that matters most: rollout reverts to r2_warm's value,
        # silently making this a duplicate of a run known to return 0.0%.
        ("rollout_reverted", r"^  rolloutLength: 30.*$", "  rolloutLength: 300"),
        # Hybrid -> Start bypasses the sampler entirely (intermimic.py:1247), so
        # a short rollout would only truncate frame-0 episodes.
        ("stateinit_start", r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
        # PSI un-gated by the short rollout = a second variable.
        ("psi_added", r"^  hybridInitProb: 0\.1.*$",
         "  hybridInitProb: 0.1\n  physicalBufferSize: 3"),
        # Inherited r2 guards: the termination regime must be preserved exactly.
        ("human_reset_off", r"^    human: 0\.5.*$", "    human: false"),
        ("object_reset_on", r"^    object: false.*$", "    object: 0.3"),
        ("igratio_reset_on", r"^    igRatio: false.*$", "    igRatio: 0.5"),
        ("contactsteps_reset_on", r"^    contactSteps: false.*$", "    contactSteps: 50"),
        ("resetthresholds_removed", r"^  resetThresholds:.*$", "  # removed"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        for label, pattern, replacement in mutations:
            path = mutated_cfg(pattern, replacement, tmpdir, label)
            code, err = run_guards(path, block)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")


def main():
    test_one_knob()
    test_coverage_arithmetic()
    test_guards()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

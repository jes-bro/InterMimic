#!/usr/bin/env python3
"""Pin the bball-r4_human1m arm: its slurm guards, and the one-knob claim.

r4 exists because r3_roll30 measured, at sim step 40000: 9,677,090 episodes,
93.9% killed by the human divergence reset, mean episode 16.9 of a 30-frame
window, and rb unmoved at 0.117. r3's coverage fix worked (starts spread over
frames 0-71, clip coverage ~38% -> ~87%), so coverage is no longer the binding
constraint. r4 asks whether the reset now is, by doubling the threshold.

r3 is the CONTROL and runs concurrently. The arm is only worth anything if
resetThresholds.human is the sole difference, so that is tested directly against
r3's committed cfgs rather than assumed.

Guard tests run the block STRAIGHT OUT of the shell script rather than
reimplementing the greps -- a reimplementation would happily agree with a stale
guard, which is the failure mode being defended against (the r2 clone-drift that
killed job 16957484, and which this arm's launcher inherits by construction
since it was seded from r3's).

Run:  python tests/test_r4_human1m_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r4_human1m.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r4_human1m_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r4_human1m_eval.yaml")
R3 = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_train.yaml")
R3_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_eval.yaml")

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def flat(d, p=""):
    if not isinstance(d, dict):
        return {p: d}
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else str(k)
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def extract_guard_block():
    lines = open(SCRIPT).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    end = next(i for i, l in enumerate(lines) if "invocation:" in l)
    block = "\n".join(lines[start:end])
    assert "human" in block, "guard block did not survive extraction"
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


def test_one_knob():
    print("1. one-knob claim vs the r3 control (parsed yaml):")
    train, r3 = (flat(yaml.safe_load(open(p)) or {}) for p in (CFG, R3))
    diffs = {k for k in set(r3) | set(train)
             if r3.get(k, "<absent>") != train.get(k, "<absent>")}
    check("train cfg differs from r3 ONLY in env.resetThresholds.human",
          diffs == {"env.resetThresholds.human"}, f"(differs in: {sorted(diffs)})")
    check("relaxed threshold is 1.0",
          train.get("env.resetThresholds.human") == 1.0,
          f"(got {train.get('env.resetThresholds.human')})")

    # Everything r3 established must survive: the coverage fix is the substrate
    # this arm is testing on top of, not something to re-litigate.
    check("rolloutLength still 30 (r3's coverage fix intact)",
          train.get("env.rolloutLength") == 30, f"(got {train.get('env.rolloutLength')})")
    check("stateInit still Hybrid", train.get("env.stateInit") == "Hybrid")
    check("PSI still absent", "env.physicalBufferSize" not in train)
    for knob in ("object", "igRatio", "contactSteps"):
        check(f"object-side reset {knob} still off",
              train.get(f"env.resetThresholds.{knob}") is False,
              f"(got {train.get(f'env.resetThresholds.{knob}')})")

    print("\n2. the eval twin is the shared measuring instrument:")
    ev, r3e = (flat(yaml.safe_load(open(p)) or {}) for p in (CFG_EVAL, R3_EVAL))
    ev_diffs = {k for k in set(r3e) | set(ev)
                if r3e.get(k, "<absent>") != ev.get(k, "<absent>")}
    check("eval cfg is identical to r3's eval", not ev_diffs,
          f"(differs in: {sorted(ev_diffs)})")
    # If eval inherited the relaxed threshold, r3 and r4 would be scored by
    # different instruments and their eval numbers would not compare.
    check("eval keeps human 0.5 (NOT the relaxed value)",
          ev.get("env.resetThresholds.human") == 0.5,
          f"(got {ev.get('env.resetThresholds.human')})")
    check("eval keeps rolloutLength 300 (full-clip success)",
          ev.get("env.rolloutLength") == 300)


def test_guards():
    print("\n3. slurm guards -- positive control:")
    block = extract_guard_block()
    code, err = run_guards(CFG, block)
    check("committed r4 cfg passes all guards", code == 0, f"(exit {code}: {err})")

    print("\n4. slurm guards -- each sabotage must be REFUSED:")
    mutations = [
        # The clone-drift this launcher is most exposed to: it was seded from
        # r3's, so a missed edit leaves it asserting r3's 0.5 and refusing the
        # very cfg it exists to run (exactly what killed r2's job 16957484).
        ("human_reverted_to_r3", r"^    human: 1\.0.*$", "    human: 0.5"),
        ("human_off_entirely", r"^    human: 1\.0.*$", "    human: false"),
        # r3's coverage fix must not be silently undone underneath this arm.
        ("rollout_reverted", r"^  rolloutLength: 30.*$", "  rolloutLength: 300"),
        ("stateinit_start", r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
        ("psi_added", r"^  hybridInitProb: 0\.1.*$",
         "  hybridInitProb: 0.1\n  physicalBufferSize: 3"),
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


def test_separate_experiment():
    """Own checkpoint dir: r4 must never write into r3's."""
    print("\n5. experiment isolation:")
    rlg = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r4_human1m_train.yaml")
    txt = open(rlg).read()
    check("full_experiment_name is smplx_cari4d_bball_r4_human1m",
          "full_experiment_name: smplx_cari4d_bball_r4_human1m" in txt)
    check("does not name r3's experiment", "smplx_cari4d_bball_r3_roll30" not in txt)
    # Warm start must be the shared sub2 teacher, NOT r3's checkpoint -- warm
    # starting off another RUN is the standing prohibition.
    check("warm start is the sub2 teacher, not another run",
          "resume_from: checkpoints/smplx_teachers_new/sub2.pth" in txt)
    sh = open(SCRIPT).read()
    check("launcher points at the r4 env cfg",
          "omomo_cari4d_bball_r4_human1m_train.yaml" in sh)
    check("launcher resumes only from its OWN checkpoint dir",
          'CKPT="checkpoints/${EXP}/nn/mimic.pth"' in sh)


def main():
    test_one_knob()
    test_guards()
    test_separate_experiment()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

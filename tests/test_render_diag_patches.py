#!/usr/bin/env python3
"""Prove the two diagnostic render instruments are what they claim to be.

Both scripts render a COMMITTED cfg (omomo_cari4d_bball_diag_{midclip,playthru}
.yaml) rather than patching one at runtime, so there are two ways this can rot:

  A. the diag cfg drifts and stops disabling what it must  -> test_diag_cfgs
  B. a script gets pointed at the wrong cfg and renders it -> test_verify_guards

Both are covered. The guard blocks are extracted VERBATIM from the shell scripts
(between the '# --- VERIFY BEGIN ---' / '# --- VERIFY END ---' markers) and run
for real -- reimplementing the greps here would happily agree with a stale guard.

Pointing either script at an arm's EVAL cfg must be refused, because that is
precisely the render that produced the uninformative dribble-reset loop:
the human 0.5m reset cuts every episode at frame ~17-38, the episode restarts at
frame 0, and every arm's video looks identical regardless of what it learned.

Also pins the two code facts that justify a separate cfg instead of NO_TERM=1,
so a future fix upstream surfaces as a failing test rather than as silently
redundant scripts.

Run:  python tests/test_render_diag_patches.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
MIDCLIP = os.path.join(REPO, "slurm_cari4d_bball_render_midclip.sh")
PLAYTHRU = os.path.join(REPO, "slurm_cari4d_bball_render_playthrough.sh")
DIAG_MID = os.path.join(CFGDIR, "omomo_cari4d_bball_diag_midclip.yaml")
DIAG_PLAY = os.path.join(CFGDIR, "omomo_cari4d_bball_diag_playthru.yaml")
# The arms' eval twins: the diag cfgs derive from these, and the guards must
# REFUSE these.
EVALS = [os.path.join(CFGDIR, f"omomo_cari4d_bball_{a}_eval.yaml")
         for a in ("r3_roll30", "r4_human1m")]

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def env(path):
    return (yaml.safe_load(open(path)) or {}).get("env", {})


def extract_verify(script):
    lines = open(script).read().splitlines()
    start = next(i for i, l in enumerate(lines) if "VERIFY BEGIN" in l)
    end = next(i for i, l in enumerate(lines) if "VERIFY END" in l)
    block = "\n".join(lines[start + 1:end])
    assert "grep" in block, f"{script}: verify block did not survive extraction"
    return block


def run_verify(block, cfg_env):
    script = 'CFG_ENV="$1"\n' + block + "\nexit 0\n"
    proc = subprocess.run(["bash", "-c", script, "bash", cfg_env],
                          capture_output=True, text=True)
    return proc.returncode, proc.stderr.strip()


def test_diag_cfgs():
    """The committed instruments must actually disable what they claim, and
    differ from the eval twin ONLY in the intended keys."""
    print("1. diag cfgs -- the instruments themselves:")
    src = env(EVALS[0])

    mid = env(DIAG_MID)
    check("midclip: stateInit -> Hybrid", mid.get("stateInit") == "Hybrid",
          f"(got {mid.get('stateInit')})")
    check("midclip: rolloutLength -> 50", mid.get("rolloutLength") == 50,
          f"(got {mid.get('rolloutLength')})")
    check("midclip: human reset disabled",
          mid.get("resetThresholds", {}).get("human") is False,
          f"(got {mid.get('resetThresholds', {}).get('human')})")
    # The point of the instrument: randint(0, 101 - L) must be able to open a
    # window BEFORE the takeoff (~frame 40-50) and still run long enough to
    # reach it. Starting AT frame 70 is mid-flight and teaches nothing, which is
    # why this is a band and not "as late as possible" -- see the cfg header.
    L = mid.get("rolloutLength", 300)
    bound = max(1, 101 - L)
    check("midclip: starts can open before the takeoff (~40-50)", bound > 40,
          f"(bound {bound})")
    check("midclip: the window outlives a start placed just before the takeoff",
          L >= 40, f"(rolloutLength {L} is shorter than the run-up it must cover)")
    changed = {k for k in set(src) | set(mid) if src.get(k) != mid.get(k)}
    check("midclip: only the 4 intended keys differ from the eval twin",
          changed == {"stateInit", "rolloutLength", "resetThresholds",
                      "enableEvaluation"},
          f"(changed: {sorted(changed)})")

    play = env(DIAG_PLAY)
    check("playthru: human reset disabled",
          play.get("resetThresholds", {}).get("human") is False,
          f"(got {play.get('resetThresholds', {}).get('human')})")
    check("playthru: stateInit stays Start", play.get("stateInit") == "Start")
    check("playthru: rolloutLength stays 300", play.get("rolloutLength") == 300)
    changed = {k for k in set(src) | set(play) if src.get(k) != play.get(k)}
    check("playthru: only the 2 intended keys differ from the eval twin",
          changed == {"resetThresholds", "enableEvaluation"},
          f"(changed: {sorted(changed)})")

    # Object-side resets must be off in BOTH or the episode is still cut. They
    # are inherited from the arms, so this is a regression check on the source.
    for tag, cfg in (("midclip", mid), ("playthru", play)):
        for knob in ("object", "igRatio", "contactSteps"):
            check(f"{tag}: resetThresholds.{knob} off",
                  cfg.get("resetThresholds", {}).get(knob) is False)

    # One instrument serves both arms only if the two eval twins agree.
    a, b = (env(p) for p in EVALS)
    check("r3 and r4 eval twins are identical (one diag cfg serves both)", a == b)


def test_verify_guards():
    print("\n2. script guards -- positive control:")
    for tag, script, cfg in (("midclip", MIDCLIP, DIAG_MID),
                             ("playthru", PLAYTHRU, DIAG_PLAY)):
        code, err = run_verify(extract_verify(script), cfg)
        check(f"{tag}: accepts its own diag cfg", code == 0, f"(exit {code}: {err})")

    print("\n3. script guards -- the arms' eval cfgs must be REFUSED:")
    # This is the whole point: rendering the eval cfg reproduces the dribble
    # loop and looks like it worked.
    for tag, script in (("midclip", MIDCLIP), ("playthru", PLAYTHRU)):
        for cfg in EVALS:
            code, _ = run_verify(extract_verify(script), cfg)
            check(f"{tag}: refuses {os.path.basename(cfg)}", code != 0,
                  "(guard accepted the cfg whose reset loop it exists to replace)")

    print("\n4. script guards -- a drifted diag cfg must be REFUSED:")
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            (MIDCLIP, DIAG_MID, "midclip", "human_back_on",
             r"^    human: false.*$", "    human: 0.5"),
            (MIDCLIP, DIAG_MID, "midclip", "init_back_to_start",
             r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
            (MIDCLIP, DIAG_MID, "midclip", "rollout_back_to_300",
             r"^  rolloutLength: 50.*$", "  rolloutLength: 300"),
            # r3's old value must ALSO be refused: it is the instrument silently
            # reverting to a regime no live arm trains on.
            (MIDCLIP, DIAG_MID, "midclip", "rollout_back_to_30",
             r"^  rolloutLength: 50.*$", "  rolloutLength: 30"),
            (MIDCLIP, DIAG_MID, "midclip", "object_reset_on",
             r"^    object: false.*$", "    object: 0.3"),
            (PLAYTHRU, DIAG_PLAY, "playthru", "human_back_on",
             r"^    human: false.*$", "    human: 0.5"),
            (PLAYTHRU, DIAG_PLAY, "playthru", "rollout_truncated",
             r"^  rolloutLength: 300.*$", "  rolloutLength: 30"),
            (PLAYTHRU, DIAG_PLAY, "playthru", "igratio_reset_on",
             r"^    igRatio: false.*$", "    igRatio: 0.5"),
        ]
        for script, cfg, tag, label, pat, repl in cases:
            mutated, n = re.subn(pat, repl, open(cfg).read(), count=1, flags=re.M)
            assert n == 1, f"mutation '{tag}/{label}' matched {n} -- fixture stale"
            path = os.path.join(tmp, f"{tag}_{label}.yaml")
            open(path, "w").write(mutated)
            code, _ = run_verify(extract_verify(script), path)
            check(f"{tag}: {label} is refused", code != 0,
                  "(guard is a no-op for this knob)")


def test_code_assumptions():
    """Pin the two code facts that justify a separate cfg over NO_TERM=1."""
    print("\n5. code assumptions these instruments rest on:")
    hum = open(os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/humanoid.py")).read()
    im = open(os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")).read()

    # `reset` is built BEFORE the enable_early_termination check, so the flag
    # cannot stop an episode ending -- only relabel the returned `terminated`.
    reset_line = re.search(r"^\s*reset = torch\.where.*$", hum, re.M)
    eet_line = re.search(r"^\s*if not enable_early_termination:", hum, re.M)
    check("NO_TERM cannot prevent resets (reset built before the flag check)",
          bool(reset_line) and bool(eet_line) and reset_line.start() < eet_line.start())

    # The kinematic reset (human ∪ object ∪ igRatio) is OR'd in unconditionally.
    check("kinematic reset applied unconditionally in compute_hoi_reset",
          re.search(r"terminated = torch\.where\(torch\.logical_or\(reset_ig, contact_reset\)",
                    im) is not None)

    # terminationHeight is an accepted cfg key that nothing reads.
    check("terminationHeight cfg key is never read (hardcoded in humanoid.py)",
          "self._termination_heights = 0.3" in hum
          and not re.search(r"cfg\[.env.\]\[.terminationHeight.\]", hum + im))


def main():
    test_diag_cfgs()
    test_verify_guards()
    test_code_assumptions()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

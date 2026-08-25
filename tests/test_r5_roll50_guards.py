#!/usr/bin/env python3
"""Pin the bball-r5_roll50 arm: its slurm guards, and the one-knob claim itself.

r5_roll50 is the third rung of a one-variable ladder:

    r2_warm  rolloutLength 300  -> starts clamp to frame 0 (the BUG)
    r3_roll30              30   -> starts 0..70, 1.0s  windows
    r5_roll50              50   -> starts 0..50, 1.67s windows   <- this arm

r3 fixed the outright bug (intermimic.py:1246 clamps randint(0, max(1, clip_len
- rolloutLength)) to randint(0,1) when rolloutLength exceeds the clip). r5 tunes
the value: Jess watched r3's mid-clip render and found that only ~1 of ~20
sampled windows opened BEFORE the takeoff -- the rest began already airborne,
which teaches nothing about leaving the ground.

Three ways this arm can rot, all covered here:
  A. the cfgs drift apart (a second knob sneaks in)      -> test_one_knob
  B. the guards fail to catch a reverted/cloned cfg      -> test_guards
  C. the value silently reverts to 30 or 300             -> both of the above

Guard tests run the block STRAIGHT OUT of the shell script rather than
reimplementing the greps -- a reimplementation would happily agree with a stale
guard, which is the failure mode being defended against (see the r2 clone-drift
that killed job 16957484).

The coverage arithmetic that justifies 50 over 30 and 60 is pinned too, so a
future edit has to confront what it does to start coverage. NOTE that it rests
on an UNMEASURED takeoff frame -- see test_coverage_arithmetic.

Run:  python tests/test_r5_roll50_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r5_roll50.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r5_roll50_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r5_roll50_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r5_roll50_train.yaml")
# The immediate predecessor: r5's whole claim is "one knob off THIS".
R3 = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_train.yaml")
R3_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r3_roll30_eval.yaml")
R3_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r3_roll30_train.yaml")

# The clip this arm trains on: sub100_bball_000.pt, 101 frames @30fps.
CLIP_FRAMES = 101
# r3's measured mean episode length, in control steps. This is how far a window
# actually survives, so it is the reach a start has toward the takeoff.
R3_MEAN_EPISODE = 17

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


def load(path):
    return flat(yaml.safe_load(open(path)) or {})


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


def start_bound(clip_frames, rollout_length):
    """Reimplements intermimic.py:1246 -- the exclusive upper bound of the
    uniform start-frame draw. 1 means 'frame 0 only'."""
    return max(1, clip_frames - rollout_length)


def useful_fraction(rollout_length, takeoff, reach=R3_MEAN_EPISODE):
    """Fraction of start draws that are USEFUL for learning the takeoff: the
    start must land in [takeoff - reach, takeoff] (near enough that the episode
    survives to reach it) AND be inside the sampler's range [0, bound)."""
    bound = start_bound(CLIP_FRAMES, rollout_length)
    lo, hi = max(0, takeoff - reach), min(takeoff, bound - 1)
    return max(0, hi - lo + 1) / bound


def test_one_knob():
    """The cfgs must differ from r3_roll30 in rolloutLength and nothing else."""
    print("1. one-knob claim vs r3_roll30 (parsed yaml, not text):")

    train, r3 = load(CFG), load(R3)
    diffs = {k for k in set(r3) | set(train)
             if r3.get(k, "<absent>") != train.get(k, "<absent>")}
    check("train cfg differs from r3_roll30 ONLY in env.rolloutLength",
          diffs == {"env.rolloutLength"}, f"(differs in: {sorted(diffs)})")
    check("train rolloutLength is 50", train.get("env.rolloutLength") == 50,
          f"(got {train.get('env.rolloutLength')})")

    # The eval twin is the SHARED measuring instrument for the whole ladder --
    # rolloutLength also ends the episode (humanoid.py:553), so shrinking it here
    # would redefine success and make the three arms incomparable.
    ev_diffs = {k for k in set(load(R3_EVAL)) | set(load(CFG_EVAL))
                if load(R3_EVAL).get(k, "<absent>") != load(CFG_EVAL).get(k, "<absent>")}
    check("eval cfg is byte-equivalent to r3's eval", not ev_diffs,
          f"(differs in: {sorted(ev_diffs)})")
    check("eval keeps rolloutLength 300 (full-clip success)",
          load(CFG_EVAL).get("env.rolloutLength") == 300,
          f"(got {load(CFG_EVAL).get('env.rolloutLength')})")

    # Own checkpoint dir, or the arm silently overwrites r3's run.
    rlg, r3_rlg = load(RLG), load(R3_RLG)
    name = rlg.get("params.config.full_experiment_name")
    check("rlg full_experiment_name is smplx_cari4d_bball_r5_roll50",
          name == "smplx_cari4d_bball_r5_roll50", f"(got {name})")
    rlg_diffs = {k for k in set(rlg) | set(r3_rlg) if rlg.get(k) != r3_rlg.get(k)}
    check("rlg cfg differs from r3's ONLY in full_experiment_name",
          rlg_diffs == {"params.config.full_experiment_name"},
          f"(differs in: {sorted(rlg_diffs)})")
    # Same teacher init as r2_warm/r3, or the ladder's rungs aren't comparable.
    check("keeps the sub2 teacher warm start",
          rlg.get("params.config.resume_from")
          == "checkpoints/smplx_teachers_new/sub2.pth",
          f"(got {rlg.get('params.config.resume_from')})")

    # PSI must stay off: rolloutLength 50 makes motions PSI-eligible
    # (intermimic.py:847), so a stray key would add a second variable.
    check("no physicalBufferSize key (PSI stays gated off)",
          "env.physicalBufferSize" not in train)


def test_coverage_arithmetic():
    """Pin why 50, in terms of the sampler and r3's measured episode length.

    CAVEAT, deliberately encoded: the takeoff frame T is NOT measured. These
    checks therefore assert the SHAPE of the argument (50 beats 30, and 50 is
    stable where 60 is not) across the plausible range of T, rather than
    asserting a number that rests on a guess. Measure T with
    scripts/inspect_bball_clip.py --every 1 (lowest body point leaving the
    grounded ~0.23m offset) before trusting this arm's result.
    """
    print("\n2. start-frame coverage (intermimic.py:1246 arithmetic):")
    check("r2_warm's 300 collapses to frame 0 only",
          start_bound(CLIP_FRAMES, 300) == 1,
          f"(bound {start_bound(CLIP_FRAMES, 300)})")
    check("r3's 30 samples starts across frames 0-70",
          start_bound(CLIP_FRAMES, 30) == 71,
          f"(bound {start_bound(CLIP_FRAMES, 30)})")
    bound = start_bound(CLIP_FRAMES, 50)
    check("r5's 50 samples starts across frames 0-50", bound == 51,
          f"(bound {bound})")

    # The whole point of the arm: more of the draws are useful, for ANY takeoff
    # in the plausible band.
    for t in (40, 45, 50):
        f50, f30 = useful_fraction(50, t), useful_fraction(30, t)
        check(f"takeoff {t}: 50 beats 30 ({f50:.0%} vs {f30:.0%})", f50 > f30)

    # And 50 is the STABLE choice where 60 is a coin flip on an unmeasured frame.
    spread50 = max(useful_fraction(50, t) for t in (40, 45, 50)) \
        - min(useful_fraction(50, t) for t in (40, 45, 50))
    spread60 = max(useful_fraction(60, t) for t in (40, 45, 50)) \
        - min(useful_fraction(60, t) for t in (40, 45, 50))
    check(f"50 is insensitive to the unmeasured takeoff, 60 is not "
          f"(spread {spread50:.0%} vs {spread60:.0%})", spread50 < spread60)

    # Episode length is the other half of the knob: 1.67s at 30Hz, so the
    # crouch->extend->flight sequence fits inside one window.
    check("50 frames = 1.67s at 30Hz (a full jump fits in one episode)",
          abs(50 / 30.0 - 1.67) < 0.01)


def test_guards():
    print("\n3. slurm guards -- positive control:")
    block = extract_guard_block()
    code, err = run_guards(CFG, block)
    check("committed r5 cfg passes all guards", code == 0, f"(exit {code}: {err})")

    print("\n4. slurm guards -- each sabotage must be REFUSED:")
    mutations = [
        # The two clone-drift reversions that matter: back to r3's value (this
        # arm becomes a duplicate run) or back to r2_warm's (the original bug).
        ("rollout_reverted_to_r3", r"^  rolloutLength: 50.*$", "  rolloutLength: 30"),
        ("rollout_reverted_to_r2", r"^  rolloutLength: 50.*$", "  rolloutLength: 300"),
        # Hybrid -> Start bypasses the sampler entirely (intermimic.py:1247), so
        # a bounded rollout would only truncate frame-0 episodes.
        ("stateinit_start", r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
        # PSI un-gated by the bounded rollout = a second variable.
        ("psi_added", r"^  hybridInitProb: 0\.1.*$",
         "  hybridInitProb: 0.1\n  physicalBufferSize: 3"),
        # Inherited r2/r3 guards: the termination regime must be preserved exactly.
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

    print("\n5. slurm guards -- the SIBLING arms' cfgs must be refused:")
    # Pointing this launcher at r3's or r4's cfg would write their regime into
    # r5's checkpoint dir under r5's name.
    for sibling in ("r3_roll30_train", "r4_human1m_train"):
        path = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sibling}.yaml")
        if not os.path.exists(path):
            check(f"refuses {sibling}", False, "(cfg missing -- fixture stale)")
            continue
        code, _ = run_guards(path, block)
        check(f"refuses {sibling}", code != 0,
              "(guard accepted another arm's cfg)")


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

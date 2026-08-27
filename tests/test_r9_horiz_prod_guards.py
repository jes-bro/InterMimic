#!/usr/bin/env python3
"""Pin bball-r9_horiz_prod: uniform obs horizons on the ORIGINAL product reward.

r9 is the fourth cell of a 2x2 over (reward shape) x (obs horizons):

                 obsHorizons [1,16]      obsHorizons [1,4,7,10,13,16]
    product            r6_cf2                   r9_horiz_prod
    geometric          r7_geom                  r8_horiz

Its value is entirely in being that cell, so the two things worth pinning are
that it really is r6-plus-horizons (not r8 with a rename), and that the guard
refuses a cfg carrying rewardShape -- which would silently duplicate r8.

r8-vs-r9 is the cleanest comparison in the set: both start fresh, so the reward
shape is the only difference. That property is checked here too, because it is
easy to lose by "helpfully" restoring a warm start to one of them.

Run:  python3 tests/test_r9_horiz_prod_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r9_horiz_prod.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r9_horiz_prod_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r9_horiz_prod_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r9_horiz_prod_train.yaml")
R6 = os.path.join(CFGDIR, "omomo_cari4d_bball_r6_cf2_train.yaml")
R8 = os.path.join(CFGDIR, "omomo_cari4d_bball_r8_horiz_train.yaml")
R8_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r8_horiz_train.yaml")
PER_HORIZON = 1599

failures = []


def check(label, cond, detail=""):
    if cond:
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


def load(path):
    return flat(yaml.safe_load(open(path)) or {})


def test_the_2x2():
    print("1. r9 is exactly the missing cell of the 2x2:")
    r9, r6, r8 = load(CFG), load(R6), load(R8)

    d6 = {k for k in set(r6) | set(r9) if r6.get(k, "<absent>") != r9.get(k, "<absent>")}
    check("vs r6: ONLY obsHorizons and numObs differ",
          d6 == {"env.obsHorizons", "env.numObs"}, f"(differs in: {sorted(d6)})")
    check("vs r6: keeps the ORIGINAL product reward (no rewardShape key)",
          "env.rewardShape" not in r9)

    d8 = {k for k in set(r8) | set(r9) if r8.get(k, "<absent>") != r9.get(k, "<absent>")}
    check("vs r8: ONLY rewardShape differs", d8 == {"env.rewardShape"},
          f"(differs in: {sorted(d8)})")
    check("vs r8: identical horizons", r9.get("env.obsHorizons") == r8.get("env.obsHorizons"))
    check("vs r8: identical numObs", r9.get("env.numObs") == r8.get("env.numObs"))


def test_horizons():
    print("\n2. the horizons are uniform between r6's endpoints:")
    env = yaml.safe_load(open(CFG))["env"]
    h = env["obsHorizons"]
    strides = [h[i + 1] - h[i] for i in range(len(h) - 1)]
    check("uniform spacing", len(set(strides)) == 1, f"(strides {strides})")
    check("keeps delta_t 1 and 16 as endpoints", h[0] == 1 and h[-1] == 16, f"(got {h})")
    check("numObs matches the horizon count",
          env["numObs"] == PER_HORIZON * len(h),
          f"({env['numObs']} vs {PER_HORIZON} x {len(h)})")
    ev = yaml.safe_load(open(CFG_EVAL))["env"]
    check("eval carries the same horizons (obs SHAPE must match)",
          ev["obsHorizons"] == h)
    check("eval carries no rewardShape either", "rewardShape" not in ev)


def test_fresh_start():
    """r8-vs-r9 is only clean while BOTH start fresh."""
    print("\n3. r8 and r9 both start fresh, so r8-vs-r9 isolates the reward shape:")
    for name, path in (("r9", RLG), ("r8", R8_RLG)):
        rlg = load(path)
        check(f"{name} has no warm start",
              str(rlg.get("params.config.resume_from")) == "None",
              f"(got {rlg.get('params.config.resume_from')!r})")
    check("experiment names are distinct (no checkpoint collision)",
          load(RLG).get("params.config.full_experiment_name")
          != load(R8_RLG).get("params.config.full_experiment_name"))
    check("r9's experiment name is its own",
          load(RLG).get("params.config.full_experiment_name")
          == "smplx_cari4d_bball_r9_horiz_prod")


def extract_guard_block():
    lines = open(SCRIPT).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    end = next(i for i, l in enumerate(lines) if "invocation:" in l)
    return "\n".join(lines[start:end])


def run_guards(cfg_path, block, cwd):
    script = 'CFG_ENV="$1"\n' + block + "\nexit 0\n"
    p = subprocess.run(["bash", "-c", script, "bash", cfg_path],
                       capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stderr.strip()


def test_guards():
    print("\n4. slurm guards:")
    block = extract_guard_block()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "InterAct/behave_cari4d_optj3d_cf2"))
        code, err = run_guards(CFG, block, tmp)
        check("committed r9 cfg passes", code == 0, f"(exit {code}: {err})")

        print("\n5. each sabotage must be REFUSED:")
        src = open(CFG).read()
        mutations = [
            # THE trap for this arm: adding rewardShape makes it r8 exactly.
            ("rewardshape_added", r"^  obsHorizons: \[.*$",
             "  rewardShape: geometric\n  obsHorizons: [1, 4, 7, 10, 13, 16]"),
            ("horizons_removed", r"^  obsHorizons: \[.*$", "  # removed"),
            ("numobs_stale_3198", r"^  numObs: 9594.*$", "  numObs: 3198"),
            ("motion_reverted_to_cf",
             r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d_optj3d_cf"),
            ("rollout_drifted_to_30", r"^  rolloutLength: 50.*$", "  rolloutLength: 30"),
        ]
        for label, pat, rep in mutations:
            out, n = re.subn(pat, rep, src, count=1, flags=re.MULTILINE)
            assert n == 1, f"mutation {label!r} matched {n} -- fixture stale"
            path = os.path.join(tmp, f"m_{label}.yaml")
            open(path, "w").write(out)
            code, _ = run_guards(path, block, tmp)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

        print("\n6. the other three cells of the 2x2 must be refused:")
        for sib in ("r8_horiz_train", "r7_geom_train", "r6_cf2_train"):
            p = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sib}.yaml")
            if os.path.exists(p):
                code, _ = run_guards(p, block, tmp)
                check(f"refuses {sib}", code != 0)


def test_documented():
    print("\n7. the design and its limits are written down:")
    sh = open(SCRIPT).read()
    check("launcher lays out the 2x2", "2x2" in sh)
    check("it names r8-vs-r9 as the clean comparison", "r8 vs r9" in sh)
    check("it records the forced warm-start confound vs r6",
          "confound" in sh.lower() and "3198" in sh and "9594" in sh)
    check("a kill criterion is written before launch", "KILL CRITERION" in sh)


def main():
    test_the_2x2()
    test_horizons()
    test_fresh_start()
    test_guards()
    test_documented()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())


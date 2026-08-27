#!/usr/bin/env python3
"""Pin bball-r10_vel: the body-velocity reward terms, and why they are nonzero.

rewardWeights pv/rv are 0. in all 384 configs in this repo, which makes
rpv = exp(-epv * 0) = 1.0 EXACTLY -- body linear and angular velocity are not
graded at all. Harmless on slow mocap; not on a ballistic clip, where takeoff
velocity is what determines the jump.

What this file pins:
  1. the inertness claim is arithmetic, so it is checked as arithmetic
  2. r10 is r7 plus those two weights and nothing else
  3. the guard refuses an inherited r7 cfg, which would be a silent duplicate
  4. the generalisation caveat stays in the launcher -- a win here does not
     license pv 3.0 as an InterMimic default without an OMOMO check

Run:  python3 tests/test_r10_vel_guards.py   (exit 0 = all green)
"""
import math
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r10_vel.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r10_vel_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r10_vel_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r10_vel_train.yaml")
R7 = os.path.join(CFGDIR, "omomo_cari4d_bball_r7_geom_train.yaml")
R7_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r7_geom_train.yaml")

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


def test_the_claim():
    print("1. the premise -- pv 0 really does disable the term:")
    # rpv = exp(-epv * w). The claim is that w=0 makes it identically 1 for ANY
    # error, so the term cannot influence the reward at all.
    for epv in (0.0, 0.01, 0.5, 12.0, 1e6):
        if math.exp(-epv * 0.0) != 1.0:
            check("rpv is identically 1.0 when pv=0", False, f"(failed at epv={epv})")
            break
    else:
        check("rpv is identically 1.0 when pv=0, for any error", True)
    # ...and that the chosen weight actually grades.
    check("pv=3.0 grades a 20% error at the clip's 1.98 m/s (~0.62)",
          abs(math.exp(-(0.4 ** 2) * 3.0) - 0.62) < 0.02,
          f"(got {math.exp(-(0.4 ** 2) * 3.0):.3f})")
    check("pv=3.0 still rewards a small error generously",
          math.exp(-(0.1 ** 2) * 3.0) > 0.95)


def test_one_knob():
    print("\n2. r10 is r7 plus the two body-velocity weights, nothing else:")
    r10, r7 = load(CFG), load(R7)
    diffs = {k for k in set(r7) | set(r10) if r7.get(k, "<absent>") != r10.get(k, "<absent>")}
    check("differs from r7 ONLY in pv and rv",
          diffs == {"env.rewardWeights.pv", "env.rewardWeights.rv"},
          f"(differs in: {sorted(diffs)})")
    check("pv is nonzero", float(r10["env.rewardWeights.pv"]) > 0)
    check("rv is nonzero", float(r10["env.rewardWeights.rv"]) > 0)
    check("rv/pv matches the position pair's ratio w.r/w.p",
          abs(float(r10["env.rewardWeights.rv"]) / float(r10["env.rewardWeights.pv"])
              - float(r10["env.rewardWeights.r"]) / float(r10["env.rewardWeights.p"])) < 1e-6,
          "(the velocity pair should inherit the position pair's balance)")
    check("object weights untouched (that would be a second experiment)",
          r10.get("env.rewardWeights.opv") == r7.get("env.rewardWeights.opv")
          and r10.get("env.rewardWeights.orv") == r7.get("env.rewardWeights.orv"))
    check("keeps r7's geometric reward (why this is off r7, not r6)",
          r10.get("env.rewardShape") == "geometric")

    ev = load(CFG_EVAL)
    check("eval carries the same weights", ev.get("env.rewardWeights.pv")
          == r10.get("env.rewardWeights.pv"))

    rlg, r7rlg = load(RLG), load(R7_RLG)
    rlg_diffs = {k for k in set(rlg) | set(r7rlg) if rlg.get(k) != r7rlg.get(k)}
    check("rlg differs ONLY in the experiment name",
          rlg_diffs == {"params.config.full_experiment_name"}, f"({sorted(rlg_diffs)})")
    check("warm start is kept (unlike r8/r9, obs width is unchanged)",
          "sub2.pth" in str(rlg.get("params.config.resume_from")))


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
    print("\n3. slurm guards:")
    block = extract_guard_block()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "InterAct/behave_cari4d_optj3d_cf2"))
        code, err = run_guards(CFG, block, tmp)
        check("committed r10 cfg passes", code == 0, f"(exit {code}: {err})")

        print("\n4. each sabotage must be REFUSED:")
        src = open(CFG).read()
        mutations = [
            ("pv_back_to_zero", r"^    pv: 3\.0.*$", "    pv: 0."),
            ("rv_back_to_zero", r"^    rv: 0\.25.*$", "    rv: 0."),
            ("rewardshape_removed", r"^  rewardShape: geometric.*$", "  # removed"),
            ("motion_reverted_to_cf",
             r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d_optj3d_cf"),
        ]
        for label, pat, rep in mutations:
            out, n = re.subn(pat, rep, src, count=1, flags=re.MULTILINE)
            assert n == 1, f"mutation {label!r} matched {n} -- fixture stale"
            path = os.path.join(tmp, f"m_{label}.yaml")
            open(path, "w").write(out)
            code, _ = run_guards(path, block, tmp)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

        print("\n5. the sibling arms must be refused (all have pv 0.):")
        for sib in ("r7_geom_train", "r8_horiz_train", "r9_horiz_prod_train", "r6_cf2_train"):
            p = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sib}.yaml")
            if os.path.exists(p):
                code, _ = run_guards(p, block, tmp)
                check(f"refuses {sib}", code != 0)


def test_documented():
    print("\n6. the reasoning and its limits are written down:")
    sh = open(SCRIPT).read()
    check("says pv=0 makes the term identically 1.0", "exp(0) = 1.0" in sh)
    check("records that this is InterMimic's default, not a bball choice",
          "384" in sh)
    check("explains why it is off r7 and not r6", "geometric" in sh and "product" in sh)
    check("cites the measured reference-velocity noise", "0.059" in sh and "0.038" in sh)
    check("states the generalisation caveat", "GENERALISATION CAVEAT" in sh)
    check("names the check that would license it as a default", "OMOMO" in sh)
    check("carries no stopping rule", "KILL CRITERION" not in sh)


def main():
    test_the_claim()
    test_one_knob()
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


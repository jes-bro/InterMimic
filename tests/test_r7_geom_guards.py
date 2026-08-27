#!/usr/bin/env python3
"""Pin the bball-r7_geom arm: the reward-shape maths, its cfgs, and its guards.

r7 is r6_cf2 with env.rewardShape: geometric -- the 4th root of the same
rb*ro*rig*rcg product. The claim that justifies it is arithmetic, so it is
checked as arithmetic rather than trusted:

  1. the transform is MONOTONE (same optimum, same ordering of rollouts)
  2. the AND property survives (any factor at zero still zeros the reward)
  3. the gradient w.r.t. a weak factor is much larger, and no longer collapses
     when the OTHER factors are weak -- which is the whole point

Plus the two ways the arm rots: the cfgs drift apart (a second knob), or the
guard fails to catch an inherited r6 cfg (silently duplicating a run in flight).

Run:  python3 tests/test_r7_geom_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r7_geom.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r7_geom_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r7_geom_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r7_geom_train.yaml")
R6 = os.path.join(CFGDIR, "omomo_cari4d_bball_r6_cf2_train.yaml")
R6_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r6_cf2_eval.yaml")
R6_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r6_cf2_train.yaml")
TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")

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


def test_maths():
    """The justification is arithmetic; check the arithmetic, not the prose."""
    print("1. the geometric mean does what the arm claims:")

    def prod(t):
        r = 1.0
        for x in t:
            r *= x
        return r

    def geo(t):
        return prod(t) ** 0.25

    # r6's measured held-frame factors -- the numbers in the launcher header.
    held = (0.304, 0.402, 0.422, 0.240)
    check("product matches the header's 0.012", abs(prod(held) - 0.0124) < 5e-4,
          f"(got {prod(held):.4f})")
    check("geometric matches the header's 0.334", abs(geo(held) - 0.334) < 5e-3,
          f"(got {geo(held):.4f})")

    # MONOTONE: a better rollout must still score better, or the change moves
    # the optimum rather than only the scale.
    better = (0.35, 0.45, 0.45, 0.30)
    check("monotone: better factors score better under both",
          (prod(better) > prod(held)) and (geo(better) > geo(held)))

    # AND property: this is why a SUM was rejected.
    dropped = (0.304, 0.402, 0.422, 0.0)
    check("AND survives: a zero factor still zeros the reward", geo(dropped) == 0.0)
    check("a sum would NOT zero it (why additive was rejected)",
          sum(dropped) > 1.0, "(sanity check on the counterfactual)")

    # The actual point: gradient w.r.t. the weakest factor.
    d_prod = held[0] * held[1] * held[2]                 # dR/d(rcg) for a product
    d_geo = geo(held) / (4 * held[3])                    # dR/d(rcg) for the root
    check("gradient on the weak factor is ~7x larger", d_geo / d_prod > 5,
          f"(product {d_prod:.4f} vs geometric {d_geo:.4f}, ratio {d_geo/d_prod:.1f})")

    # ...and that it does NOT collapse when the others get worse, which is the
    # failure mode the product has.
    worse_others = (0.10, 0.10, 0.10, 0.240)
    d_prod_w = worse_others[0] * worse_others[1] * worse_others[2]
    d_geo_w = geo(worse_others) / (4 * worse_others[3])
    check("product's gradient collapses as other terms worsen",
          d_prod_w < d_prod / 10, f"({d_prod_w:.5f} vs {d_prod:.5f})")
    check("geometric's gradient does not collapse",
          d_geo_w > d_geo / 3, f"({d_geo_w:.4f} vs {d_geo:.4f})")


def test_task_code():
    print("\n2. the task code implements it, and refuses a typo:")
    src = open(TASK).read()
    check("rewardShape is read from the env cfg", "cfg['env'].get('rewardShape'" in src)
    check("default stays 'product' (other arms unaffected)",
          "get('rewardShape', 'product')" in src)
    check("an unknown value raises rather than silently defaulting",
          re.search(r"rewardShape=.*expected", src) is not None)
    check("the 4th root is applied", "** 0.25" in src)
    check("factors are clamped away from zero before the root",
          "clamp_min(1e-8)" in src)
    check("rewardShape is whitelisted in the env-key validator",
          "'rewardShape'," in src)


def test_one_knob():
    print("\n3. one-knob claim vs r6_cf2 (parsed yaml):")
    train, r6 = load(CFG), load(R6)
    diffs = {k for k in set(r6) | set(train)
             if r6.get(k, "<absent>") != train.get(k, "<absent>")}
    check("train cfg differs from r6 ONLY in env.rewardShape",
          diffs == {"env.rewardShape"}, f"(differs in: {sorted(diffs)})")
    check("train rewardShape is geometric",
          train.get("env.rewardShape") == "geometric")
    check("train still reads the relabelled _cf2 data",
          train.get("env.motion_file") == "InterAct/behave_cari4d_optj3d_cf2")
    check("train rolloutLength stays 50", train.get("env.rolloutLength") == 50)

    # The eval twin deliberately does NOT move: eval stays on the original
    # product so r7's eval numbers remain comparable with r2..r6.
    ev, r6e = load(CFG_EVAL), load(R6_EVAL)
    ev_diffs = {k for k in set(r6e) | set(ev)
                if r6e.get(k, "<absent>") != ev.get(k, "<absent>")}
    check("eval cfg is UNCHANGED from r6's (comparability)", ev_diffs == set(),
          f"(differs in: {sorted(ev_diffs)})")
    check("eval does NOT set rewardShape", "env.rewardShape" not in ev)

    rlg, r6rlg = load(RLG), load(R6_RLG)
    check("rlg full_experiment_name is smplx_cari4d_bball_r7_geom",
          rlg.get("params.config.full_experiment_name") == "smplx_cari4d_bball_r7_geom")
    rlg_diffs = {k for k in set(rlg) | set(r6rlg) if rlg.get(k) != r6rlg.get(k)}
    check("rlg differs from r6's ONLY in full_experiment_name",
          rlg_diffs == {"params.config.full_experiment_name"},
          f"(differs in: {sorted(rlg_diffs)})")


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
        check("committed r7 cfg passes", code == 0, f"(exit {code}: {err})")

        print("\n5. each sabotage must be REFUSED:")
        mutations = [
            # THE trap: an inherited r6 cfg has no rewardShape at all, and would
            # silently duplicate a run already in flight under a new name.
            ("rewardshape_removed", r"^  rewardShape: geometric.*$", "  # removed"),
            ("rewardshape_product", r"^  rewardShape: geometric.*$", "  rewardShape: product"),
            ("motion_reverted_to_cf", r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d_optj3d_cf"),
            ("rollout_drifted_to_30", r"^  rolloutLength: 50.*$", "  rolloutLength: 30"),
            ("stateinit_start", r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
            ("human_reset_off", r"^    human: 0\.5.*$", "    human: false"),
        ]
        src = open(CFG).read()
        for label, pat, rep in mutations:
            out, n = re.subn(pat, rep, src, count=1, flags=re.MULTILINE)
            assert n == 1, f"mutation '{label}' matched {n} -- fixture stale"
            path = os.path.join(tmp, f"m_{label}.yaml")
            open(path, "w").write(out)
            code, _ = run_guards(path, block, tmp)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

        print("\n6. sibling arms' cfgs must be refused:")
        for sib in ("r6_cf2_train", "r5_roll50_train", "r3_roll30_train"):
            p = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sib}.yaml")
            if os.path.exists(p):
                code, _ = run_guards(p, block, tmp)
                check(f"refuses {sib}", code != 0)



def main():
    test_maths()
    test_task_code()
    test_one_knob()
    test_guards()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

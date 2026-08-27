#!/usr/bin/env python3
"""Pin the bball-r8_horiz arm: uniform obs horizons, and the confound it carries.

r8 is r7_geom with env.obsHorizons [1,16] -> [1,4,7,10,13,16]. The hypothesis is
that the failure is a TRANSITION, not a skill: the MLP policy sees the reference
at delta_t 1 and 16 and nothing between, a 15-frame blind gap that at 30fps
spans the layup's whole countermovement.

Two things this file exists to stop:
  1. the horizon set drifting into something tuned for THIS clip -- it has to be
     uniform between r7's endpoints, or it does not generalise to a dataset
  2. the numObs/horizon-count mismatch, which is a silent shape bug
And one it exists to RECORD: r8 forfeits the sub2 teacher warm start, because a
3198-wide checkpoint cannot load into 9594. That is a forced second difference,
and the launcher has to say so or a null result gets misattributed.

Run:  python3 tests/test_r8_horiz_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r8_horiz.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r8_horiz_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r8_horiz_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r8_horiz_train.yaml")
R7 = os.path.join(CFGDIR, "omomo_cari4d_bball_r7_geom_train.yaml")
R7_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r7_geom_train.yaml")
TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
PER_HORIZON = 1599          # bball carries no betas: 3198 / 2 horizons

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


def test_horizons():
    print("1. the horizon set is uniform and not tuned to this clip:")
    env = yaml.safe_load(open(CFG))["env"]
    h = env["obsHorizons"]
    strides = [h[i + 1] - h[i] for i in range(len(h) - 1)]
    check("spacing is UNIFORM (generalises across motions)",
          len(set(strides)) == 1, f"(strides {strides})")
    check("keeps r7's near endpoint (delta_t 1)", h[0] == 1, f"(got {h[0]})")
    check("keeps r7's far endpoint (delta_t 16)", h[-1] == 16, f"(got {h[-1]})")
    check("no gap wider than the old 15", max(strides) < 15, f"(max {max(strides)})")
    check("horizons are distinct and ascending", h == sorted(set(h)))
    check("numObs matches the horizon count",
          env["numObs"] == PER_HORIZON * len(h),
          f"({env['numObs']} vs {PER_HORIZON} x {len(h)})")
    ev = yaml.safe_load(open(CFG_EVAL))["env"]
    check("eval uses the SAME horizons (a shape requirement, not a choice)",
          ev["obsHorizons"] == h)
    check("eval numObs matches too", ev["numObs"] == env["numObs"])


def test_one_knob():
    print("\n2. differences from r7_geom are the intended one, plus the forced one:")
    train, r7 = load(CFG), load(R7)
    diffs = {k for k in set(r7) | set(train)
             if r7.get(k, "<absent>") != train.get(k, "<absent>")}
    check("env cfg differs ONLY in obsHorizons and numObs",
          diffs == {"env.obsHorizons", "env.numObs"}, f"(differs in: {sorted(diffs)})")
    check("rewardShape stays geometric (r7's knob is retained)",
          train.get("env.rewardShape") == "geometric")
    check("still the relabelled _cf2 data",
          train.get("env.motion_file") == "InterAct/behave_cari4d_optj3d_cf2")

    rlg, r7rlg = load(RLG), load(R7_RLG)
    rlg_diffs = {k for k in set(rlg) | set(r7rlg) if rlg.get(k) != r7rlg.get(k)}
    check("rlg differs ONLY in name and resume_from",
          rlg_diffs == {"params.config.full_experiment_name",
                        "params.config.resume_from"},
          f"(differs in: {sorted(rlg_diffs)})")
    check("warm start is dropped (3198-wide teacher cannot load into 9594)",
          str(rlg.get("params.config.resume_from")) == "None",
          f"(got {rlg.get('params.config.resume_from')!r})")


def test_task_code():
    print("\n3. the task code honours obsHorizons and refuses nonsense:")
    src = open(TASK).read()
    check("obsHorizons is read from the env cfg", "cfg['env'].get('obsHorizons'" in src)
    check("defaults preserve the historical [1,16] / [0,1,4,16]",
          "_default_h = [0, 1, 4, 16] if self._use_transformer_obs else [1, 16]" in src)
    check("horizons are used where they were hardcoded", "horizons = self._obs_horizons" in src)
    check("invalid horizons raise rather than defaulting silently",
          "expected a " in src and "obsHorizons=" in src)
    check("obsHorizons is whitelisted in the env-key validator", "'obsHorizons'," in src)


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
        check("committed r8 cfg passes", code == 0, f"(exit {code}: {err})")

        print("\n5. each sabotage must be REFUSED:")
        mutations = [
            ("horizons_removed", r"^  obsHorizons: \[.*$", "  # removed"),
            ("numobs_stale_3198", r"^  numObs: 9594.*$", "  numObs: 3198"),
            ("horizons_shortened_numobs_stale",
             r"^  obsHorizons: \[.*$", "  obsHorizons: [1, 16]"),
            ("rewardshape_removed", r"^  rewardShape: geometric.*$", "  # removed"),
            ("motion_reverted_to_cf",
             r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d_optj3d_cf"),
        ]
        src = open(CFG).read()
        for label, pat, rep in mutations:
            out, n = re.subn(pat, rep, src, count=1, flags=re.MULTILINE)
            assert n == 1, f"mutation {label!r} matched {n} -- fixture stale"
            path = os.path.join(tmp, f"m_{label}.yaml")
            open(path, "w").write(out)
            code, _ = run_guards(path, block, tmp)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

        print("\n6. sibling arms' cfgs must be refused:")
        for sib in ("r7_geom_train", "r6_cf2_train", "r3_roll30_train"):
            p = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sib}.yaml")
            if os.path.exists(p):
                code, _ = run_guards(p, block, tmp)
                check(f"refuses {sib}", code != 0)


def test_confound_is_recorded():
    """A forced confound that is not written down becomes a misattributed null."""
    print("\n7. the forced warm-start confound is documented:")
    sh = open(SCRIPT).read()
    check("launcher names the confound", "confound" in sh.lower())
    check("it explains WHY the warm start cannot be kept",
          "3198" in sh and "9594" in sh)
    check("it names the control run to settle it",
          "resume_from None" in sh or "resume_from: None" in sh)


def main():
    test_horizons()
    test_one_knob()
    test_task_code()
    test_guards()
    test_confound_is_recorded()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Pin the cross-body sub2 pair: r11_sub2_plain and r12_sub2_ret.

The first cross-body arms on the fast task. r11 puts sub2's body on the bball
subject's UNCHANGED reference (the no-retargeting baseline); r12 gives it a
reference re-solved onto sub2's proportions. They differ in motion_file alone,
so a difference between them IS the retargeting.

The trap this file mostly exists for: retarget_contact.py resolves a subject id
to smplx_omomo_<id>.xml, and smplx_omomo_sub100.xml is a synthetic OMOMO body,
NOT the CARI4D bball subject (smplh_behave_sub100.xml). Same number, different
person. A build command missing --source-mjcf retargets from the wrong body and
reports nothing wrong, so the launcher must carry the flag.

Run:  python3 tests/test_crossbody_sub2_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
RLG = os.path.join(CFGDIR, "train/rlg")
PLAIN, RET, BASE = "r11_sub2_plain", "r12_sub2_ret", "r8_horiz"
RETARGET_DIR = "InterAct/behave_cari4d_optj3d_cf2_sub2"

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


def env(arm, t="train"):
    return flat(yaml.safe_load(open(f"{CFGDIR}/omomo_cari4d_bball_{arm}_{t}.yaml")))


def diff(a, b):
    return sorted({k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")})


def test_one_knob_chain():
    print("1. the chain r8 -> r11 -> r12 moves one knob at a time:")
    b, p, r = env(BASE), env(PLAIN), env(RET)
    check("r11 differs from r8 ONLY in robotType",
          diff(b, p) == ["env.robotType"], f"({diff(b, p)})")
    check("r12 differs from r11 ONLY in motion_file",
          diff(p, r) == ["env.motion_file"], f"({diff(p, r)})")
    check("r11 uses sub2's body",
          p.get("env.robotType") == "smplx/smplx_omomo_sub2.xml",
          f"(got {p.get('env.robotType')})")
    check("r12 uses the SAME body as r11",
          r.get("env.robotType") == p.get("env.robotType"))
    check("r11 keeps the UNRETARGETED reference",
          p.get("env.motion_file") == "InterAct/behave_cari4d_optj3d_cf2")
    check("r12 uses the retargeted reference",
          r.get("env.motion_file") == RETARGET_DIR)


def test_recipe_preserved():
    print("\n2. r8's recipe survives in both -- otherwise the comparison drifts:")
    for arm in (PLAIN, RET):
        e = env(arm)
        check(f"{arm}: geometric reward", e.get("env.rewardShape") == "geometric")
        check(f"{arm}: six obs horizons",
              e.get("env.obsHorizons") == [1, 4, 7, 10, 13, 16])
        check(f"{arm}: numObs matches the horizons", e.get("env.numObs") == 9594)
        check(f"{arm}: rolloutLength 50", e.get("env.rolloutLength") == 50)
        check(f"{arm}: dataSub still sub100 (the CLIP's name, not the body)",
              e.get("env.dataSub") == ["sub100"])
        rlg = flat(yaml.safe_load(open(f"{RLG}/omomo_cari4d_bball_{arm}_train.yaml")))
        check(f"{arm}: fresh start, no warm start",
              str(rlg.get("params.config.resume_from")) == "None",
              f"(got {rlg.get('params.config.resume_from')!r})")
        check(f"{arm}: its own experiment name",
              rlg.get("params.config.full_experiment_name")
              == f"smplx_cari4d_bball_{arm}")


def test_eval_twins_track():
    print("\n3. the eval twins carry the same body and reference:")
    for arm in (PLAIN, RET):
        t, e = env(arm, "train"), env(arm, "eval")
        check(f"{arm}: eval body matches train",
              e.get("env.robotType") == t.get("env.robotType"))
        check(f"{arm}: eval reference matches train",
              e.get("env.motion_file") == t.get("env.motion_file"))
        check(f"{arm}: eval obs shape matches train",
              e.get("env.numObs") == t.get("env.numObs")
              and e.get("env.obsHorizons") == t.get("env.obsHorizons"))


def guard_block(arm):
    lines = open(os.path.join(REPO, f"slurm_cari4d_bball_{arm}.sh")).read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    e = next(i for i, l in enumerate(lines) if "invocation:" in l)
    return "\n".join(lines[s:e])


def run_guards(arm, cfg_path, cwd):
    script = 'CFG_ENV="$1"\n' + guard_block(arm) + "\nexit 0\n"
    p = subprocess.run(["bash", "-c", script, "bash", cfg_path],
                       capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stderr.strip()


def test_guards():
    print("\n4. guards -- each arm must refuse the other's cfg:")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "InterAct/behave_cari4d_optj3d_cf2"))
        os.makedirs(os.path.join(tmp, RETARGET_DIR))
        for arm in (PLAIN, RET):
            code, err = run_guards(arm, f"{CFGDIR}/omomo_cari4d_bball_{arm}_train.yaml", tmp)
            check(f"{arm}: own cfg passes", code == 0, f"(exit {code}: {err})")
        code, _ = run_guards(PLAIN, f"{CFGDIR}/omomo_cari4d_bball_{RET}_train.yaml", tmp)
        check("r11 refuses r12's cfg (retargeted dir)", code != 0)
        code, _ = run_guards(RET, f"{CFGDIR}/omomo_cari4d_bball_{PLAIN}_train.yaml", tmp)
        check("r12 refuses r11's cfg (plain dir)", code != 0)
        code, _ = run_guards(PLAIN, f"{CFGDIR}/omomo_cari4d_bball_{BASE}_train.yaml", tmp)
        check("r11 refuses r8's cfg (bball subject's body)", code != 0)

        print("\n5. the missing-data guard fires when the retargeted dir is absent:")
        with tempfile.TemporaryDirectory() as bare:
            os.makedirs(os.path.join(bare, "InterAct/behave_cari4d_optj3d_cf2"))
            code, err = run_guards(RET, f"{CFGDIR}/omomo_cari4d_bball_{RET}_train.yaml", bare)
            check("absent retargeted data is refused", code != 0)
            check("the error gives the build command", "retarget_contact.py" in err)
            check("and that command carries --source-mjcf", "--source-mjcf" in err,
                  "(without it the script silently uses the WRONG sub100 body)")


def test_the_sub100_trap_is_documented():
    """Two different bodies are called sub100. If that stops being written down,
    someone regenerates the data from the wrong one and nothing complains."""
    print("\n6. the sub100 collision is spelled out:")
    sh = open(os.path.join(REPO, f"slurm_cari4d_bball_{RET}.sh")).read()
    check("names both files", "smplh_behave_sub100" in sh and "smplx_omomo_sub100" in sh)
    check("says they are different bodies",
          "wrong person" in sh or "NOT the bball subject" in sh)
    check("says --source-mjcf is required", "REQUIRED" in sh)
    rc = open(os.path.join(REPO, "scripts/retarget_contact.py")).read()
    check("the script itself explains the override", "smplh_behave_sub100.xml" in rc)


def main():
    test_one_knob_chain()
    test_recipe_preserved()
    test_eval_twins_track()
    test_guards()
    test_the_sub100_trap_is_documented()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Pin the warm-start CONTROLS: r3_roll30_fresh and r6_cf2_fresh.

Every bball arm since r2 warm starts from checkpoints/smplx_teachers_new/sub2.pth
-- InterMimic's own default, inherited down the config lineage from omomo.yaml
rather than chosen for this task. The teacher imitates subject 2 doing ~0.5 m/s
tabletop manipulation; this clip is 1.98 m/s median, 7.33 m/s peak, different
body, intermittent contact.

The one matched pair that already exists (rectinj3 vs rectinj3_warm, identical
apart from resume_from) says the prior HURTS: 1.48 fresh vs 0.57 warm at
comparable own-epochs, with the warm run training longer. But that pair ran on
behave_cari4d_rectinj3. These two arms test it on the data actually in use.

The design point these tests protect: each control SHARES its parent's env cfg
FILE. Not a copy -- the same path. A copy could drift; a shared file cannot, so
"one knob" is true by construction rather than by vigilance.

Run:  python3 tests/test_fresh_start_controls.py   (exit 0 = all green)
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
PAIRS = [("r3_roll30", "r3_roll30_fresh"), ("r6_cf2", "r6_cf2_fresh")]

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


def test_one_knob():
    print("1. each control differs from its parent in resume_from alone:")
    for parent, fresh in PAIRS:
        a = flat(yaml.safe_load(open(f"{RLG}/omomo_cari4d_bball_{parent}_train.yaml")))
        b = flat(yaml.safe_load(open(f"{RLG}/omomo_cari4d_bball_{fresh}_train.yaml")))
        diffs = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        check(f"{fresh}: rlg differs ONLY in name and resume_from",
              diffs == {"params.config.full_experiment_name",
                        "params.config.resume_from"}, f"({sorted(diffs)})")
        check(f"{fresh}: resume_from is None",
              str(b.get("params.config.resume_from")) == "None",
              f"(got {b.get('params.config.resume_from')!r})")
        check(f"{parent}: still warm started (the thing being controlled for)",
              "smplx_teachers_new" in str(a.get("params.config.resume_from")))
        check(f"{fresh}: distinct experiment name (no checkpoint collision)",
              b.get("params.config.full_experiment_name")
              != a.get("params.config.full_experiment_name"))


def test_shared_env_cfg():
    """The env cfg is SHARED, not copied. A copy is a place for drift to hide."""
    print("\n2. each control reads its parent's env cfg file, not a copy:")
    for parent, fresh in PAIRS:
        sh = open(os.path.join(REPO, f"slurm_cari4d_bball_{fresh}.sh")).read()
        env_line = re.search(r"^CFG_ENV=(\S+)", sh, re.MULTILINE)
        train_line = re.search(r"^CFG_TRAIN=(\S+)", sh, re.MULTILINE)
        check(f"{fresh}: CFG_ENV is {parent}'s own file",
              env_line and env_line.group(1).endswith(
                  f"omomo_cari4d_bball_{parent}_train.yaml"),
              f"(got {env_line.group(1) if env_line else None})")
        check(f"{fresh}: CFG_TRAIN is its own rlg cfg",
              train_line and train_line.group(1).endswith(
                  f"omomo_cari4d_bball_{fresh}_train.yaml"),
              f"(got {train_line.group(1) if train_line else None})")
        check(f"{fresh}: no duplicate env cfg was created",
              not os.path.exists(f"{CFGDIR}/omomo_cari4d_bball_{fresh}_train.yaml"),
              "(a copy exists -- it can drift from the parent)")


def test_guard():
    print("\n3. the guard refuses a cfg that has a warm start after all:")
    for parent, fresh in PAIRS:
        sh_path = os.path.join(REPO, f"slurm_cari4d_bball_{fresh}.sh")
        lines = open(sh_path).read().splitlines()
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("# Guard: this arm IS the absence"))
        end = next(i for i, l in enumerate(lines[start:], start)
                   if l.startswith("# --- resume resolution"))
        block = "\n".join(lines[start:end])
        with tempfile.TemporaryDirectory() as tmp:
            def run(path):
                script = 'CFG_TRAIN="$1"\n' + block + "\nexit 0\n"
                p = subprocess.run(["bash", "-c", script, "bash", path],
                                   capture_output=True, text=True, cwd=tmp)
                return p.returncode
            ok = run(f"{RLG}/omomo_cari4d_bball_{fresh}_train.yaml")
            bad = run(f"{RLG}/omomo_cari4d_bball_{parent}_train.yaml")
            check(f"{fresh}: its own cfg passes", ok == 0, f"(exit {ok})")
            check(f"{fresh}: {parent}'s warm-started cfg is REFUSED", bad != 0,
                  "(guard did not fire -- the control would duplicate its parent)")


def test_documented():
    print("\n4. the reasoning and the epoch-offset trap are written down:")
    for _, fresh in PAIRS:
        sh = open(os.path.join(REPO, f"slurm_cari4d_bball_{fresh}.sh")).read()
        check(f"{fresh}: cites the rectinj3 evidence",
              "1.48" in sh and "0.57" in sh)
        check(f"{fresh}: says that evidence does not transfer on its own",
              "does not" in sh and "rectinj3" in sh)
        check(f"{fresh}: warns the epoch counter starts at 0, not 13,005",
              "13,005" in sh or "13005" in sh)
        check(f"{fresh}: first-launch message no longer promises a warm start",
              "FRESH START" in sh and "first launch: EXPLICIT warm start" not in sh)
        check(f"{fresh}: carries no stopping rule", "KILL CRITERION" not in sh)


def main():
    test_one_knob()
    test_shared_env_cfg()
    test_guard()
    test_documented()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())


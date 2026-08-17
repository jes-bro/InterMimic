#!/usr/bin/env python3
"""Tests for the cfg-driven early-termination thresholds (resetThresholds).

Runs WITHOUT Isaac Gym (extracts _parse_reset_thresholds from intermimic.py
source, same trick as test_audit_guards). Pins the properties that protect the
main experiments:
  1. absent block -> EXACTLY the historical hardcoded values (0.5 / 0.5 / 2 / 10),
     so every gen-2 / OMOMO cfg (none of which set the block) is byte-identical.
  2. `false` disables a criterion (-> None); numbers override; unknown keys raise.
  3. the bball train + eval cfgs both carry the intended block (object/igRatio/
     contactSteps off, human default) and stay in lockstep with each other.
  4. no gen-2 / teacher cfg sets resetThresholds (nothing drifted silently).
  5. 'resetThresholds' is whitelisted in KNOWN_ENV_KEYS (else every run with the
     block dies at the typo guard).

Run:  python tests/test_reset_thresholds.py   (exit 0 = all green)
"""
import glob
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMIMIC = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

src = open(INTERMIMIC).read()
m = re.search(r"def _parse_reset_thresholds\(block\):.*?\n        return out\n", src, re.DOTALL)
assert m, "could not find _parse_reset_thresholds in intermimic.py"
ns = {}
exec("def _parse_reset_thresholds(block):" + m.group(0).split("(block):", 1)[1], ns)  # noqa: S102
parse = ns["_parse_reset_thresholds"]


def test_defaults_match_historical():
    for block in (None, {}):
        assert parse(block) == {"human": 0.5, "object": 0.5, "igRatio": 2.0,
                                "contactSteps": 10.0}, parse(block)
    print("ok: absent block == historical hardcoded values")


def test_disable_and_override():
    out = parse({"object": False, "igRatio": False, "contactSteps": False})
    assert out == {"human": 0.5, "object": None, "igRatio": None, "contactSteps": None}
    assert parse({"human": 0.8})["human"] == 0.8
    try:
        parse({"objcet": 0.5})
    except ValueError as e:
        assert "objcet" in str(e)
    else:
        raise AssertionError("unknown key must raise")
    print("ok: false disables, numbers override, typos raise")


def test_bball_cfgs_carry_the_block():
    # ORIGINAL experiment cfgs: untouched (no block) -- separate-run rule
    for f in ("omomo_cari4d_bball_train.yaml", "omomo_cari4d_bball_eval.yaml"):
        assert "resetThresholds" not in open(os.path.join(CFG, f)).read(), f
    # NORESET experiment: block present in both, in lockstep
    want = {"object": False, "igRatio": False, "contactSteps": False}
    train = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_noreset_train.yaml")))
    ev = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_noreset_eval.yaml")))
    assert train["env"].get("resetThresholds") == want, train["env"].get("resetThresholds")
    assert ev["env"].get("resetThresholds") == want, "eval twin out of lockstep"
    out = parse(train["env"]["resetThresholds"])
    assert out["human"] == 0.5 and out["object"] is None
    tr = open(os.path.join(CFG, "train/rlg/omomo_cari4d_bball_noreset_train.yaml")).read()
    assert "full_experiment_name: smplx_cari4d_bball_noreset" in tr
    # LOOSETERM: noreset + human also off; own experiment name; eval in lockstep
    want_lt = {"human": False, "object": False, "igRatio": False, "contactSteps": False}
    lt = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_looseterm_train.yaml")))
    lte = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_looseterm_eval.yaml")))
    assert lt["env"].get("resetThresholds") == want_lt, lt["env"].get("resetThresholds")
    assert lte["env"].get("resetThresholds") == want_lt, "looseterm eval twin out of lockstep"
    assert all(v is None for v in parse(want_lt).values())
    ltr = open(os.path.join(CFG, "train/rlg/omomo_cari4d_bball_looseterm_train.yaml")).read()
    assert "full_experiment_name: smplx_cari4d_bball_looseterm" in ltr
    # RECTINJ3: looseterm twin on the IMPROVED recon -- identical thresholds,
    # own experiment name, and the ONLY env difference is the motion dir.
    rt = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_rectinj3_train.yaml")))
    rte = yaml.safe_load(open(os.path.join(CFG, "omomo_cari4d_bball_rectinj3_eval.yaml")))
    assert rt["env"].get("resetThresholds") == want_lt, rt["env"].get("resetThresholds")
    assert rte["env"].get("resetThresholds") == want_lt, "rectinj3 eval twin out of lockstep"
    assert rt["env"]["motion_file"] == "InterAct/behave_cari4d_rectinj3", rt["env"]["motion_file"]
    assert lt["env"]["motion_file"] == "InterAct/behave_cari4d", "looseterm motion dir changed?!"
    rtr = open(os.path.join(CFG, "train/rlg/omomo_cari4d_bball_rectinj3_train.yaml")).read()
    assert "full_experiment_name: smplx_cari4d_bball_rectinj3" in rtr
    print("ok: original untouched; noreset/looseterm/rectinj3 trios each correct")


def test_no_other_cfg_sets_the_block():
    offenders = []
    for p in glob.glob(os.path.join(CFG, "omomo_teacher_*.yaml")):
        if "resetThresholds" in open(p).read():
            offenders.append(os.path.basename(p))
    assert not offenders, f"teacher cfgs must not set resetThresholds: {offenders}"
    print(f"ok: no teacher cfg sets resetThresholds ({len(glob.glob(os.path.join(CFG, 'omomo_teacher_*.yaml')))} checked)")


def test_key_whitelisted():
    m2 = re.search(r"KNOWN_ENV_KEYS = frozenset\((\{.*?\})\)", src, re.DOTALL)
    import ast
    assert "resetThresholds" in ast.literal_eval(m2.group(1))
    print("ok: resetThresholds in KNOWN_ENV_KEYS")


if __name__ == "__main__":
    test_defaults_match_historical()
    test_disable_and_override()
    test_bball_cfgs_carry_the_block()
    test_no_other_cfg_sets_the_block()
    test_key_whitelisted()
    print("ALL GREEN")

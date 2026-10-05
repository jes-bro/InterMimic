#!/usr/bin/env python3
"""scripts/render_arms.py against the eval runner it depends on.

What broke (2026-10-05): render_arms.py resolves each arm with
`EMIT=1 sh scripts/eval_one.sh <run>` and required BETAS_FILE and BASE_YAML from
it. eval_one.sh had stopped emitting both when it moved to per-arm eval configs,
so every render died at "EMIT gave no BETAS_FILE". Nothing tied the two files
together. These tests do:

  * every key render_arms.py requires is a key eval_one.sh's EMIT line prints;
  * the render config is the arm's OWN eval config with only the per-render keys
    patched -- in particular retargetedMotionDir survives, so a retargeted arm is
    rendered tracking its retargeted reference;
  * two arms sharing an eval config get byte-identical render configs, i.e. the
    same environment and the same reference file.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_render_arms.py -q
"""
import importlib.util
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
SRCALL13_EVAL = CFG / "omomo_eval_g3_omomo_geoall_srcall13__f0.yaml"

_spec = importlib.util.spec_from_file_location("render_arms", REPO / "scripts/render_arms.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def emitted_keys():
    """The KEY names in eval_one.sh's EMIT printf format string."""
    txt = (REPO / "scripts/eval_one.sh").read_text()
    m = re.search(r'if \[ "\$\{EMIT:-0\}" = 1 \]; then\s*\n\s*printf "([^"]*)"', txt)
    assert m, "eval_one.sh: EMIT printf not found -- the resolver's interface moved"
    # the format string separates pairs with a literal backslash-n: KEY='%s'\nKEY='%s'...
    return {pair.split("=")[0] for pair in m.group(1).split("\\n") if "='%s'" in pair}


def test_every_required_plan_key_is_emitted_by_eval_one():
    keys = emitted_keys()
    assert keys, "no KEY='%s' pairs parsed out of the EMIT printf"
    missing = [k for k in ra.REQUIRED_PLAN_KEYS if k not in keys]
    assert not missing, f"render_arms.py requires {missing}, which eval_one.sh EMIT does not print ({sorted(keys)})"
    assert "BETAS_FILE" not in ra.REQUIRED_PLAN_KEYS and "BASE_YAML" not in ra.REQUIRED_PLAN_KEYS


def test_render_cfg_is_the_arms_eval_cfg_with_only_the_render_keys_patched():
    base = yaml.safe_load(SRCALL13_EVAL.read_text())
    out = ra.make_render_yaml(str(SRCALL13_EVAL), "sub16", "sub2", "largetable", 1)
    new = yaml.safe_load(Path(out).read_text())
    env0, env1 = base["env"], new["env"]
    assert env1["subjectBodies"] == ["sub16"] and env1["dataSub"] == ["sub2"]
    assert env1["dataObjects"] == ["largetable"] and env1["maxClipsPerObject"] == 1
    assert env1["numEnvs"] == 1
    patched = {"subjectBodies", "dataSub", "dataObjects", "maxClipsPerObject", "numEnvs"}
    for k in set(env0) | set(env1):
        if k not in patched:
            assert env0.get(k) == env1.get(k), f"render changed env.{k}: {env0.get(k)!r} -> {env1.get(k)!r}"
    # the point of the fix: the retargeted reference tree is the arm's own, untouched
    assert env1["retargetedMotionDir"] == "InterAct/OMOMO_retarget_contact_srcall13"
    assert new.get("sim") == base.get("sim")


def test_reference_render_only_adds_playdataset_and_a_pinned_motion_dir():
    plain = yaml.safe_load(Path(ra.make_render_yaml(str(SRCALL13_EVAL), "sub16", "sub2", "largetable", 1)).read_text())
    ref = yaml.safe_load(Path(ra.make_render_yaml(str(SRCALL13_EVAL), "sub16", "sub2", "largetable", 1,
                                                  motion_dir="/tmp/pinned_one_clip", playdataset=True)).read_text())
    diff = {k for k in set(plain["env"]) | set(ref["env"]) if plain["env"].get(k) != ref["env"].get(k)}
    assert diff == {"playdataset", "motion_file"}
    assert ref["env"]["playdataset"] is True and ref["env"]["motion_file"] == "/tmp/pinned_one_clip"
    assert ref["env"]["retargetedMotionDir"] == plain["env"]["retargetedMotionDir"]   # the replay shows the retargeted ref


def test_two_arms_sharing_an_eval_cfg_get_identical_render_configs():
    """The srcall13 MLP and transformer teachers are both served by this one eval
    config (its evalFor lists both), so their renders differ in the checkpoint and
    network only -- same environment, same clip, same retargeted reference."""
    served = yaml.safe_load(SRCALL13_EVAL.read_text())["evalFor"]
    assert "g3_omomo_geoall_srcall13__f0" in served and "g3_omomo_geoall_srcall13_xf_nvadlr__f0" in served
    a = Path(ra.make_render_yaml(str(SRCALL13_EVAL), "sub16", "sub2", "trashcan", 1)).read_text()
    b = Path(ra.make_render_yaml(str(SRCALL13_EVAL), "sub16", "sub2", "trashcan", 1)).read_text()
    assert a == b

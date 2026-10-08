#!/usr/bin/env python3
"""The 3 gen4 EgoExo teacher arms (scripts/make_gen4_egoexo_teacher_cfgs.py):

  bball7, soccer15   exact-five-edit copies of their g3 transformer twins
                     (subjectBodies, subjectHeightsFile, humanoidAssetSuffix,
                     motionScaleReward.enable, motionScaleReward.referenceStd)
  act22              the bball7 gen4 arm + the generalist edits (22 sources, act
                     motion dir + tree, objectPropsFile instead of objectMass /
                     objectShapeProps, plane 0.7)
  train cfgs         full_experiment_name only
  launchers          own names/paths, HELDOUT hint with sub4, gen4 + motion-scale
                     guards; the generalist swaps the objectMass guard for an
                     objectPropsFile guard and loops over the 22 sources

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_egoexo_teacher_cfgs.py -q
"""
import importlib.util
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
TRAIN_REALS = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
               "sub11", "sub12", "sub14", "sub15", "sub17"]
GEN4 = [f"sub{i}" for i in range(600, 630)]
TEST = {"sub4", "sub10", "sub13", "sub16"}
GEN4_EDITS = {"env.subjectBodies", "env.subjectHeightsFile", "env.humanoidAssetSuffix",
              "env.motionScaleReward.enable", "env.motionScaleReward.referenceStd"}


def _gen():
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_egoexo_teacher_cfgs.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    return mk


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def differing_keys(a, b):
    fa, fb = flatten(a), flatten(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _env(name):
    return yaml.safe_load(open(CFG / f"omomo_teacher_{name}.yaml"))


def _assert_gen4_edits(env, name):
    assert env["subjectBodies"] == TRAIN_REALS + GEN4, name
    assert not TEST & set(env["subjectBodies"]), name
    assert env["subjectHeightsFile"] == "scripts/synthetic_heights_gen4.json", name
    assert env["humanoidAssetSuffix"] == "_inertial", name
    assert env["motionScaleReward"] == {"enable": True, "referenceStd": 0.5157}, name
    assert env["rewardShape"] == "geometric_all" and env["rewardTerms"]["pose"]["enable"] is False, name
    assert "betas_file" not in env, name


def test_specialists_are_five_edit_copies_of_their_g3_twins():
    mk = _gen()
    for name in ("bball7", "soccer15"):
        base, srcs, mem, general = mk.ARMS[name]
        assert not general and mem is None
        b, n = _env(base), _env(mk.arm(name))
        assert differing_keys(b, n) == GEN4_EDITS, name
        _assert_gen4_edits(n["env"], name)
        assert n["env"]["dataSub"] == srcs == b["env"]["dataSub"], name
        # the activity-specific physics is untouched
        for k in ("objectMass", "objectShapeProps", "plane", "motion_file", "retargetedMotionDir"):
            assert n["env"][k] == b["env"][k], (name, k)


def test_generalist_is_the_bball7_gen4_arm_plus_the_student_data_layout():
    mk = _gen()
    base, srcs, mem, general = mk.ARMS["act22"]
    assert general and mem == "96G"
    g, bb = _env(mk.arm("act22")), _env(mk.arm("bball7"))
    _assert_gen4_edits(g["env"], "act22")
    assert differing_keys(bb, g) == {"env.dataSub", "env.motion_file", "env.retargetedMotionDir",
                                     "env.objectMass", "env.objectShapeProps.restitution",
                                     "env.objectPropsFile", "env.plane.restitution"}
    e = g["env"]
    assert e["dataSub"] == srcs and len(srcs) == 22
    assert set(srcs) == set(mk.BBALL7) | set(mk.SOCCER15) and not (set(mk.BBALL7) & set(mk.SOCCER15))
    assert not any(500 <= int(s[3:]) < 600 for s in srcs)                # CPR (sub5xx) left out
    assert e["motion_file"] == "InterAct/behave_cari4d_act"
    assert e["retargetedMotionDir"] == "InterAct/behave_cari4d_act_f0_bodymajor"
    assert "objectMass" not in e and "objectShapeProps" not in e
    assert e["objectPropsFile"] == "isaacgym/src/intermimic/data/cfg/object_props_g3_act.yaml"
    assert e["plane"] == {"dynamicFriction": 0.9, "restitution": 0.7, "staticFriction": 0.9}
    # ... which is exactly how the g3 activity STUDENT lays out the same data
    st = yaml.safe_load(open(CFG / "omomo_student_g3_act_xf_ret_nvadlr__f0.yaml"))["env"]
    for k in ("motion_file", "retargetedMotionDir", "objectPropsFile", "plane"):
        assert e[k] == st[k], k
    assert set(e["dataSub"]) < set(st["dataSub"])                        # the student also has CPR


def test_train_cfgs_name_only():
    mk = _gen()
    for name, (base, *_) in mk.ARMS.items():
        b = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"))
        n = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{mk.arm(name)}.yaml"))
        assert differing_keys(b, n) == {"params.config.full_experiment_name"}, name
        assert n["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{mk.arm(name)}", name
        assert n["params"]["network"]["name"] == "intermimic_transformer", name


def test_launchers():
    mk = _gen()
    for name, (base, srcs, mem, general) in mk.ARMS.items():
        txt = (REPO / f"slurm_teacher_{mk.arm(name)}.sh").read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        # the g3 base is named in comments and in the provenance echo; it must never be
        # what the launcher RUNS: no cfg path, train cfg or checkpoint dir of the base
        assert f"omomo_teacher_{base}" not in code and f"smplx_teacher_{base}" not in code, name
        assert f"slurm_teacher_{base}" not in code, name
        assert f"omomo_teacher_{mk.arm(name)}.yaml" in code and f"train/rlg/omomo_teacher_{mk.arm(name)}.yaml" in code, name
        assert 'HELDOUT="sub4 sub10 sub13 sub16"' in txt, name
        assert "_inertial.xml" in code and "RT_DIR/$b" in code, name            # gen4 guards
        assert "motionScaleReward" in code and "referenceStd:\\s*0.5157" in code, name
        assert f"for s in {' '.join(srcs)}; do" in code, name                   # per-source data loop
        assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(mk.arm(name)), txt, flags=re.M), name
        assert "--time=7-00:00:00" in txt, name
        if general:
            assert "#SBATCH --mem=96G\n" in txt and "objectPropsFile" in code and "objectMass:" not in code, name
        else:
            assert "#SBATCH --mem=64G\n" in txt and "objectMass:" in code, name

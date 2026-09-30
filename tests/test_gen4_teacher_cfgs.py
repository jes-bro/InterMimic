#!/usr/bin/env python3
"""The 13 gen4 OMOMO teacher arms (scripts/make_gen4_teacher_cfgs.py), each an
exact-three-edit copy of its g3 nopose transformer twin:

  env      subjectBodies = 13 fold-0 training reals + sub600..sub629 (43);
           subjectHeightsFile -> gen4; humanoidAssetSuffix: _inertial; nothing else
  train    full_experiment_name only
  launcher its own names/paths, HELDOUT hint has sub4, the two gen4 guards

Also pins the data the arms point at: the heights file covers sub600-629,
no test body (sub4 sub10 sub13 sub16) ever trains, and eval_one.sh holds out
four bodies for gen4_ arms.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_teacher_cfgs.py -q
"""
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
SOURCES = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]
TRAIN_REALS = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
               "sub11", "sub12", "sub14", "sub15", "sub17"]
GEN4 = [f"sub{i}" for i in range(600, 630)]
TEST = {"sub4", "sub10", "sub13", "sub16"}


def g3(s): return f"g3_omomo_geoall_src{s}_xf_nvadlr_nopose__f0"
def gen4(s): return f"gen4_omomo_geoall_src{s}_xf_nvadlr_nopose__f0"


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


def test_env_cfgs_three_edits_only():
    for s in SOURCES:
        base = yaml.safe_load(open(CFG / f"omomo_teacher_{g3(s)}.yaml"))
        new = yaml.safe_load(open(CFG / f"omomo_teacher_{gen4(s)}.yaml"))
        assert differing_keys(base, new) == {"env.subjectBodies", "env.subjectHeightsFile",
                                             "env.humanoidAssetSuffix"}, s
        env = new["env"]
        assert env["subjectBodies"] == TRAIN_REALS + GEN4, s
        assert not TEST & set(env["subjectBodies"]), s
        assert env["humanoidAssetSuffix"] == "_inertial", s
        assert env["subjectHeightsFile"] == "scripts/synthetic_heights_gen4.json", s
        assert env["dataSub"] == [f"sub{s}"], s
        assert "betas_file" not in env, s                      # nobetas recipe
        assert env["rewardTerms"]["pose"]["enable"] is False, s
        assert env["retargetedMotionDir"] == f"InterAct/OMOMO_retarget_contact_src{s}", s


def test_train_cfgs_name_only():
    for s in SOURCES:
        base = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{g3(s)}.yaml"))
        new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{gen4(s)}.yaml"))
        assert differing_keys(base, new) == {"params.config.full_experiment_name"}, s
        assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{gen4(s)}", s


def test_launchers():
    for s in SOURCES:
        txt = (REPO / f"slurm_teacher_{gen4(s)}.sh").read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        assert g3(s) not in code, s              # the g3 name may appear in comments (provenance) only
        assert f"omomo_teacher_{gen4(s)}.yaml" in txt and f"train/rlg/omomo_teacher_{gen4(s)}.yaml" in txt, s
        assert 'HELDOUT="sub4 sub10 sub13 sub16"' in txt, s
        assert "humanoidAssetSuffix" in txt and "_inertial.xml" in txt and "RT_DIR/$b" in txt, s
        assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(gen4(s)), txt, flags=re.M), s
        assert "--time=7-00:00:00" in txt, s


def test_heights_cover_gen4_bodies():
    h = json.load(open(REPO / "scripts/synthetic_heights_gen4.json"))
    assert set(h) == {str(i) for i in range(600, 630)}
    assert all(1.4 <= v <= 2.1 for v in h.values())


def test_eval_one_holds_out_four_for_gen4():
    txt = (REPO / "scripts/eval_one.sh").read_text()
    assert 'gen4_*) HELDOUT_DEFAULT="sub4 sub10 sub16 sub13"' in txt

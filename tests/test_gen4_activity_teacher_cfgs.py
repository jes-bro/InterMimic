#!/usr/bin/env python3
"""The 4 gen4 OBJECT-GROUP teacher arms (scripts/make_gen4_activity_teacher_cfgs.py),
each an exact-four-edit copy of the gen4 src1 nopose transformer teacher:

  env      dataSub = 13 training sources; dataObjects = the group; raggedMotionData
           true; retargetedMotionDir = the merged srcall13 tree; nothing else
  train    full_experiment_name only
  launcher its own names/paths, its own --mem, the ragged + merged-tree guards,
           the gen4 guards and HELDOUT hint kept

Also pins the grouping itself: the four groups partition the 13 OMOMO objects,
no object twice, none missing, and the budget script's object filter counts
clips the way the task's dataObjects filter does.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_activity_teacher_cfgs.py -q
"""
import importlib.util
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
BASE = "gen4_omomo_geoall_src1_xf_nvadlr_nopose__f0"
TRAIN_SOURCES = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
                 "sub11", "sub12", "sub14", "sub15", "sub17"]
TEST = {"sub4", "sub10", "sub13", "sub16"}
GROUPS = {
    "boxes": ["largebox", "smallbox", "plasticbox", "suitcase", "trashcan", "monitor"],
    "stands": ["tripod", "clothesstand", "floorlamp"],
    "chairs": ["whitechair", "woodchair"],
    "tables": ["largetable", "smalltable"],
}
ALL_OBJECTS = {"clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
               "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"}


def arm(g, tag=""): return f"gen4_omomo_geoall_{g}_xf_nvadlr_nopose{tag}__f0"


def _gen():
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_activity_teacher_cfgs.py")
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


def test_groups_partition_the_objects():
    listed = [o for objs in GROUPS.values() for o in objs]
    assert len(listed) == len(set(listed)), "an object is in two groups"
    assert set(listed) == ALL_OBJECTS


def test_env_cfgs_four_edits_only():
    base = yaml.safe_load(open(CFG / f"omomo_teacher_{BASE}.yaml"))
    for g, objs in GROUPS.items():
        new = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(g)}.yaml"))
        assert differing_keys(base, new) == {"env.dataSub", "env.dataObjects",
                                             "env.raggedMotionData", "env.retargetedMotionDir"}, g
        env = new["env"]
        assert env["dataSub"] == TRAIN_SOURCES, g
        assert not TEST & set(env["dataSub"]), g
        assert env["dataObjects"] == objs, g
        assert env["raggedMotionData"] is True, g
        assert env["retargetedMotionDir"] == "InterAct/OMOMO_retarget_contact_srcall13", g
        # inherited from the gen4 base, re-asserted because they are what makes it a gen4 arm
        assert env["humanoidAssetSuffix"] == "_inertial", g
        assert not TEST & set(env["subjectBodies"]), g
        assert len(env["subjectBodies"]) == 43, g
        assert env["rewardTerms"]["pose"]["enable"] is False, g
        assert "betas_file" not in env, g


def test_train_cfgs_name_only():
    base = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{BASE}.yaml"))
    for g in GROUPS:
        new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm(g)}.yaml"))
        assert differing_keys(base, new) == {"params.config.full_experiment_name"}, g
        assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm(g)}", g


def test_launchers():
    for g, objs in GROUPS.items():
        txt = (REPO / f"slurm_teacher_{arm(g)}.sh").read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        assert BASE not in code, g                # the base name may appear in comments (provenance) only
        assert f"omomo_teacher_{arm(g)}.yaml" in txt and f"train/rlg/omomo_teacher_{arm(g)}.yaml" in txt, g
        assert 'HELDOUT="sub4 sub10 sub13 sub16"' in txt, g
        assert "raggedMotionData" in code, g
        assert f"for o in {' '.join(objs)}; do" in code, g
        assert "for b in sub2 sub600; do" in code, g
        # the gen4 guards survive the rewrite
        assert "humanoidAssetSuffix" in code and "_inertial.xml" in code and "RT_DIR/$b" in code, g
        assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(arm(g)), txt, flags=re.M), g
        assert re.search(r"^#SBATCH --mem=\d+G$", txt, flags=re.M), g
        assert "--time=7-00:00:00" in txt, g
        assert f"GEN4 OBJECT-GROUP RECIPE {arm(g)}" in code, g


def test_launcher_mem_matches_generator_table():
    spec = importlib.util.spec_from_file_location(
        "mk", REPO / "scripts/make_gen4_activity_teacher_cfgs.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    for g, (objs, mem) in mk.GROUPS.items():
        assert objs == GROUPS[g], g
        txt = (REPO / f"slurm_teacher_{arm(g)}.sh").read_text()
        assert f"#SBATCH --mem={mem}\n" in txt, g


def test_budget_object_filter_matches_task_rule(tmp_path):
    """motion_memory_budget.py --objects keeps exactly the clips intermimic.py's
    dataObjects filter keeps (object = second-to-last '_' field of the stem)."""
    import torch
    spec = importlib.util.spec_from_file_location("mb", REPO / "scripts/motion_memory_budget.py")
    mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)
    for name, t in [("sub1_largebox_000.pt", 5), ("sub1_monitor_000.pt", 7),
                    ("sub2_woodchair_000.pt", 11), ("sub2_largebox_001.pt", 13)]:
        torch.save(torch.zeros(t, 3), tmp_path / name)
    got = mb.clip_lengths(str(tmp_path), ["sub1", "sub2"], ["largebox", "monitor"])
    assert got == {"sub1": [5, 7], "sub2": [13]}
    got = mb.clip_lengths(str(tmp_path), ["sub1", "sub2"], ["woodchair"])   # sub1 has none: warn, drop
    assert got == {"sub2": [11]}
    got = mb.clip_lengths(str(tmp_path), ["sub1", "sub2"])                  # no filter: everything
    assert got == {"sub1": [5, 7], "sub2": [13, 11]}                         # filename-sorted order


# --- the _msexp variant: the same four arms with the motion-scale reward exponent on ---

def test_msexp_env_cfgs_five_edits_only():
    mk = _gen()
    base = yaml.safe_load(open(CFG / f"omomo_teacher_{BASE}.yaml"))
    block = {"env.motionScaleReward.enable", "env.motionScaleReward.referenceStd"}
    for g, objs in GROUPS.items():
        plain = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(g)}.yaml"))
        new = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(g, mk.MS_TAG)}.yaml"))
        assert differing_keys(base, new) == {"env.dataSub", "env.dataObjects", "env.raggedMotionData",
                                             "env.retargetedMotionDir"} | block, g
        # identical to the plain group arm except the block: the reward is the ONLY extra change
        assert differing_keys(plain, new) == block, g
        assert new["env"]["motionScaleReward"] == {"enable": True, "referenceStd": mk.MS_REFERENCE_STD}, g
        assert "floorStd" not in new["env"]["motionScaleReward"], g       # capped at product: no floor
        assert mk.MS_REFERENCE_STD == 0.5157                       # OMOMO_new median key-body std, all 4421 clips
        assert new["env"]["rewardShape"] == "geometric_all", g     # the exponent replaces this shape's root
        assert new["env"]["rewardTerms"]["pose"]["enable"] is False, g   # 4 factors under the root


def test_msexp_train_cfgs_name_only():
    mk = _gen()
    base = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{BASE}.yaml"))
    for g in GROUPS:
        new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm(g, mk.MS_TAG)}.yaml"))
        assert differing_keys(base, new) == {"params.config.full_experiment_name"}, g
        assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm(g, mk.MS_TAG)}", g


def test_msexp_launchers_carry_the_guard_and_their_own_names():
    mk = _gen()
    for g, objs in GROUPS.items():
        txt = (REPO / f"slurm_teacher_{arm(g, mk.MS_TAG)}.sh").read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        assert BASE not in code and arm(g) + ".yaml" not in code, g      # never the plain arm's cfgs
        assert f"omomo_teacher_{arm(g, mk.MS_TAG)}.yaml" in code, g
        assert "motionScaleReward" in code and f"referenceStd:\\s*{mk.MS_REFERENCE_STD}" in code, g
        assert "floorStd" not in txt, g
        assert "raggedMotionData" in code and f"for o in {' '.join(objs)}; do" in code, g
        assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(arm(g, mk.MS_TAG)), txt, flags=re.M), g
        assert "MOTION-SCALE EXPONENT" in code, g

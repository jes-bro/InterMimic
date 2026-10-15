#!/usr/bin/env python3
"""The 13 gen4 SINGLE-OBJECT teacher arms (scripts/make_gen4_activity_teacher_cfgs.py
--single-object --motion-scale), one per OMOMO object, msexp only. Each is the
gen4 src1 nopose transformer teacher with exactly FIVE env edits:

  env      dataSub = 13 training sources; dataObjects = [the object];
           raggedMotionData true; retargetedMotionDir = merged srcall13 tree;
           motionScaleReward {enable, referenceStd 0.5157}; nothing else
  train    full_experiment_name only
  launcher its own names/paths, its own --mem, the ragged + merged-tree + motion-
           scale guards, AND the single-object pre-flight (count_motion_clips.py
           with the arm's baked clip count)

The object FILTER is the experiment, so it is checked against the REAL dataset
(~/new_one/OMOMO_new, all 4421 clips; hard-fails if absent -- no fixtures): per
arm, the task's filename rule keeps exactly the baked number of clips, all of
that one object, and the 13 arms partition the 3356 training clips.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_single_object_teacher_cfgs.py -q
"""
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
BASE = "gen4_omomo_geoall_src1_xf_nvadlr_nopose__f0"
TRAIN_SOURCES = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
                 "sub11", "sub12", "sub14", "sub15", "sub17"]
ALL_OBJECTS = {"clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
               "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"}
OMOMO_NEW = Path(os.path.expanduser("~/new_one/OMOMO_new"))     # the full release, 4421 clips
TRAIN_CLIPS = 3356                                               # 4421 minus sub4/10/13/16


def arm(o): return f"gen4_omomo_geoall_{o}_xf_nvadlr_nopose_msexp__f0"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _gen(): return _load("mk", "scripts/make_gen4_activity_teacher_cfgs.py")


def _counter(): return _load("cmc", "scripts/count_motion_clips.py")


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


@pytest.fixture(scope="module")
def dataset():
    if not OMOMO_NEW.is_dir():
        pytest.fail(f"real OMOMO_new not found at {OMOMO_NEW}; this test runs on the real data, no fixtures")
    files = sorted(f for f in os.listdir(OMOMO_NEW) if f.endswith(".pt"))
    assert len(files) == 4421, f"expected the full release (4421 clips), found {len(files)}"
    return files


# --- the table itself -------------------------------------------------------------------

def test_singles_cover_the_13_objects_exactly_once():
    mk = _gen()
    assert sorted(mk.SINGLES) == sorted(ALL_OBJECTS)
    for o, (objs, _) in mk.SINGLES.items():
        assert objs == [o], o
    assert sorted(mk.SINGLE_CLIPS_GIB) == sorted(ALL_OBJECTS)
    assert sum(n for n, _ in mk.SINGLE_CLIPS_GIB.values()) == TRAIN_CLIPS


def test_mem_rule_matches_the_group_budgets_and_floors_at_64():
    mk = _gen()
    # the four group budgets, as measured -> as written (the rule the singles follow)
    assert mk.single_mem(94) == "112G" and mk.single_mem(104) == "112G"
    assert mk.single_mem(128) == "144G" and mk.single_mem(189) == "208G"
    assert mk.single_mem(33) == "64G" and mk.single_mem(10) == "64G"           # floor
    assert mk.single_mem(67) == "80G" and mk.single_mem(63) == "80G"
    for o, (_, gib) in mk.SINGLE_CLIPS_GIB.items():
        assert mk.SINGLES[o][1] == mk.single_mem(gib), o


def test_single_object_requires_motion_scale():
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_activity_teacher_cfgs.py"), "--single-object"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "motion-scale" in (r.stdout + r.stderr)


def test_no_plain_single_object_arms_exist():
    for o in ALL_OBJECTS:
        assert not (CFG / f"omomo_teacher_gen4_omomo_geoall_{o}_xf_nvadlr_nopose__f0.yaml").exists(), o
        assert not (REPO / f"slurm_teacher_gen4_omomo_geoall_{o}_xf_nvadlr_nopose__f0.sh").exists(), o


# --- the written cfgs ---------------------------------------------------------------------

def test_env_cfgs_five_edits_only():
    mk = _gen()
    base = yaml.safe_load(open(CFG / f"omomo_teacher_{BASE}.yaml"))
    for o in ALL_OBJECTS:
        new = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(o)}.yaml"))
        assert differing_keys(base, new) == {"env.dataSub", "env.dataObjects", "env.raggedMotionData",
                                             "env.retargetedMotionDir", "env.motionScaleReward.enable",
                                             "env.motionScaleReward.referenceStd"}, o
        e = new["env"]
        assert e["dataObjects"] == [o], o
        assert e["dataSub"] == TRAIN_SOURCES, o
        assert e["raggedMotionData"] is True, o
        assert e["retargetedMotionDir"] == mk.MERGED_TREE, o
        assert e["motionScaleReward"] == {"enable": True, "referenceStd": mk.MS_REFERENCE_STD}, o
        assert e["humanoidAssetSuffix"] == "_inertial", o
        assert e["rewardTerms"]["pose"]["enable"] is False, o
        assert not ({"sub4", "sub10", "sub13", "sub16"} & set(e["subjectBodies"])), o
        assert len(e["subjectBodies"]) == 43, o


def test_single_arm_differs_from_its_group_twin_by_the_object_list_only():
    """A single-object arm is its group's msexp arm with a one-element dataObjects."""
    groups = {"boxes": ["largebox", "smallbox", "plasticbox", "suitcase", "trashcan", "monitor"],
              "stands": ["tripod", "clothesstand", "floorlamp"],
              "chairs": ["whitechair", "woodchair"], "tables": ["largetable", "smalltable"]}
    for g, objs in groups.items():
        grp = yaml.safe_load(open(CFG / f"omomo_teacher_gen4_omomo_geoall_{g}_xf_nvadlr_nopose_msexp__f0.yaml"))
        for o in objs:
            single = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(o)}.yaml"))
            assert differing_keys(grp, single) == {"env.dataObjects"}, (g, o)


def test_train_cfgs_name_only():
    base = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{BASE}.yaml"))
    for o in ALL_OBJECTS:
        new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm(o)}.yaml"))
        assert differing_keys(base, new) == {"params.config.full_experiment_name"}, o
        assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm(o)}", o


def test_launchers_guards_names_and_memory():
    mk = _gen()
    for o in ALL_OBJECTS:
        n_clips, _ = mk.SINGLE_CLIPS_GIB[o]
        txt = (REPO / f"slurm_teacher_{arm(o)}.sh").read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        assert BASE not in code, o                                           # never the src1 arm's cfgs
        assert f"omomo_teacher_{arm(o)}.yaml" in code, o
        assert f"#SBATCH --mem={mk.SINGLES[o][1]}\n" in txt, o
        assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(arm(o)), txt, flags=re.M), o
        assert "raggedMotionData" in code and f"for o in {o}; do" in code, o          # merged-tree guard
        assert "motionScaleReward" in code and f"referenceStd:\\s*{mk.MS_REFERENCE_STD}" in code, o
        # the pre-flight: exact clip count, exact object, before the GPU
        assert (f"scripts/count_motion_clips.py --cfg \"$CFG_ENV\" --expect-clips {n_clips} "
                f"--expect-objects {o}") in code, o
        assert code.index("count_motion_clips.py") < code.index("python -u -m intermimic.run"), o
        assert "SINGLE-OBJECT" in txt and f"expanded {n_clips} clips" in txt, o


# --- the filter, against the real data --------------------------------------------------

def test_each_arm_keeps_exactly_its_clips_all_one_object(dataset):
    mk, cmc = _gen(), _counter()
    seen = []
    for o in ALL_OBJECTS:
        env = yaml.safe_load(open(CFG / f"omomo_teacher_{arm(o)}.yaml"))["env"]
        kept = cmc.kept_clips(OMOMO_NEW, env["dataSub"], env["dataObjects"])
        n_clips, _ = mk.SINGLE_CLIPS_GIB[o]
        assert len(kept) == n_clips, (o, len(kept))
        assert {obj for _, _, obj in kept} == {o}, o
        assert all(t in {int(s[3:]) for s in TRAIN_SOURCES} for _, t, _ in kept), o
        seen += [f for f, _, _ in kept]
    assert len(seen) == len(set(seen)) == TRAIN_CLIPS          # a partition: every training clip once


def test_count_script_refuses_a_wrong_object_or_count(dataset):
    cfg = CFG / f"omomo_teacher_{arm('smallbox')}.yaml"
    base = [sys.executable, str(REPO / "scripts/count_motion_clips.py"), "--cfg", str(cfg),
            "--motion-dir", str(OMOMO_NEW)]
    ok = subprocess.run(base + ["--expect-clips", "144", "--expect-objects", "smallbox"], capture_output=True, text=True)
    assert ok.returncode == 0 and "OK" in ok.stdout, ok.stdout + ok.stderr
    bad_obj = subprocess.run(base + ["--expect-clips", "144", "--expect-objects", "largebox"], capture_output=True, text=True)
    assert bad_obj.returncode == 1 and "REFUSING" in bad_obj.stderr
    bad_n = subprocess.run(base + ["--expect-clips", "145", "--expect-objects", "smallbox"], capture_output=True, text=True)
    assert bad_n.returncode == 1 and "expected 145 clips, got 144" in bad_n.stderr


def test_filter_rule_mirrors_the_task(dataset):
    """The counter's rule is the task's (intermimic.py _load_motion): object = 2nd-to-last
    '_' token of the stem; target subject from the first token; non-.pt skipped."""
    cmc = _counter()
    assert cmc.object_of("sub6_woodchair_012.pt") == "woodchair"
    assert cmc.object_of("sub2to8_largetable_007.pt") == "largetable"
    assert cmc.parse_target_subject("sub2to8_largetable_007.pt") == 8
    assert cmc.parse_target_subject("sub12_tripod_003.pt") == 12
    assert cmc.parse_target_subject("README.md") is None
    assert cmc.parse_target_subject("notes_sub2.pt") is None
    # no dataObjects = everything of those sources (the per-source teachers' case)
    assert len(cmc.kept_clips(OMOMO_NEW, ["sub2"], None)) == 52

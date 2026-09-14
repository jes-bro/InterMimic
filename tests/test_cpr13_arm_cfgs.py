#!/usr/bin/env python3
"""The g3_cpr13 arm is bball7's recipe with the CPR data and a static manikin:
exactly the documented keys differ, and every file that names the 13 sources
agrees (env cfg, eval cfg, launcher guard, retarget array, betas npz, manifest).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_cpr13_arm_cfgs.py -v
"""
import os
import re

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
ARM, BASE = "g3_cpr13_geoall__f0", "g3_bball7_geoall__f0"
SUBS = ["sub509", "sub510", "sub512", "sub513", "sub517", "sub564", "sub565", "sub566",
        "sub567", "sub569", "sub577", "sub579", "sub580"]


def _env(name):
    return yaml.safe_load(open(os.path.join(C, f"omomo_teacher_{name}.yaml")))


def test_env_cfg_is_bball7_with_cpr_data_and_a_manikin():
    b, n = _env(BASE), _env(ARM)
    diff = {k for k in set(b["env"]) | set(n["env"]) if b["env"].get(k) != n["env"].get(k)}
    assert diff == {"motion_file", "dataSub", "retargetedMotionDir", "objectMass",
                    "objectShapeProps", "plane"}
    assert b["sim"] == n["sim"]
    e = n["env"]
    assert e["subjectBodies"] == b["env"]["subjectBodies"]          # f0's 43 bodies
    assert e["dataSub"] == SUBS
    assert e["motion_file"] == "InterAct/behave_cari4d_cpr_kn"       # the KNEE-passed set, not _cf2
    assert e["retargetedMotionDir"] == "InterAct/behave_cari4d_cpr_f0_bodymajor"
    assert "objectDensity" not in e and e["objectMass"] == 3.7
    assert e["objectConvexHull"] is True
    assert "objectShapeProps" not in e                               # task defaults = OMOMO objects
    assert e["plane"] == dict(b["env"]["plane"], restitution=0.7)   # the OMOMO plane
    assert e["resetThresholds"]["object"] == 0.5                     # the "don't shove it" guard stays
    assert e["rewardTerms"]["freeFlightGate"] == {"reward": False, "resets": True}


def test_train_cfg_is_bball7_renamed():
    b = yaml.safe_load(open(os.path.join(C, "train/rlg", f"omomo_teacher_{BASE}.yaml")))
    n = yaml.safe_load(open(os.path.join(C, "train/rlg", f"omomo_teacher_{ARM}.yaml")))
    assert n["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{ARM}"
    b["params"]["config"]["full_experiment_name"] = f"smplx_teacher_{ARM}"
    assert n == b


def test_eval_cfg_mirrors_train_and_can_finish_the_longest_clip():
    tr = _env(ARM)["env"]
    ev = yaml.safe_load(open(os.path.join(C, f"omomo_eval_{ARM}.yaml")))
    assert ev["evalFor"] == [ARM]
    for k in ("motion_file", "retargetedMotionDir", "dataSub", "objectMass", "objectConvexHull",
              "plane", "rewardShape", "rewardTerms", "numObs", "obsHorizons", "resetThresholds"):
        assert ev["env"][k] == tr[k], k
    assert "objectShapeProps" not in ev["env"]
    assert ev["env"]["numEnvs"] == 2048 and ev["env"]["stateInit"] == "Start"
    assert ev["env"]["rolloutLength"] == 1000          # longest CPR clip is 691 frames (> bball7's 300)


def test_launcher_and_retarget_array_name_the_same_sources():
    src = open(os.path.join(REPO, f"slurm_teacher_{ARM}.sh")).read()
    m = re.search(r"^for s in ((?:sub\d+\s*)+); do$", src, re.M)
    assert m and m.group(1).split() == SUBS
    assert "#SBATCH --time=7-00:00:00" in src and "#SBATCH --mem=64G" in src
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{ARM}.yaml" in src
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{ARM}.yaml" in src
    assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src
    assert BASE not in "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    rt = open(os.path.join(REPO, "scripts/slurm_cari4d_cpr_retarget.sh")).read()
    m = re.search(r"^SUBJECTS=\(([\d ]+)\)$", rt, re.M)
    assert m and ["sub" + s for s in m.group(1).split()] == SUBS
    assert "behave_cari4d_cpr_kn" in rt and f"omomo_teacher_{ARM}.yaml" in rt
    assert "--source-mjcf" in rt and "sub10 sub13 sub16" in rt
    assert "behave_cari4d_cpr_f0_bodymajor" in rt


TOP7 = ["sub509", "sub512", "sub517", "sub564", "sub567", "sub569", "sub579"]


def test_cpr7_differs_from_cpr13_only_in_sources():
    """The top-7-by-clip-count arm: exactly the seven people with >= 2 clips (17 of 23)."""
    a, b = _env(ARM), _env("g3_cpr7_geoall__f0")
    diff = {k for k in set(a["env"]) | set(b["env"]) if a["env"].get(k) != b["env"].get(k)}
    assert diff == {"dataSub"} and a["sim"] == b["sim"]
    assert b["env"]["dataSub"] == TOP7 and set(TOP7) < set(SUBS)
    import csv
    rows = list(csv.DictReader(open(os.path.expanduser("~/cari4d_cpr/manifest.csv")))) \
        if os.path.exists(os.path.expanduser("~/cari4d_cpr/manifest.csv")) else None
    if rows:                                            # local manifest: the cut is clips >= 2, no tie
        n = {}
        for r in rows:
            n["sub" + r["subject_id"]] = n.get("sub" + r["subject_id"], 0) + 1
        assert sorted(s for s, c in n.items() if c >= 2) == TOP7
        assert sum(n[s] for s in TOP7) == 17
    ev = yaml.safe_load(open(os.path.join(C, "omomo_eval_g3_cpr7_geoall__f0.yaml")))
    assert ev["evalFor"] == ["g3_cpr7_geoall__f0"] and ev["env"]["dataSub"] == TOP7
    rlg = yaml.safe_load(open(os.path.join(C, "train/rlg/omomo_teacher_g3_cpr7_geoall__f0.yaml")))
    assert rlg["params"]["config"]["full_experiment_name"] == "smplx_teacher_g3_cpr7_geoall__f0"
    src = open(os.path.join(REPO, "slurm_teacher_g3_cpr7_geoall__f0.sh")).read()
    m = re.search(r"^for s in ((?:sub\d+\s*)+); do$", src, re.M)
    assert m and m.group(1).split() == TOP7
    assert "omomo_teacher_g3_cpr7_geoall__f0.yaml" in src and "#SBATCH --mem=64G" in src


def test_betas_npz_and_convert_driver_agree_with_the_sources():
    d = np.load(os.path.join(REPO, "scripts/cpr_subject_betas.npz"))
    assert sorted(d.files) == SUBS
    assert all(d[k].shape == (10,) and np.isfinite(d[k]).all() for k in d.files)
    drv = open(os.path.join(REPO, "scripts/slurm_cari4d_cpr_convert.sh")).read()
    assert 'DATASET_TAG="${DATASET_TAG:-behave_cari4d_cpr}"' in drv
    assert 'BETAS_NPZ="${BETAS_NPZ:-scripts/cpr_subject_betas.npz}"' in drv
    assert "ROTATE_AXIS=x" in drv and "REPLAY=0" in drv
    assert "relabel_contact_cpr.py" in drv
    for bad in ("relabel_contact_human.py", "relabel_contact_flags.py", "relabel_contact_soccer.py"):
        assert f"python3 scripts/{bad}" not in drv                 # never invoked (a warning echo names them)
    assert "python3 scripts/relabel_contact_cpr.py" in drv           # the knee pass IS spelled out
    assert 'sub5*_*.pt' in drv                                       # CPR ids are 509..580

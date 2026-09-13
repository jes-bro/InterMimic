#!/usr/bin/env python3
"""The two soccer teacher arms (g3_soccer15 / g3_soccer7) are the bball7 arm
with exactly the documented keys changed, and every file that names the
sources agrees (env cfg, eval cfg, launcher guard, retarget array).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_soccer_arm_cfgs.py -v
"""
import os
import re

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

ALL15 = ["sub405", "sub480", "sub482", "sub484", "sub485", "sub487", "sub488", "sub489",
         "sub490", "sub491", "sub493", "sub494", "sub495", "sub496", "sub497"]
TOP7 = ["sub405", "sub482", "sub484", "sub485", "sub487", "sub490", "sub493"]


def _env(name):
    return yaml.safe_load(open(os.path.join(C, f"omomo_teacher_{name}.yaml")))


def _launcher_sources(path):
    """The `for s in sub... ; do` data guard in a teacher launcher."""
    m = re.search(r"^for s in ((?:sub\d+\s*)+); do$", open(path).read(), re.M)
    assert m, path
    return m.group(1).split()


def test_soccer15_is_bball7_with_soccer_data_and_ball():
    base = _env("g3_bball7_geoall__f0")
    new = _env("g3_soccer15_geoall__f0")
    diff = {k for k in set(base["env"]) | set(new["env"]) if base["env"].get(k) != new["env"].get(k)}
    assert diff == {"motion_file", "dataSub", "retargetedMotionDir", "objectMass",
                    "objectShapeProps", "plane"}
    assert base["sim"] == new["sim"]
    assert new["env"]["subjectBodies"] == base["env"]["subjectBodies"]     # f0's 43 bodies
    assert new["env"]["dataSub"] == ALL15
    assert new["env"]["motion_file"] == "InterAct/behave_cari4d_soccer_cf"
    assert new["env"]["retargetedMotionDir"] == "InterAct/behave_cari4d_soccer_f0_bodymajor"
    assert "objectDensity" not in new["env"] and new["env"]["objectMass"] == 0.43
    assert new["env"]["objectConvexHull"] is True
    # restitution moved on BOTH sides of the contact pair, nothing else in those blocks
    assert new["env"]["objectShapeProps"] == {"restitution": 0.65}
    assert new["env"]["plane"] == dict(base["env"]["plane"], restitution=0.65)


def test_soccer7_differs_from_soccer15_only_in_sources():
    a = _env("g3_soccer15_geoall__f0")
    b = _env("g3_soccer7_geoall__f0")
    diff = {k for k in set(a["env"]) | set(b["env"]) if a["env"].get(k) != b["env"].get(k)}
    assert diff == {"dataSub"}
    assert a["sim"] == b["sim"]
    assert b["env"]["dataSub"] == TOP7
    assert set(TOP7) < set(ALL15)


def test_eval_cfgs_mirror_their_train_cfgs():
    for arm in ("g3_soccer15_geoall__f0", "g3_soccer7_geoall__f0"):
        tr = _env(arm)
        ev = yaml.safe_load(open(os.path.join(C, f"omomo_eval_{arm}.yaml")))
        assert ev["evalFor"] == [arm]
        for k in ("motion_file", "retargetedMotionDir", "dataSub", "objectMass", "objectConvexHull",
                  "objectShapeProps", "plane", "rewardShape", "rewardTerms", "numObs", "obsHorizons"):
            assert ev["env"][k] == tr["env"][k], (arm, k)
        assert ev["env"]["numEnvs"] == 2048 and ev["env"]["stateInit"] == "Start"
        # Eval episodes end at rolloutLength-1 and success needs the clip's last
        # frame: three soccer clips run 333/359/677 frames, so bball7's 300 would
        # cap success at 58/61 and 44/47. 1000 = every OMOMO eval cfg.
        assert ev["env"]["rolloutLength"] == 1000, arm


def test_train_cfgs_name_their_experiment():
    for arm in ("g3_soccer15_geoall__f0", "g3_soccer7_geoall__f0"):
        rlg = yaml.safe_load(open(os.path.join(C, "train/rlg", f"omomo_teacher_{arm}.yaml")))
        assert rlg["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"
        base = yaml.safe_load(open(os.path.join(C, "train/rlg/omomo_teacher_g3_bball7_geoall__f0.yaml")))
        base["params"]["config"]["full_experiment_name"] = f"smplx_teacher_{arm}"
        assert rlg == base


def test_launchers_guard_exactly_their_sources():
    assert _launcher_sources(os.path.join(REPO, "slurm_teacher_g3_soccer15_geoall__f0.sh")) == ALL15
    assert _launcher_sources(os.path.join(REPO, "slurm_teacher_g3_soccer7_geoall__f0.sh")) == TOP7
    for arm in ("g3_soccer15_geoall__f0", "g3_soccer7_geoall__f0"):
        src = open(os.path.join(REPO, f"slurm_teacher_{arm}.sh")).read()
        assert "#SBATCH --time=7-00:00:00" in src
        assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{arm}.yaml" in src
        assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{arm}.yaml" in src
        # the fixed auto-resume (sed rewrite + grep guard), not the old broken one
        assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src


def test_retarget_array_covers_all_15_sources():
    src = open(os.path.join(REPO, "scripts/slurm_cari4d_soccer_retarget.sh")).read()
    m = re.search(r"^SUBJECTS=\(([\d ]+)\)$", src, re.M)
    assert m and ["sub" + s for s in m.group(1).split()] == ALL15
    assert "omomo_teacher_g3_soccer15_geoall__f0.yaml" in src      # --targets-from the 15-arm
    assert "--source-mjcf" in src
    assert "sub10 sub13 sub16" in src                                # held-out eval bodies too
    assert "behave_cari4d_soccer_f0_bodymajor" in src                 # merge target in the header

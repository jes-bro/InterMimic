#!/usr/bin/env python3
"""The 21 component-ablation arms: 3 bases x 7 ablations, each exactly ONE
key off its base.

  bases      g3_omomo_geoall_src1__f0 (OMOMO sub1, 532 clips)
             g3_omomo_geoall_srchalf7__f0 (OMOMO 7 sources, ragged)
             g3_bball7_geoall__f0 (EgoExo4D basketball, 7 people)
  ablations  nonorm   bodyNormalizedReward false
             nopose   rewardTerms.pose.enable false
             product  rewardShape product
             nogate   rewardTerms.freeFlightGate.resets false
             realonly subjectBodies = the 13 real f0 bodies
             noret    retargetedMotionDir removed
             obs2     obsHorizons [1, 16], numObs 3198

The src1 ablations additionally turn on raggedMotionData (audited
byte-identical; lets seven copies fit at 192G instead of 384G), so their
env diff is the ablation key PLUS that flag, and nothing else.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_ablation_cfgs.py -v
"""
import os
import re

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

BASES = {
    "g3_omomo_geoall_src1__f0": {"raggedMotionData"},     # extra, semantics-free diff
    "g3_omomo_geoall_srchalf7__f0": set(),
    "g3_bball7_geoall__f0": set(),
}
ABL = ["nonorm", "nopose", "product", "nogate", "realonly", "noret", "obs2"]
REAL_F0 = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
           "sub11", "sub12", "sub14", "sub15", "sub17"]
# which leaf-path(s) each ablation may change in the env block
EXPECTED = {
    "nonorm":   {"bodyNormalizedReward"},
    "nopose":   {"rewardTerms.pose.enable"},
    "product":  {"rewardShape"},
    "nogate":   {"rewardTerms.freeFlightGate.resets"},
    "realonly": {"subjectBodies"},
    "noret":    {"retargetedMotionDir"},
    "obs2":     {"obsHorizons", "numObs"},
}
PAIRS = [(b, a) for b in BASES for a in ABL]


def _arm(base, abl):
    return base[:-len("__f0")] + f"_{abl}__f0"


def _flat(d, prefix=""):
    """{leaf path: value}; lists are leaves."""
    out = {}
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, p + "."))
        else:
            out[p] = v
    return out


def _env(name):
    return yaml.safe_load(open(os.path.join(C, f"omomo_teacher_{name}.yaml")))


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


@pytest.mark.parametrize("base,abl", PAIRS)
def test_env_cfg_is_exactly_one_key_off_its_base(base, abl):
    b, n = _env(base), _env(_arm(base, abl))
    assert _diff(b["env"], n["env"]) == EXPECTED[abl] | BASES[base], (base, abl)
    assert b["sim"] == n["sim"]
    e = n["env"]
    if abl == "nonorm":
        assert e["bodyNormalizedReward"] is False
    elif abl == "nopose":
        assert e["rewardTerms"]["pose"]["enable"] is False
        assert e["rewardTerms"]["pose"]["lambda"] == 0.02          # untouched, just inert
    elif abl == "product":
        assert e["rewardShape"] == "product"
    elif abl == "nogate":
        assert e["rewardTerms"]["freeFlightGate"] == {"resets": False, "reward": False}
    elif abl == "realonly":
        assert e["subjectBodies"] == REAL_F0
        assert all(int(s[3:]) < 100 for s in e["subjectBodies"])
    elif abl == "noret":
        assert "retargetedMotionDir" not in e
        assert e["cpuMotionData"] is True                           # still streams from host
    elif abl == "obs2":
        assert e["obsHorizons"] == [1, 16] and e["numObs"] == 2 * 1599
    if "raggedMotionData" in BASES[base]:
        assert e["raggedMotionData"] is True


@pytest.mark.parametrize("base,abl", PAIRS)
def test_train_cfg_is_the_base_renamed(base, abl):
    arm = _arm(base, abl)
    b = yaml.safe_load(open(os.path.join(C, "train/rlg", f"omomo_teacher_{base}.yaml")))
    n = yaml.safe_load(open(os.path.join(C, "train/rlg", f"omomo_teacher_{arm}.yaml")))
    assert n["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"
    b["params"]["config"]["full_experiment_name"] = f"smplx_teacher_{arm}"
    assert n == b


@pytest.mark.parametrize("base,abl", PAIRS)
def test_eval_cfg_mirrors_the_ablated_key(base, abl):
    arm = _arm(base, abl)
    tr = _env(arm)["env"]
    ev = yaml.safe_load(open(os.path.join(C, f"omomo_eval_{arm}.yaml")))
    assert ev["evalFor"] == [arm]
    e = ev["env"]
    # the ablated key is mirrored (subjectBodies is eval-owned: patched per pair)
    for k in EXPECTED[abl] - {"subjectBodies"}:
        path = k.split(".")
        tv, evv = tr, e
        for p in path:
            tv = tv.get(p, "<absent>") if isinstance(tv, dict) else "<absent>"
            evv = evv.get(p, "<absent>") if isinstance(evv, dict) else "<absent>"
        assert tv == evv, (arm, k)
    for k in ("motion_file", "rewardShape", "bodyNormalizedReward", "numObs", "obsHorizons",
              "rewardTerms", "resetThresholds", "cpuMotionData"):
        assert e[k] == tr[k], (arm, k)
    assert e.get("retargetedMotionDir") == tr.get("retargetedMotionDir"), arm
    assert e.get("raggedMotionData") == tr.get("raggedMotionData"), arm
    # scoring budget and rollout window are the BASE eval's (bball7 evals use 300:
    # its longest clip is 189 frames; the OMOMO evals use 1000), never re-chosen here
    bev = yaml.safe_load(open(os.path.join(C, f"omomo_eval_{base}.yaml")))["env"]
    assert e["numEnvs"] == bev["numEnvs"] == 2048 and e["stateInit"] == "Start"
    assert e["rolloutLength"] == bev["rolloutLength"]


@pytest.mark.parametrize("base,abl", PAIRS)
def test_launcher_points_at_its_own_files(base, abl):
    arm = _arm(base, abl)
    src = open(os.path.join(REPO, f"slurm_teacher_{arm}.sh")).read()
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#") or l.startswith("#SBATCH"))
    assert "#SBATCH --time=7-00:00:00" in src
    assert f'--job-name="tch-{arm}"' in src
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{arm}.yaml" in src
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{arm}.yaml" in src
    assert f"checkpoints/smplx_teacher_{arm}/nn/" in src
    assert f"ABLATION {abl} of {base}" in src
    # the base's name must not survive anywhere it would steer the job (cfg paths,
    # checkpoint dir, job name) -- only inside the ABLATION provenance line
    stray = [l for l in code.split("\n") if base in l and "ABLATION" not in l]
    assert not stray, (arm, stray)
    # the fixed auto-resume (sed rewrite + grep guard)
    assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src
    if abl == "noret":
        # no retarget tree exists for this arm: the guards that demand one are gone
        assert "retargetedMotionDir:'" not in code and 'RT=$(' not in code
    else:
        assert "retargetedMotionDir:'" in code
    if "raggedMotionData" in BASES[base]:
        assert "raggedMotionData:\\s*[Tt]rue" in code
        assert "#SBATCH --mem=192G" in src
    else:
        m = re.search(r"^#SBATCH --mem=(\S+)$", open(os.path.join(REPO, f"slurm_teacher_{base}.sh")).read(), re.M)
        assert f"#SBATCH --mem={m.group(1)}" in src          # same memory as the base

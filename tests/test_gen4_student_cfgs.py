#!/usr/bin/env python3
"""The gen4 object-group student (scripts/make_gen4_student_cfgs.py): the g3 OMOMO
XF student with exactly the documented edits; its env mirrors the gen4 msexp
teachers on every shared key; its launcher runs InterMimicDistillG4 with the
object-keyed teacher guard, the gen4 guards and the motion-scale guard.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_student_cfgs.py -q
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"


def _gen():
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_student_cfgs.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    return mk


mk = _gen()
BASE, ARM, TEACHER_ARM = mk.BASE, mk.ARM, mk.TEACHER_ARM


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


def _env(name): return yaml.safe_load(open(CFG / f"omomo_{name}.yaml"))
def _train(name): return yaml.safe_load(open(CFG / "train/rlg" / f"omomo_{name}.yaml"))


def test_env_is_the_g3_student_plus_exactly_the_documented_edits():
    base, new = _env(BASE), _env(ARM)
    assert differing_keys(base, new) == {
        "env.subjectBodies", "env.subjectHeightsFile", "env.humanoidAssetSuffix",
        "env.rewardTerms.pose.enable", "env.motionScaleReward.enable", "env.motionScaleReward.referenceStd",
        "env.teacherPolicy", "env.teacherPolicyCFG"}
    e = new["env"]
    assert e["subjectHeightsFile"] == "scripts/synthetic_heights_gen4.json"
    assert e["humanoidAssetSuffix"] == "_inertial"
    assert e["rewardTerms"]["pose"]["enable"] is False and e["rewardTerms"]["pose"]["lambda"] == 0.02
    assert e["motionScaleReward"] == {"enable": True, "referenceStd": mk.MS_REFERENCE_STD}
    assert e["teacherPolicy"] == mk.TEACHER_SET
    assert e["teacherPolicyCFG"] == f"intermimic/data/cfg/train/rlg/omomo_teacher_{TEACHER_ARM}.yaml"
    # unchanged on purpose
    assert "dataObjects" not in e                                   # all 13 objects
    assert e["dataSub"] == base["env"]["dataSub"] and len(e["dataSub"]) == 13
    assert e["retargetedMotionDir"] == "InterAct/OMOMO_retarget_contact_srcall13"
    assert e["raggedMotionData"] is True and e["cpuMotionData"] is True
    assert e["studentObsHorizons"] == [1, 4, 7, 10, 13, 16] and e["numObsRetarget"] == 9594


def test_env_mirrors_the_gen4_teachers_on_every_shared_key():
    """Student env == the chairs msexp teacher env except the student's own keys and the
    teacher's object filter: same bodies, heights, MJCFs, reward, tree, storage."""
    teacher, new = _env(f"teacher_{TEACHER_ARM}"), _env(ARM)
    assert differing_keys(teacher, new) == {
        "env.dataObjects",                                           # teacher: chairs only; student: all
        "env.teacherPolicy", "env.teacherPolicyCFG", "env.studentObsHorizons", "env.numObsRetarget"}
    bodies = new["env"]["subjectBodies"]
    assert bodies == teacher["env"]["subjectBodies"] and len(bodies) == 43
    assert [b for b in bodies if int(b[3:]) >= 600] == [f"sub{i}" for i in range(600, 630)]
    assert not (set(mk.HELDOUT) & set(bodies))


def test_teacher_policy_cfg_is_the_fleets_transformer():
    t = _train(f"teacher_{TEACHER_ARM}")["params"]
    assert t["network"]["name"] == "intermimic_transformer" and t["network"]["transformer"]["num_tokens"] == 6
    # the same network for all four group teachers: the student may load any of them through this cfg
    for g in mk.GROUPS:
        tg = _train(f"teacher_gen4_omomo_geoall_{g}_xf_nvadlr_nopose_msexp__f0")["params"]
        assert tg["network"] == t["network"], g


def test_train_cfg_name_only():
    base, new = _train(BASE), _train(ARM)
    assert differing_keys(base, new) == {"params.config.full_experiment_name"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_{ARM}"
    assert new["params"]["network"]["name"] == "intermimic_transformer"
    assert new["params"]["network"]["transformer"]["num_tokens"] == 6
    assert new["params"]["config"]["normalize_value"] is True and new["params"]["config"]["lr_schedule"] == "adaptive"


def test_launcher():
    txt = (REPO / f"slurm_{ARM}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert re.search(r'^#SBATCH --job-name="stu-' + re.escape(mk.SHORT_ARM), txt, flags=re.M)
    assert f"student-{mk.SHORT_ARM}-%j.out" in txt and f"smplx_{ARM}/nn/" in code
    assert "#SBATCH --mem=480G\n" in txt
    assert f"omomo_{ARM}.yaml" in code and mk.SHORT_BASE not in code
    assert "--task InterMimicDistillG4 \\" in code and "InterMimicDistillG3" not in code
    assert "for b in sub4 sub10 sub13 sub16; do" in code                       # fold guard holds out sub4
    assert "e['objects']" in code and "'sources' in e" in code                 # object-keyed teacher guard
    assert "'woodchair'" in code and "'tripod'" in code and "missing objects" in code
    assert "need={1,2,3,5,6,7,8,9,11,12,14,15,17}" not in code
    assert "humanoidAssetSuffix:\\s*_inertial" in code and "_inertial.xml" in code     # gen4 guards
    assert f"referenceStd:\\s*{mk.MS_REFERENCE_STD}" in code                           # motion-scale guard
    assert "raggedMotionData" in code and "--gen4-objects" in txt
    assert code.index("_inertial.xml") < code.index("python -u -m intermimic.run_distill")
    # bash syntax
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_{ARM}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_generator_is_idempotent(tmp_path):
    before = {p: (REPO / p).read_text() for p in (f"isaacgym/src/intermimic/data/cfg/omomo_{ARM}.yaml",
                                                   f"isaacgym/src/intermimic/data/cfg/train/rlg/omomo_{ARM}.yaml",
                                                   f"slurm_{ARM}.sh")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_student_cfgs.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert (REPO / p).read_text() == txt, p

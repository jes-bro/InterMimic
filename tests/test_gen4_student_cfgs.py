#!/usr/bin/env python3
"""The gen4 object-group student (scripts/make_gen4_student_cfgs.py): the g3 OMOMO
XF student with exactly the documented edits; its env mirrors the gen4 msexp
teachers on every shared key; its launcher runs InterMimicDistillG4 with the
object-keyed teacher guard, the gen4 guards and the motion-scale guard.

Both arms are pinned: the 50-frame-window student and its --fullep sibling, which
must differ from it in exactly rolloutLength 1000 + minEpisodeFrames 30 (the
_fullep teacher arms' two keys) plus the launcher's full-episode guard.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_student_cfgs.py -q
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"


def _gen():
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_student_cfgs.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    return mk


mk = _gen()
BASE, ARM, TEACHER_ARM = mk.BASE, mk.ARM, mk.TEACHER_ARM
FULLEP_KEYS = {"env.rolloutLength", "env.minEpisodeFrames"}
BOTH = pytest.mark.parametrize("fullep", [False, True], ids=["win50", "fullep"])


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


def test_arm_names():
    assert mk.arm(False) == ARM == "student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0"
    assert mk.arm(True) == "student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0"
    assert mk.short(ARM) == mk.SHORT_ARM == "gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0"
    assert (mk.FULLEP_ROLLOUT, mk.FULLEP_MIN_FRAMES) == (1000, 30)       # the _fullep teachers' numbers


@BOTH
def test_env_is_the_g3_student_plus_exactly_the_documented_edits(fullep):
    base, new = _env(BASE), _env(mk.arm(fullep))
    assert differing_keys(base, new) == {
        "env.subjectBodies", "env.subjectHeightsFile", "env.humanoidAssetSuffix",
        "env.rewardTerms.pose.enable", "env.motionScaleReward.enable", "env.motionScaleReward.referenceStd",
        "env.teacherPolicy", "env.teacherPolicyCFG"} | (FULLEP_KEYS if fullep else set())
    e = new["env"]
    assert e["subjectHeightsFile"] == "scripts/synthetic_heights_gen4.json"
    assert e["humanoidAssetSuffix"] == "_inertial"
    assert e["rewardTerms"]["pose"]["enable"] is False and e["rewardTerms"]["pose"]["lambda"] == 0.02
    assert e["motionScaleReward"] == {"enable": True, "referenceStd": mk.MS_REFERENCE_STD}
    assert e["teacherPolicy"] == mk.TEACHER_SET
    assert e["teacherPolicyCFG"] == f"intermimic/data/cfg/train/rlg/omomo_teacher_{TEACHER_ARM}.yaml"
    if fullep:
        assert e["rolloutLength"] == 1000 and e["minEpisodeFrames"] == 30
    else:
        assert e["rolloutLength"] == 50 and "minEpisodeFrames" not in e
    # unchanged on purpose
    assert "dataObjects" not in e                                   # all 13 objects
    assert e["dataSub"] == base["env"]["dataSub"] and len(e["dataSub"]) == 13
    assert e["retargetedMotionDir"] == "InterAct/OMOMO_retarget_contact_srcall13"
    assert e["raggedMotionData"] is True and e["cpuMotionData"] is True
    assert e["studentObsHorizons"] == [1, 4, 7, 10, 13, 16] and e["numObsRetarget"] == 9594
    assert e["numEnvs"] == 4096 and e["physicalBufferSize"] == 3          # launcher passes 2048


def test_fullep_env_is_the_win50_env_plus_exactly_the_two_keys():
    """The one read of the arm: everything but the episode rule is shared, so a
    matched-epoch comparison against job 17894921's checkpoints isolates it."""
    assert differing_keys(_env(mk.arm(False)), _env(mk.arm(True))) == FULLEP_KEYS
    # and those two keys are what the _fullep TEACHER arms carry
    t50, tfe = _env(f"teacher_{TEACHER_ARM}"), _env(f"teacher_{TEACHER_ARM.replace('__f0', '_fullep__f0')}")
    assert differing_keys(t50, tfe) == FULLEP_KEYS
    assert _env(mk.arm(True))["env"]["minEpisodeFrames"] == tfe["env"]["minEpisodeFrames"]
    assert _env(mk.arm(True))["env"]["rolloutLength"] == tfe["env"]["rolloutLength"]


@BOTH
def test_env_mirrors_the_gen4_teachers_on_every_shared_key(fullep):
    """Student env == the chairs msexp teacher env except the student's own keys and the
    teacher's object filter: same bodies, heights, MJCFs, reward, tree, storage."""
    teacher, new = _env(f"teacher_{TEACHER_ARM}"), _env(mk.arm(fullep))
    assert differing_keys(teacher, new) == {
        "env.dataObjects",                                           # teacher: chairs only; student: all
        "env.teacherPolicy", "env.teacherPolicyCFG", "env.studentObsHorizons", "env.numObsRetarget"} | (FULLEP_KEYS if fullep else set())
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


@BOTH
def test_train_cfg_name_only(fullep):
    base, new = _train(BASE), _train(mk.arm(fullep))
    assert differing_keys(base, new) == {"params.config.full_experiment_name"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_{mk.arm(fullep)}"
    assert new["params"]["network"]["name"] == "intermimic_transformer"
    assert new["params"]["network"]["transformer"]["num_tokens"] == 6
    assert new["params"]["config"]["normalize_value"] is True and new["params"]["config"]["lr_schedule"] == "adaptive"


@BOTH
def test_launcher(fullep):
    a, short_arm = mk.arm(fullep), mk.short(mk.arm(fullep))
    txt = (REPO / f"slurm_{a}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert re.search(r'^#SBATCH --job-name="stu-' + re.escape(short_arm) + '"', txt, flags=re.M)
    assert f"student-{short_arm}-%j.out" in txt and f"smplx_{a}/nn/" in code
    assert "#SBATCH --mem=480G\n" in txt
    assert f"omomo_{a}.yaml" in code and mk.SHORT_BASE not in code
    if fullep:
        assert mk.SHORT_ARM + "." not in code and mk.SHORT_ARM + "/" not in code   # no 50-frame path survives
    assert "--task InterMimicDistillG4 \\" in code and "InterMimicDistillG3" not in code
    assert "for b in sub4 sub10 sub13 sub16; do" in code                       # fold guard holds out sub4
    assert "e['objects']" in code and "'sources' in e" in code                 # object-keyed teacher guard
    assert "'woodchair'" in code and "'tripod'" in code and "missing objects" in code
    assert "need={1,2,3,5,6,7,8,9,11,12,14,15,17}" not in code
    assert "humanoidAssetSuffix:\\s*_inertial" in code and "_inertial.xml" in code     # gen4 guards
    assert f"referenceStd:\\s*{mk.MS_REFERENCE_STD}" in code                           # motion-scale guard
    assert "raggedMotionData" in code and "--gen4-objects" in txt
    assert code.index("_inertial.xml") < code.index("python -u -m intermimic.run_distill")
    # full-episode guard: present (and before the invocation) on the fullep arm only
    guard = "rolloutLength:\\s*1000\\s*$' \"$CFG_ENV\" || ! grep -qE '^\\s*minEpisodeFrames:\\s*30"
    if fullep:
        assert guard in code and code.index(guard) < code.index("python -u -m intermimic.run_distill")
        assert "FULL EPISODES" in code                                          # the recipe echo says so
    else:
        assert guard not in code and "minEpisodeFrames" not in code and "FULL EPISODES" not in code
    # bash syntax
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_{a}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_launchers_differ_only_in_name_header_and_the_fullep_guard():
    w, f = [(REPO / f"slurm_{mk.arm(x)}.sh").read_text() for x in (False, True)]
    f_renamed = f.replace(mk.short(mk.arm(True)), mk.SHORT_ARM).replace(mk.arm(True), ARM)
    w_code = [l for l in w.splitlines() if not l.lstrip().startswith("#")]
    f_code = [l for l in f_renamed.splitlines() if not l.lstrip().startswith("#")]
    only_fullep = [l for l in f_code if l not in w_code]
    only_win50 = [l for l in w_code if l not in f_code]
    # fullep adds the guard (its if/echo/fi lines) and a longer recipe echo; nothing else
    assert all(("rolloutLength" in l or "minEpisodeFrames" in l or l.strip() == "fi" or "FULL EPISODES" in l)
               for l in only_fullep), only_fullep
    assert all("GEN4 OBJECT-GROUP student" in l for l in only_win50), only_win50   # only the recipe echo differs


@BOTH
def test_generator_is_idempotent(fullep, tmp_path):
    a = mk.arm(fullep)
    before = {p: (REPO / p).read_text() for p in (f"isaacgym/src/intermimic/data/cfg/omomo_{a}.yaml",
                                                   f"isaacgym/src/intermimic/data/cfg/train/rlg/omomo_{a}.yaml",
                                                   f"slurm_{a}.sh")}
    args = [sys.executable, str(REPO / "scripts/make_gen4_student_cfgs.py")] + (["--fullep"] if fullep else [])
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert (REPO / p).read_text() == txt, p

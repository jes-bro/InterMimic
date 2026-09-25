"""The two `nopose` student arms (2026-09-25): rewardTerms.pose.enable false
(no relative joint-angle reward factor), each hand-written from a tfinal base.

  omomo_xf_ret_nvadlr_tfinal_nopose  ONE key off omomo_xf_ret_nvadlr_tfinal;
                                     same teacher dir (g3_omomo_tfinal), same data
  act_nocpr_xf_ret_nvadlr_nopose     a NEW source set (bball7 + soccer15 = the act
                                     roster minus cpr13) on its own merged data,
                                     props file and teacher dir, plus the key

These are METHOD CANDIDATES, not ablations: the teacher nopose arm beat the
with-pose teacher, so the pose term is being dropped from the method. The read
of the OMOMO arm is the clean one; the act_nocpr arm has no with-pose twin, so
"no cpr" and "no pose" are confounded there (pinned in its header).
Both keep plane.restitution 0.7 and the tfinal sim block, so the physics is
the base's.

What this pins (the same classes test_ablation_cfgs.py covers for teachers):
a second env key, a train cfg that is not the base renamed, a launcher pointing
at the base's cfgs or teacher dir, a wrong --mem, a data-prep command whose
paths do not match the cfg, a stray "cpr" in the act_nocpr code lines, and any
changed launcher code line outside the enumerated classes.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_nopose_students.py -q
"""
import difflib
import glob
import os
import re

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")

OMOMO_BASE, OMOMO_ARM = "omomo_xf_ret_nvadlr_tfinal", "omomo_xf_ret_nvadlr_tfinal_nopose"
ACT_BASE, ACT_ARM = "act_xf_ret_nvadlr_tfinal", "act_nocpr_xf_ret_nvadlr_nopose"
ACT_NOCPR_KEYS = {"env.dataSub", "env.motion_file", "env.retargetedMotionDir",
                  "env.objectPropsFile", "env.teacherPolicy", "env.rewardTerms.pose.enable"}


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _env(n):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_{n}__f0.yaml")))


def _train(n):
    return yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_{n}__f0.yaml")))


def _teacher_subs(arm):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_teacher_g3_{arm}_geoall__f0.yaml")))["env"]["dataSub"]


def _launcher(n):
    return open(os.path.join(ROOT, f"slurm_student_g3_{n}__f0.sh")).read()


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _code_lines(text):
    """Launcher lines that do something: drop blank and comment-only lines, keep #SBATCH."""
    return [l for l in text.split("\n")
            if l.strip() and (not l.lstrip().startswith("#") or l.startswith("#SBATCH"))]


# ----------------------------------------------------------------------------- env cfgs
def test_omomo_env_is_exactly_the_pose_key_off_its_base():
    b, n = _env(OMOMO_BASE), _env(OMOMO_ARM)
    assert _diff(b, n) == {"env.rewardTerms.pose.enable"}
    assert n["env"]["rewardTerms"]["pose"] == {"enable": False, "lambda": 0.02}   # lambda untouched, inert
    assert b["env"]["rewardTerms"]["pose"]["enable"] is True
    assert n["sim"] == b["sim"]
    assert n["env"]["teacherPolicy"] == "checkpoints/teachers/g3_omomo_tfinal"    # SAME teachers as the base


def test_act_nocpr_env_is_the_act_tfinal_env_plus_its_own_data_and_the_pose_key():
    b, n = _env(ACT_BASE), _env(ACT_ARM)
    assert _diff(b, n) == ACT_NOCPR_KEYS
    e = n["env"]
    assert e["rewardTerms"]["pose"] == {"enable": False, "lambda": 0.02}
    assert e["motion_file"] == "InterAct/behave_cari4d_act_nocpr"
    assert e["retargetedMotionDir"] == "InterAct/behave_cari4d_act_nocpr_f0_bodymajor"
    assert e["objectPropsFile"] == "isaacgym/src/intermimic/data/cfg/object_props_g3_act_nocpr.yaml"
    assert e["teacherPolicy"] == "checkpoints/teachers/g3_act_nocpr"
    # physics the act student's: the plane the per-object restitutions were solved against
    assert e["plane"]["restitution"] == 0.7 == b["env"]["plane"]["restitution"]
    assert n["sim"] == b["sim"]


def test_act_nocpr_roster_is_bball7_plus_soccer15_and_the_act_roster_minus_cpr13():
    got = _env(ACT_ARM)["env"]["dataSub"]
    bball, soccer, cpr = _teacher_subs("bball7"), _teacher_subs("soccer15"), _teacher_subs("cpr13")
    assert set(bball).isdisjoint(soccer)
    assert got == sorted(set(bball) | set(soccer), key=lambda s: int(s[3:]))
    assert len(got) == 22 and len(bball) == 7 and len(soccer) == 15
    act = _env(ACT_BASE)["env"]["dataSub"]
    assert got == [s for s in act if s not in set(cpr)], "act minus cpr13 must be exactly this roster"
    assert not (set(got) & set(cpr))


# --------------------------------------------------------------------------- train cfgs
@pytest.mark.parametrize("base,arm", [(OMOMO_BASE, OMOMO_ARM), (ACT_BASE, ACT_ARM)])
def test_train_cfg_is_the_base_renamed(base, arm):
    b, n = _train(base), _train(arm)
    assert n["params"]["config"]["full_experiment_name"] == f"smplx_student_g3_{arm}__f0"
    b["params"]["config"]["full_experiment_name"] = f"smplx_student_g3_{arm}__f0"
    assert n == b


def test_experiment_names_are_new_and_distinct():
    names = {}
    for p in glob.glob(os.path.join(RLG, "omomo_student_g3_*__f0.yaml")):
        names.setdefault(yaml.safe_load(open(p))["params"]["config"]["full_experiment_name"], []).append(os.path.basename(p))
    dup = {k: v for k, v in names.items() if len(v) > 1}
    assert not dup, dup
    assert f"smplx_student_g3_{OMOMO_ARM}__f0" in names and f"smplx_student_g3_{ACT_ARM}__f0" in names


# ---------------------------------------------------------------------------- launchers
_ALLOWED_ANY = [
    r'^#SBATCH --job-name="stu-',
    r"^#SBATCH --output=student-",
    r"^CFG_(ENV|TRAIN)=isaacgym/src/intermimic/data/cfg/",
    r'^echo "\[student\] (OMOMO|ACTIVITY|ACTIVITY-NOCPR) XF/RET/NVADLR student:',
    r'^echo "\[student\] (host=|METHOD CANDIDATE )',
]


@pytest.mark.parametrize("base,arm,mem,teacher_dir", [
    (OMOMO_BASE, OMOMO_ARM, "480G", "checkpoints/teachers/g3_omomo_tfinal"),
    (ACT_BASE, ACT_ARM, "64G", "checkpoints/teachers/g3_act_nocpr"),
])
def test_launcher_points_at_its_own_files_and_differs_only_in_the_allowed_classes(base, arm, mem, teacher_dir):
    bsrc, src = _launcher(base), _launcher(arm)
    code = _code_lines(src)
    assert "#SBATCH --time=7-00:00:00" in src
    assert f'--job-name="stu-g3_{arm}__f0"' in src
    assert f"#SBATCH --output=student-g3_{arm}__f0-%j.out" in src
    assert f"#SBATCH --mem={mem}" in src
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{arm}__f0.yaml" in src
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{arm}__f0.yaml" in src
    assert f"checkpoints/smplx_student_g3_{arm}__f0/nn/" in src
    assert "METHOD CANDIDATE nopose" in src and "rewardTerms.pose.enable true -> false" in src
    assert "ABLATION" not in src, "nopose is a method candidate, not an ablation (Jess 2026-09-25)"
    assert f"--out {teacher_dir}" in src, "the header's collect step must write the cfg's teacherPolicy dir"
    assert _env(arm)["env"]["teacherPolicy"] == teacher_dir
    # no code line still names the base experiment (the METHOD CANDIDATE echo may)
    stray = [l for l in code if f"{base}__f0" in l and "METHOD CANDIDATE" not in l]
    assert not stray, stray
    # invariants a one-key arm never touches
    assert 'NUM_ENVS="${NUM_ENVS:-2048}"' in code
    assert any("default_buffer_size_multiplier" in l for l in code)
    assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src
    assert "-m intermimic.run_distill" in src and "--task InterMimicDistillG3" in src
    assert "num_tokens" in src, "XF launcher must carry the token guard"
    # every changed code line falls in an allowed class (after renaming base -> arm)
    bcode = [l.replace(base, arm) for l in _code_lines(bsrc)]
    changed = [l[2:] for l in difflib.ndiff(bcode, code) if l[:2] in ("- ", "+ ")]
    bad = [l for l in changed if not any(re.search(p, l) for p in _ALLOWED_ANY)]
    assert not bad, (arm, bad)


def test_act_nocpr_launcher_data_prep_matches_its_cfg_and_never_mentions_cpr():
    src = _launcher(ACT_ARM)
    e = _env(ACT_ARM)["env"]
    # the merge command builds exactly the dirs / props file the cfg reads, from its own body roster
    assert "merge_activity_data.py --arms bball7 soccer15 \\" in src
    assert f"--out-motion {e['motion_file']}" in src
    assert f"--out-retarget {e['retargetedMotionDir']}" in src
    assert f"--props-out {e['objectPropsFile']}" in src
    assert f"--bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_{ACT_ARM}__f0.yaml" in src
    assert f"--student-plane-restitution {e['plane']['restitution']}" in src
    assert "collect_g3_teachers.py --activities bball7 soccer15 \\" in src
    # the guards still read every path from the cfg (no hard-coded act paths in code lines)
    code = "\n".join(_code_lines(src))
    assert re.search(r"(?<!no)cpr", code) is None, "a cpr arm/dir leaked into the code lines"   # 'nocpr' is the arm name
    assert re.search(r"behave_cari4d_act(?!_nocpr)", code) is None and "object_props_g3_act.yaml" not in code
    assert "MD=$(grep -oE '^\\s*motion_file:" in src and "OP=$(grep -oE '^\\s*objectPropsFile:" in src

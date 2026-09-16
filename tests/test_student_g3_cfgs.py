"""Pin every g3 student cfg pair against its base TEACHER cfg: the env is the
base's data verbatim plus exactly the keys the table says; MLP/XF twins share
the env body byte-for-byte; train cfgs carry the intended network/optimizer
with distinct experiment names; launchers point at their own cfgs and task.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_student_g3_cfgs.py -q
"""
import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")

H6 = [1, 4, 7, 10, 13, 16]
TCFG = "intermimic/data/cfg/train/rlg/omomo_teacher_g3_omomo_geoall__f0.yaml"


def _distill(tp):
    return {"env.teacherPolicy": tp, "env.teacherPolicyCFG": TCFG,
            "env.studentObsHorizons": H6, "env.numObsRetarget": 9594}


ACT_MERGE = lambda tag: {"env.raggedMotionData": True,
                         "env.objectPropsFile": f"isaacgym/src/intermimic/data/cfg/object_props_g3_{tag}.yaml"}

# arm -> (base teacher cfg stem, added keys, changed keys, removed keys, sbatch mem)
TABLE = {
    "omomo":        ("omomo_teacher_g3_omomo_geoall_srcall13__f0", _distill("checkpoints/teachers/g3_omomo"), {}, set(), "480G"),
    "omomo_halves": ("omomo_teacher_g3_omomo_geoall_srcall13__f0", _distill("checkpoints/teachers/g3_omomo_halves"), {}, set(), "480G"),
    "omomo_all13t": ("omomo_teacher_g3_omomo_geoall_srcall13__f0", _distill("checkpoints/teachers/g3_omomo_all13t"), {}, set(), "480G"),
    "omomo7":       ("omomo_teacher_g3_omomo_geoall_srchalf7__f0", _distill("checkpoints/teachers/g3_omomo7"), {}, set(), "320G"),
    "src1":         ("omomo_teacher_g3_omomo_geoall_src1__f0",
                     {**_distill("checkpoints/teachers/g3_src1"), "env.raggedMotionData": True}, {}, set(), "224G"),
    "bball7":       ("omomo_teacher_g3_bball7_geoall__f0", _distill("checkpoints/teachers/g3_bball7"), {}, set(), "64G"),
    "act":          ("omomo_teacher_g3_bball7_geoall__f0", {**_distill("checkpoints/teachers/g3_act"), **ACT_MERGE("act")},
                     {"env.motion_file": "InterAct/behave_cari4d_act",
                      "env.retargetedMotionDir": "InterAct/behave_cari4d_act_f0_bodymajor",
                      "env.plane.restitution": 0.7, "env.dataSub": ("union", "bball7", "soccer15", "cpr13")},
                     {"env.objectMass", "env.objectShapeProps.restitution"}, "64G"),
    "act7":         ("omomo_teacher_g3_bball7_geoall__f0", {**_distill("checkpoints/teachers/g3_act7"), **ACT_MERGE("act7")},
                     {"env.motion_file": "InterAct/behave_cari4d_act7",
                      "env.retargetedMotionDir": "InterAct/behave_cari4d_act7_f0_bodymajor",
                      "env.plane.restitution": 0.7, "env.dataSub": ("union", "bball7", "soccer7", "cpr7")},
                     {"env.objectMass", "env.objectShapeProps.restitution"}, "64G"),
}
STUDENTS = ["mlp_ret_stock", "xf_ret_nvadlr"]
ALL = [(a, s) for a in TABLE for s in STUDENTS]


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _env(arm, st):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_{arm}_{st}__f0.yaml")))


def _train(arm, st):
    return yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_{arm}_{st}__f0.yaml")))


def _teacher(stem):
    return yaml.safe_load(open(os.path.join(CFG, f"{stem}.yaml")))


def _body(path):
    lines = open(path).read().splitlines()
    return "\n".join(lines[next(i for i, l in enumerate(lines) if l.startswith("env:")):])


def _union(*arms):
    subs = {s for a in arms for s in _teacher(f"omomo_teacher_g3_{a}_geoall__f0")["env"]["dataSub"]}
    return sorted(subs, key=lambda s: int(s[3:]))


@pytest.mark.parametrize("arm,st", ALL)
def test_env_is_base_teacher_plus_table(arm, st):
    stem, added, changed, removed, _ = TABLE[arm]
    base = _flat({"env": _teacher(stem)["env"]})
    stu = _flat({"env": _env(arm, st)["env"]})
    assert _env(arm, st)["sim"] == _teacher(stem)["sim"]
    assert {k: v for k, v in stu.items() if k not in base} == added
    assert {k for k in base if k not in stu} == removed
    exp_changed = {k: (_union(*v[1:]) if isinstance(v, tuple) else v) for k, v in changed.items()}
    got_changed = {k: stu[k] for k in base if k in stu and stu[k] != base[k]}
    assert got_changed == exp_changed


@pytest.mark.parametrize("arm", list(TABLE))
def test_twins_share_env_body(arm):
    assert _body(os.path.join(CFG, f"omomo_student_g3_{arm}_mlp_ret_stock__f0.yaml")) == \
           _body(os.path.join(CFG, f"omomo_student_g3_{arm}_xf_ret_nvadlr__f0.yaml"))


@pytest.mark.parametrize("arm,st", ALL)
def test_student_width_rule(arm, st):
    e = _env(arm, st)["env"]
    assert e["numObs"] % len(e["obsHorizons"]) == 0
    assert e["numObsRetarget"] == (e["numObs"] // len(e["obsHorizons"])) * len(e["studentObsHorizons"])


@pytest.mark.parametrize("arm", list(TABLE))
def test_train_cfgs(arm):
    m = _train(arm, "mlp_ret_stock")["params"]
    x = _train(arm, "xf_ret_nvadlr")["params"]
    assert m["network"]["name"] == "intermimic" and m["network"]["mlp"]["units"] == [1024, 1024, 512]
    assert m["config"]["lr_schedule"] == "constant" and m["config"]["normalize_value"] is False
    assert "kl_threshold" not in m["config"]
    assert x["network"]["name"] == "intermimic_transformer"
    assert x["network"]["transformer"] == {"num_tokens": 6, "readout_token": 0}
    assert x["network"]["mlp"]["units"] == [1024, 1024, 512]
    assert x["config"]["lr_schedule"] == "adaptive" and x["config"]["kl_threshold"] == 0.06
    assert x["config"]["normalize_value"] is True
    for t, st in ((m, "mlp_ret_stock"), (x, "xf_ret_nvadlr")):
        assert t["config"]["expert_loss_coef"] == 1 and t["config"]["save_intermediate"] is True
        assert t["config"]["full_experiment_name"] == f"smplx_student_g3_{arm}_{st}__f0"
    diff = {k for k in set(_flat(m)) | set(_flat(x)) if _flat(m).get(k) != _flat(x).get(k)}
    assert diff == {"network.name", "network.transformer.num_tokens", "network.transformer.readout_token",
                    "config.lr_schedule", "config.kl_threshold", "config.normalize_value",
                    "config.full_experiment_name"}


def test_xf_readout_is_delta_t_one():
    for arm, st in ALL:
        if st == "xf_ret_nvadlr":
            e = _env(arm, st)["env"]
            t = _train(arm, st)["params"]["network"]["transformer"]
            assert t["num_tokens"] == len(e["studentObsHorizons"])
            assert e["studentObsHorizons"][t["readout_token"]] == 1


def test_all_experiment_names_distinct():
    names = [_train(a, s)["params"]["config"]["full_experiment_name"] for a, s in ALL]
    assert len(set(names)) == len(ALL)


@pytest.mark.parametrize("arm,st", ALL)
def test_launcher(arm, st):
    stem, added, changed, removed, mem = TABLE[arm]
    path = os.path.join(ROOT, f"slurm_student_g3_{arm}_{st}__f0.sh")
    assert os.path.isfile(path), f"missing launcher {path}"
    s = open(path).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{arm}_{st}__f0.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{arm}_{st}__f0.yaml" in s
    assert "-m intermimic.run_distill" in s and "--task InterMimicDistillG3" in s
    assert "teachers.yaml" in s and "resume_from" in s
    assert f'--job-name="stu-g3_{arm}_{st}__f0"' in s
    assert f"#SBATCH --mem={mem}" in s
    tp = added["env.teacherPolicy"]
    assert f"--out {tp}" in s, "collect command in the header must write the cfg's teacherPolicy dir"
    if "env.objectPropsFile" in added:
        assert "merge_activity_data.py" in s and "objectPropsFile" in s
    if st == "xf_ret_nvadlr":
        assert "num_tokens" in s, "XF launcher must carry the token guard"
    ragged = added.get("env.raggedMotionData") or _teacher(stem)["env"].get("raggedMotionData")
    assert ("raggedMotionData" in s) == bool(ragged)

"""soccer8 / cpr6 teacher arms (the other halves of soccer15 / cpr13) and the
act_halves student pair.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_act_halves.py -q
"""
import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _teacher(name):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_teacher_g3_{name}_geoall__f0.yaml")))


def _body(path):
    lines = open(path).read().splitlines()
    return "\n".join(lines[next(i for i, l in enumerate(lines) if l.startswith("env:")):])


@pytest.mark.parametrize("arm,full,seven", [("soccer8", "soccer15", "soccer7"), ("cpr6", "cpr13", "cpr7")])
def test_half_teacher_is_full_arm_with_complement_datasub(arm, full, seven):
    a, f, s = _flat(_teacher(arm)), _flat(_teacher(full)), _teacher(seven)["env"]["dataSub"]
    diff = {k for k in set(a) | set(f) if a.get(k) != f.get(k)}
    assert diff == {"env.dataSub"}
    assert a["env.dataSub"] == [x for x in f["env.dataSub"] if x not in set(s)]
    assert set(a["env.dataSub"]).isdisjoint(s)
    assert sorted(a["env.dataSub"] + s, key=lambda x: int(x[3:])) == \
           sorted(f["env.dataSub"], key=lambda x: int(x[3:])), "half + seven must be exactly the full set"
    ta = _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_teacher_g3_{arm}_geoall__f0.yaml"))))
    tf = _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_teacher_g3_{full}_geoall__f0.yaml"))))
    assert {k for k in set(ta) | set(tf) if ta.get(k) != tf.get(k)} == {"params.config.full_experiment_name"}
    assert ta["params.config.full_experiment_name"] == f"smplx_teacher_g3_{arm}_geoall__f0"
    la = open(os.path.join(ROOT, f"slurm_teacher_g3_{arm}_geoall__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_{arm}_geoall__f0.yaml" in la
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_g3_{arm}_geoall__f0.yaml" in la
    assert f'--job-name="tch-g3_{arm}_geoall__f0"' in la
    assert "for s in " + " ".join(a["env.dataSub"]) + "; do" in la      # the data guard uses the new list
    assert f"g3_{full}_geoall" not in la


@pytest.mark.parametrize("st", ["mlp_ret_stock", "xf_ret_nvadlr"])
def test_act_halves_student_is_act_with_other_teachers(st):
    a = _flat(yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_act_{st}__f0.yaml"))))
    h = _flat(yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_act_halves_{st}__f0.yaml"))))
    assert {k for k in set(a) | set(h) if a.get(k) != h.get(k)} == {"env.teacherPolicy"}
    assert h["env.teacherPolicy"] == "checkpoints/teachers/g3_act_halves"
    ta = _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_act_{st}__f0.yaml"))))
    th = _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_act_halves_{st}__f0.yaml"))))
    assert {k for k in set(ta) | set(th) if ta.get(k) != th.get(k)} == {"params.config.full_experiment_name"}
    la = open(os.path.join(ROOT, f"slurm_student_g3_act_halves_{st}__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_halves_{st}__f0.yaml" in la
    assert "--activities bball7 soccer7 soccer8 cpr7 cpr6" in la and "--out checkpoints/teachers/g3_act_halves" in la
    assert "--arms bball7 soccer15 cpr13" in la        # same merged data as act


def test_act_halves_twins_share_env_body():
    assert _body(os.path.join(CFG, "omomo_student_g3_act_halves_mlp_ret_stock__f0.yaml")) == \
           _body(os.path.join(CFG, "omomo_student_g3_act_halves_xf_ret_nvadlr__f0.yaml"))


def test_five_half_teachers_cover_all_35_people_once():
    subs = [s for n in ("bball7", "soccer7", "soccer8", "cpr7", "cpr6") for s in _teacher(n)["env"]["dataSub"]]
    assert len(subs) == 35 and len(set(subs)) == 35
    act = yaml.safe_load(open(os.path.join(CFG, "omomo_student_g3_act_halves_mlp_ret_stock__f0.yaml")))["env"]["dataSub"]
    assert sorted(subs, key=lambda x: int(x[3:])) == act

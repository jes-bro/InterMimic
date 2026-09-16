"""Pin the g3 student cfg pairs: the OMOMO students' env cfgs are the srcall13
teacher's data verbatim plus exactly the four distill keys; the MLP and XF
twins share an env byte-for-byte below the header; train cfgs carry the
intended network / optimizer and distinct experiment names; launchers point at
their own cfgs and the right task.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_student_g3_cfgs.py -q
"""
import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")

DISTILL_KEYS = {
    "teacherPolicy": "checkpoints/teachers/g3_omomo",
    "teacherPolicyCFG": "intermimic/data/cfg/train/rlg/omomo_teacher_g3_omomo_geoall__f0.yaml",
    "studentObsHorizons": [1, 4, 7, 10, 13, 16],
    "numObsRetarget": 9594,
}
OMOMO_ARMS = ["omomo_mlp_ret_stock", "omomo_xf_ret_nvadlr"]


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _env(name):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_{name}__f0.yaml")))


def _train(name):
    return yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_{name}__f0.yaml")))


def _body(path):
    lines = open(path).read().splitlines()
    return "\n".join(lines[next(i for i, l in enumerate(lines) if l.startswith("env:")):])


@pytest.mark.parametrize("arm", OMOMO_ARMS)
def test_omomo_env_is_srcall13_plus_distill_keys(arm):
    base = _flat(yaml.safe_load(open(os.path.join(CFG, "omomo_teacher_g3_omomo_geoall_srcall13__f0.yaml"))))
    stu = _flat(_env(arm))
    added = {k: v for k, v in stu.items() if k not in base}
    assert added == {f"env.{k}": v for k, v in DISTILL_KEYS.items()}
    changed = {k for k in base if k in stu and stu[k] != base[k]}
    assert not changed, f"student env changes teacher-data keys: {changed}"
    assert set(base) <= set(stu), f"student env drops keys: {set(base) - set(stu)}"


def test_omomo_twins_share_env_body():
    assert _body(os.path.join(CFG, "omomo_student_g3_omomo_mlp_ret_stock__f0.yaml")) == \
           _body(os.path.join(CFG, "omomo_student_g3_omomo_xf_ret_nvadlr__f0.yaml"))


def test_student_width_is_consistent_with_task_rule():
    e = _env("omomo_mlp_ret_stock")["env"]
    per_h = e["numObs"] // len(e["obsHorizons"])
    assert e["numObs"] % len(e["obsHorizons"]) == 0
    assert e["numObsRetarget"] == per_h * len(e["studentObsHorizons"])


def test_mlp_stock_train_cfg():
    t = _train("omomo_mlp_ret_stock")["params"]
    assert t["network"]["name"] == "intermimic"
    assert t["network"]["mlp"]["units"] == [1024, 1024, 512]
    c = t["config"]
    assert c["lr_schedule"] == "constant" and c["normalize_value"] is False
    assert "kl_threshold" not in c
    assert c["expert_loss_coef"] == 1 and c["save_intermediate"] is True
    assert c["full_experiment_name"] == "smplx_student_g3_omomo_mlp_ret_stock__f0"


def test_xf_nvadlr_train_cfg():
    t = _train("omomo_xf_ret_nvadlr")["params"]
    assert t["network"]["name"] == "intermimic_transformer"
    assert t["network"]["transformer"] == {"num_tokens": 6, "readout_token": 0}
    assert t["network"]["mlp"]["units"] == [1024, 1024, 512]
    c = t["config"]
    assert c["lr_schedule"] == "adaptive" and c["kl_threshold"] == 0.06 and c["normalize_value"] is True
    assert c["expert_loss_coef"] == 1 and c["save_intermediate"] is True
    assert c["full_experiment_name"] == "smplx_student_g3_omomo_xf_ret_nvadlr__f0"


def test_xf_tokens_match_student_horizons():
    e = _env("omomo_xf_ret_nvadlr")["env"]
    t = _train("omomo_xf_ret_nvadlr")["params"]["network"]["transformer"]
    assert t["num_tokens"] == len(e["studentObsHorizons"])
    assert e["studentObsHorizons"][t["readout_token"]] == 1, "readout must be the delta_t=1 token"


def test_train_cfgs_differ_only_where_intended():
    a = _flat(_train("omomo_mlp_ret_stock"))
    b = _flat(_train("omomo_xf_ret_nvadlr"))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {
        "params.network.name", "params.network.transformer.num_tokens",
        "params.network.transformer.readout_token", "params.config.lr_schedule",
        "params.config.kl_threshold", "params.config.normalize_value",
        "params.config.full_experiment_name",
    }


@pytest.mark.parametrize("arm", OMOMO_ARMS)
def test_launcher_points_at_own_cfgs_and_task(arm):
    path = os.path.join(ROOT, f"slurm_student_g3_{arm}__f0.sh")
    assert os.path.isfile(path), f"missing launcher {path}"
    s = open(path).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{arm}__f0.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{arm}__f0.yaml" in s
    assert "-m intermimic.run_distill" in s and "--task InterMimicDistillG3" in s
    assert "teachers.yaml" in s                       # teacher-set guard present
    assert "resume_from" in s                         # auto-resume block present
    assert f'--job-name="stu-g3_{arm}__f0"' in s

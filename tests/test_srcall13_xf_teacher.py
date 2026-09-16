"""The XF+NVADLR teacher twin of the srcall13 MLP teacher: env body identical
to the MLP arm's, train cfg = the g3 train cfg with exactly the transformer +
nvadlr knobs and its own name, launcher points at its own cfgs.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_srcall13_xf_teacher.py -q
"""
import os

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")
MLP = "omomo_teacher_g3_omomo_geoall_srcall13__f0"
XF = "omomo_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0"


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _body(path):
    lines = open(path).read().splitlines()
    return "\n".join(lines[next(i for i, l in enumerate(lines) if l.startswith("env:")):])


def test_env_body_identical_to_mlp_twin():
    assert _body(os.path.join(CFG, f"{MLP}.yaml")) == _body(os.path.join(CFG, f"{XF}.yaml"))


def test_train_cfg_differs_only_by_the_knobs():
    a = _flat(yaml.safe_load(open(os.path.join(RLG, f"{MLP}.yaml"))))
    b = _flat(yaml.safe_load(open(os.path.join(RLG, f"{XF}.yaml"))))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"params.network.name", "params.network.transformer.num_tokens",
                    "params.network.transformer.readout_token", "params.config.lr_schedule",
                    "params.config.kl_threshold", "params.config.normalize_value",
                    "params.config.full_experiment_name"}
    assert b["params.network.name"] == "intermimic_transformer"
    assert b["params.network.transformer.num_tokens"] == 6 and b["params.network.transformer.readout_token"] == 0
    assert b["params.config.lr_schedule"] == "adaptive" and b["params.config.kl_threshold"] == 0.06
    assert b["params.config.normalize_value"] is True
    assert b["params.config.full_experiment_name"] == f"smplx_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0"
    env = yaml.safe_load(open(os.path.join(CFG, f"{XF}.yaml")))["env"]
    assert len(env["obsHorizons"]) == 6 and env["obsHorizons"][0] == 1     # token 0 = delta_t 1


def test_launcher_points_at_own_cfgs():
    s = open(os.path.join(ROOT, f"slurm_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/{XF}.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/{XF}.yaml" in s
    assert '--job-name="tch-g3_omomo_geoall_srcall13_xf_nvadlr__f0"' in s
    assert "-m intermimic.run " in s and "--task InterMimic " in s
    assert "raggedMotionData" in s and "resume_from" in s

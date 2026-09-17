"""Arm A cfg pairs: bodyctr (body wire + contrastive twins) and bodyonly (wire
only) are the omomo XF student with exactly the Arm-A keys added; launchers
point at their own cfgs; the agent/task wiring exists behind those keys.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_arm_a_cfgs.py -q
"""
import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")
PKG = os.path.join(ROOT, "isaacgym", "src", "intermimic")
BASE = "omomo_xf_ret_nvadlr"


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _env(n):
    return _flat(yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_{n}__f0.yaml"))))


def _train(n):
    return _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_{n}__f0.yaml"))))


@pytest.mark.parametrize("arm,twins", [("bodyctr", True), ("bodyonly", False)])
def test_env_is_base_plus_arm_a_keys(arm, twins):
    b, a = _env(BASE), _env(f"{BASE}_{arm}")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"env.numObsRetarget", "env.studentBodyFeatures", "env.twinEnvs"}
    assert a["env.numObsRetarget"] == 9594 + 156 and a["env.studentBodyFeatures"] is True
    assert a["env.twinEnvs"] is twins
    assert a["env.teacherPolicy"] == b["env.teacherPolicy"]          # same 13 teachers


def test_bodyctr_train_cfg():
    b, a = _train(BASE), _train(f"{BASE}_bodyctr")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"params.config.full_experiment_name", "params.network.transformer.body_dim",
                    "params.network.transformer.contrastive", "params.network.transformer.proj_dim",
                    "params.config.contrastive.coef", "params.config.contrastive.temperature"}
    assert a["params.network.transformer.body_dim"] == 156 and a["params.network.transformer.contrastive"] is True
    assert a["params.config.contrastive.coef"] == 0.1 and a["params.config.contrastive.temperature"] == 0.1
    assert a["params.config.full_experiment_name"] == f"smplx_student_g3_{BASE}_bodyctr__f0"


def test_bodyonly_train_cfg():
    b, a = _train(BASE), _train(f"{BASE}_bodyonly")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"params.config.full_experiment_name", "params.network.transformer.body_dim"}
    assert a["params.network.transformer.body_dim"] == 156
    assert "params.config.contrastive.coef" not in a                   # no twin term


def test_body_dim_matches_env_and_token_math():
    for arm in ("bodyctr", "bodyonly"):
        e, t = _env(f"{BASE}_{arm}"), _train(f"{BASE}_{arm}")
        bd = t["params.network.transformer.body_dim"]
        assert e["env.numObsRetarget"] - bd == e["env.numObs"]          # tokens = the teacher's 6 x 1599
        assert (e["env.numObsRetarget"] - bd) % t["params.network.transformer.num_tokens"] == 0


@pytest.mark.parametrize("arm", ["bodyctr", "bodyonly"])
def test_launcher(arm):
    s = open(os.path.join(ROOT, f"slurm_student_g3_{BASE}_{arm}__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{BASE}_{arm}__f0.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{BASE}_{arm}__f0.yaml" in s
    assert f'--job-name="stu-g3_{BASE}_{arm}__f0"' in s and "--task InterMimicDistillG3" in s


def test_wiring_exists_behind_flags():
    task = open(os.path.join(PKG, "env", "tasks", "intermimic_distill_g3.py")).read()
    agent = open(os.path.join(PKG, "learning", "intermimic_agent_distill.py")).read()
    parent = open(os.path.join(PKG, "env", "tasks", "intermimic.py")).read()
    assert "studentBodyFeatures" in task and "twinEnvs" in task and "def _twin_sync" in task
    assert "def _update_twin_valid" in task
    assert "def _contrastive_loss" in agent and "config.get('contrastive')" in agent
    assert "self.contrastive_coef * ctr_loss" in agent
    assert parent.count("self._twin_sync(env_ids, i, motion_times, idx)") == 2   # both reset paths
    assert "'studentBodyFeatures', 'twinEnvs'" in parent                          # whitelisted

"""matchctr arms (matched-frame contrastive term, cohort clip sampling) for both
students, with and without the body wire:

  matchctr      = plain + cohortClips 4 (env) + contrastive{positives matched, coef, temp}
                  + transformer.contrastive/proj_dim (train)
  matchctr_body = matchctr + studentBodyFeatures true / numObsRetarget 9750 (env)
                  + transformer.body_dim 156 (train)

The 2x2 with plain and bodyonly is what attributes the effect: each comparison
changes exactly one thing, and these tests pin that.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_matchctr_cfgs.py -q
"""
import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")
PKG = os.path.join(ROOT, "isaacgym", "src", "intermimic")
BASES = ["omomo_xf_ret_nvadlr", "act_xf_ret_nvadlr"]

RLG_CTR_KEYS = {"params.config.full_experiment_name", "params.config.contrastive.positives",
                "params.config.contrastive.coef", "params.config.contrastive.temperature",
                "params.network.transformer.contrastive", "params.network.transformer.proj_dim"}


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


def _diff(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


@pytest.mark.parametrize("BASE", BASES)
def test_matchctr_env_is_plain_plus_cohorts(BASE):
    b, a = _env(BASE), _env(f"{BASE}_matchctr")
    assert _diff(a, b) == {"env.cohortClips"} and a["env.cohortClips"] == 4
    assert "env.twinEnvs" not in a and "env.studentBodyFeatures" not in a


@pytest.mark.parametrize("BASE", BASES)
def test_matchctr_train_is_plain_plus_matched_term(BASE):
    b, a = _train(BASE), _train(f"{BASE}_matchctr")
    assert _diff(a, b) == RLG_CTR_KEYS
    assert a["params.config.contrastive.positives"] == "matched"
    assert a["params.config.contrastive.coef"] == 0.1 and a["params.config.contrastive.temperature"] == 0.1
    assert a["params.network.transformer.contrastive"] is True and a["params.network.transformer.proj_dim"] == 128
    assert "params.network.transformer.body_dim" not in a
    assert a["params.config.full_experiment_name"] == f"smplx_student_g3_{BASE}_matchctr__f0"


@pytest.mark.parametrize("BASE", BASES)
def test_matchctr_body_is_matchctr_plus_wire(BASE):
    e0, e1 = _env(f"{BASE}_matchctr"), _env(f"{BASE}_matchctr_body")
    assert _diff(e1, e0) == {"env.studentBodyFeatures", "env.numObsRetarget"}
    assert e1["env.studentBodyFeatures"] is True and e1["env.numObsRetarget"] == 9594 + 156
    t0, t1 = _train(f"{BASE}_matchctr"), _train(f"{BASE}_matchctr_body")
    assert _diff(t1, t0) == {"params.config.full_experiment_name", "params.network.transformer.body_dim"}
    assert t1["params.network.transformer.body_dim"] == 156
    # and vs bodyonly (wire, no term): exactly the matched term. bodyonly spells
    # out twinEnvs: false; matchctr_body leaves the key absent -- both are off.
    bo = _train(f"{BASE}_bodyonly")
    assert _diff(t1, bo) == RLG_CTR_KEYS
    eb = _env(f"{BASE}_bodyonly")
    assert eb.pop("env.twinEnvs") is False and "env.twinEnvs" not in e1
    assert _diff(e1, eb) == {"env.cohortClips"}


@pytest.mark.parametrize("BASE", BASES)
@pytest.mark.parametrize("arm", ["matchctr", "matchctr_body"])
def test_launcher(BASE, arm):
    s = open(os.path.join(ROOT, f"slurm_student_g3_{BASE}_{arm}__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{BASE}_{arm}__f0.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{BASE}_{arm}__f0.yaml" in s
    assert f'--job-name="stu-g3_{BASE}_{arm}__f0"' in s and "--task InterMimicDistillG3" in s
    code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#") or l.startswith("#SBATCH"))
    assert f"{BASE}__f0" not in code.replace(f"{BASE}_{arm}__f0", "")     # no stale plain names in code


def test_wiring_exists_behind_flags():
    task = open(os.path.join(PKG, "env", "tasks", "intermimic_distill_g3.py")).read()
    agent = open(os.path.join(PKG, "learning", "intermimic_agent_distill.py")).read()
    parent = open(os.path.join(PKG, "env", "tasks", "intermimic.py")).read()
    assert "cohortClips" in task and "def _cohort_sync" in task
    assert "'cohortClips'" in parent                                       # whitelisted
    # cohort sync runs inside the reset hook the parent already calls on both reset paths
    hook = task.split("def _twin_sync")[1].split("def _cohort_sync")[0]
    assert "self._cohort_sync(env_ids, motion_ids, motion_times, ref_idx)" in hook
    assert "def _matched_loss" in agent and "find_matched_pairs(" in agent and "matched_infonce(" in agent
    assert "contrastive_mode == 'matched'" in agent
    # matched mode must refuse to coexist with twins, and twins mode is untouched
    assert "'matched' with twinEnvs on: pick one" in agent
    assert "self.contrastive_coef > 0 and self.contrastive_mode == 'twins'" in agent

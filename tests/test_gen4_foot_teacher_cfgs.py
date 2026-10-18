"""The gen4 FOOT-SKATE teacher arms (scripts/make_gen4_foot_teacher_cfgs.py): every
_flight2_fullep teacher twinned as _flight2foot_fullep with rewardTerms.footSkate on
and nothing else changed.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_foot_teacher_cfgs.py -q
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_foot_teacher_cfgs.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
ARMS = [(b, mk.arm_name(b)) for b in mk.BASES]


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def test_the_set():
    assert len(ARMS) == 21 and all(b.endswith("_flight2_fullep__f0") for b, _ in ARMS)
    assert mk.arm_name("gen4_x_flight2_fullep__f0") == "gen4_x_flight2foot_fullep__f0"
    assert mk.FOOT == {"lambda": 2.0, "bodies": ["L_Toe", "R_Toe"], "contactThreshold": 1.0}


@pytest.mark.parametrize("base,arm", ARMS, ids=[a for _, a in ARMS])
def test_env_is_the_flight2_twin_plus_the_foot_block(base, arm):
    twin = yaml.safe_load(open(CFG / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / f"omomo_teacher_{arm}.yaml"))
    assert _diff(twin, new) == {"env.rewardTerms.footSkate.enable", "env.rewardTerms.footSkate.lambda",
                                "env.rewardTerms.footSkate.bodies", "env.rewardTerms.footSkate.contactThreshold"}
    e = new["env"]
    assert e["rewardTerms"]["footSkate"] == {"enable": True, "lambda": 2.0, "bodies": ["L_Toe", "R_Toe"], "contactThreshold": 1.0}
    assert e["rewardTerms"]["pose"]["enable"] is False                         # still nopose
    assert e["objectFlightReward"] == {"enable": True, "rootFlight": 0.6, "flatFlight": 0.3}
    assert e["rewardShape"] == "geometric_all"
    assert list(e["rewardTerms"]) == ["footSkate", "freeFlightGate", "pose"]  # key order kept
    assert e["keyBodies"].count("L_Toe") == 1 and e["contactBodies"].count("R_Toe") == 1   # the feet exist in the body lists


@pytest.mark.parametrize("base,arm", ARMS, ids=[a for _, a in ARMS])
def test_train_cfg_differs_only_in_experiment_name(base, arm):
    twin = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml"))
    assert _diff(twin, new) == {"params.config.full_experiment_name"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"


@pytest.mark.parametrize("base,arm", ARMS, ids=[a for _, a in ARMS])
def test_launcher(base, arm):
    txt = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert base not in code and f'--job-name="tch-{arm}"' in txt and f"smplx_teacher_{arm}/nn/" in code
    assert "footSkate:" in code and "lambda:\\s*2.0" in code                    # the guard
    assert "objectFlightReward:" in code and "flatFlight:\\s*0.3" in code      # the flight2 guard survives
    assert code.index("footSkate:") < code.index("python -u -m intermimic.run")
    assert "FOOT-SKATE PENALTY" in code
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{arm}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_guard_refuses_the_flight2_cfg():
    import re
    base, arm = ARMS[2]                                                         # chairs
    sh = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    g = re.search(r"if ! grep -qE '\^\\s\*footSkate:'.*?\nfi\n", sh, re.S).group(0)
    for cfg, expect in ((CFG / f"omomo_teacher_{arm}.yaml", 0), (CFG / f"omomo_teacher_{base}.yaml", 1)):
        r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"\n{g}'], capture_output=True, text=True)
        assert r.returncode == expect, (cfg.name, r.stderr)


def test_generator_is_idempotent():
    before = {p: p.read_text() for _, arm in ARMS for p in (
        CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml", REPO / f"slurm_teacher_{arm}.sh")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_foot_teacher_cfgs.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert p.read_text() == txt, p

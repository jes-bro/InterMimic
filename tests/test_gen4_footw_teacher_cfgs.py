"""The from-scratch FOOT-WEIGHT teacher arms (scripts/make_gen4_footw_teacher_cfgs.py): every
_msexp_fullep (and _flight_fullep) teacher twinned with rewardTerms.footWeight {5} on and
nothing else changed -- the from-scratch twins of fine-tune variants D and B.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_footw_teacher_cfgs.py -q
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
spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_footw_teacher_cfgs.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
ARMS = [(src, b, mk.arm_name(b, src)) for src in ("msexp", "flight") for b in mk.bases(src)]


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


def test_the_sets():
    assert len(ARMS) == 42
    assert mk.arm_name("gen4_x_msexp_fullep__f0", "msexp") == "gen4_x_msexpfootw_fullep__f0"
    assert mk.arm_name("gen4_x_flight_fullep__f0", "flight") == "gen4_x_flightfootw_fullep__f0"
    assert mk.FOOT_WEIGHT == 5.0
    # the from-scratch flightfootw arms never collide with the ft<epoch>k_flightfootw fine-tunes
    assert not any("_ft" in a for _, _, a in ARMS)


@pytest.mark.parametrize("src,base,arm", ARMS, ids=[a for _, _, a in ARMS])
def test_env_is_the_twin_plus_exactly_the_foot_weight(src, base, arm):
    twin = yaml.safe_load(open(CFG / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / f"omomo_teacher_{arm}.yaml"))
    assert _diff(twin, new) == {"env.rewardTerms.footWeight.enable", "env.rewardTerms.footWeight.weight"}
    e = new["env"]
    assert e["rewardTerms"]["footWeight"] == {"enable": True, "weight": 5.0}
    assert list(e["rewardTerms"]) == sorted(e["rewardTerms"])
    assert ("motionScaleReward" in e) == (src == "msexp") and ("objectFlightReward" in e) == (src == "flight")
    assert e["rolloutLength"] == 1000 and e["minEpisodeFrames"] == 30 and e["rewardShape"] == "geometric_all"


@pytest.mark.parametrize("src,base,arm", ARMS, ids=[a for _, _, a in ARMS])
def test_train_cfg_differs_only_in_experiment_name(src, base, arm):
    twin = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml"))
    assert _diff(twin, new) == {"params.config.full_experiment_name"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"
    assert new["params"]["config"]["resume_from"] == "None"                      # from scratch


@pytest.mark.parametrize("src,base,arm", ARMS, ids=[a for _, _, a in ARMS])
def test_launcher(src, base, arm):
    txt = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert base not in code and f'--job-name="tch-{arm}"' in txt and f"smplx_teacher_{arm}/nn/" in code
    assert "footWeight:" in code and "weight:\\s*5.0" in code
    assert ("motionScaleReward:" in code) == (src == "msexp") and ("objectFlightReward:" in code) == (src == "flight")
    assert code.index("footWeight:") < code.index("\npython -u -m intermimic.run")
    assert "FOOT WEIGHT x5.0" in code
    assert "fresh start" in code                                                  # a from-scratch launcher
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{arm}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_guard_refuses_the_twin_cfg():
    src, base, arm = ARMS[2]                                                      # chairs msexpfootw
    sh = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    g = re.search(r"if ! grep -qE '\^\\s\*footWeight:'.*?\nfi\n", sh, re.S).group(0)
    for cfg, expect in ((CFG / f"omomo_teacher_{arm}.yaml", 0), (CFG / f"omomo_teacher_{base}.yaml", 1)):
        r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"\n{g}'], capture_output=True, text=True)
        assert r.returncode == expect, (cfg.name, r.stderr)


def test_generator_is_idempotent():
    files = [p for _, _, arm in ARMS for p in (CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml", REPO / f"slurm_teacher_{arm}.sh")]
    before = {p: p.read_text() for p in files}
    for src in ("msexp", "flight"):
        r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_footw_teacher_cfgs.py"), "--from", src], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert p.read_text() == txt, p


def test_weight_2p5_set_exists_and_differs_only_in_the_weight():
    """--weight 2.5 writes the footw2p5 twins ('p' for the decimal point): same derivation, weight 2.5."""
    for src, base, arm in ARMS[:4] + ARMS[21:25]:
        arm2 = arm.replace("footw_", "footw2p5_")
        e5 = yaml.safe_load(open(CFG / f"omomo_teacher_{arm}.yaml"))
        e2 = yaml.safe_load(open(CFG / f"omomo_teacher_{arm2}.yaml"))
        assert _diff(e5, e2) == {"env.rewardTerms.footWeight.weight"}
        assert e2["env"]["rewardTerms"]["footWeight"]["weight"] == 2.5
        sh = (REPO / f"slurm_teacher_{arm2}.sh").read_text()
        assert "weight:\\s*2.5" in sh and f"smplx_teacher_{arm2}/nn/" in sh


def test_weight_2_flight_set_exists_and_differs_only_in_the_weight():
    """--weight 2 --from flight writes the flightfootw2 twins (Jess 2026-10-21: flight gets the lighter weight)."""
    for src, base, arm in ARMS[21:27]:
        arm2 = arm.replace("footw_", "footw2_")
        e5 = yaml.safe_load(open(CFG / f"omomo_teacher_{arm}.yaml"))
        e2 = yaml.safe_load(open(CFG / f"omomo_teacher_{arm2}.yaml"))
        assert _diff(e5, e2) == {"env.rewardTerms.footWeight.weight"}
        assert e2["env"]["rewardTerms"]["footWeight"]["weight"] == 2.0
        assert "weight:\\s*2.0" in (REPO / f"slurm_teacher_{arm2}.sh").read_text()

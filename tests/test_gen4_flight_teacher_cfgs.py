"""The gen4 OBJECT-FLIGHT teacher arms (scripts/make_gen4_flight_teacher_cfgs.py): every
_msexp_fullep teacher twinned as _flight (flatFlight 0) and _flight2 (flatFlight 0.3),
each DERIVED from its twin with exactly the reward block swapped, the guard swapped,
and the arm renamed.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_flight_teacher_cfgs.py -q
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
spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_flight_teacher_cfgs.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)

PAIRS = [(base, tag, flat) for base in mk.BASES for tag, flat in mk.VARIANTS.items()]
IDS = [mk.arm_name(b, t) for b, t, _ in PAIRS]


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


def test_the_fleet():
    assert len(mk.BASES) == 21 and len(PAIRS) == 42
    assert all(b.endswith("_msexp_fullep__f0") for b in mk.BASES)            # full episodes only
    assert mk.VARIANTS == {"flight": 0.0, "flight2": 0.3} and mk.ROOT_FLIGHT == 0.6
    assert mk.arm_name("gen4_x_msexp_fullep__f0", "flight2") == "gen4_x_flight2_fullep__f0"


@pytest.mark.parametrize("base,tag,flat", PAIRS, ids=IDS)
def test_env_is_the_msexp_twin_with_exactly_the_reward_block_swapped(base, tag, flat):
    twin = yaml.safe_load(open(CFG / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml"))
    assert _diff(twin, new) == {
        "env.motionScaleReward.enable", "env.motionScaleReward.referenceStd",
        "env.objectFlightReward.enable", "env.objectFlightReward.rootFlight", "env.objectFlightReward.flatFlight"}
    e = new["env"]
    assert "motionScaleReward" not in e
    assert e["objectFlightReward"] == {"enable": True, "rootFlight": 0.6, "flatFlight": flat}
    assert e["rewardShape"] == "geometric_all"                                 # the exponent replaces its root
    assert e["rolloutLength"] == 1000 and e["minEpisodeFrames"] == 30         # the fullep twin's episode rule
    # the block sits in key order (the fleet's cfgs are key-sorted)
    keys = [l.split(":")[0].strip() for l in open(CFG / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml")
            if re.match(r"^  [A-Za-z_]+:", l)]
    i = keys.index("objectFlightReward")
    assert keys[i - 1] < "objectFlightReward" < keys[i + 1], (keys[i - 1], keys[i + 1])


@pytest.mark.parametrize("base,tag,flat", PAIRS, ids=IDS)
def test_train_cfg_differs_only_in_experiment_name(base, tag, flat):
    twin = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"))
    new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml"))
    assert _diff(twin, new) == {"params.config.full_experiment_name"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{mk.arm_name(base, tag)}"


@pytest.mark.parametrize("base,tag,flat", PAIRS, ids=IDS)
def test_launcher(base, tag, flat):
    arm = mk.arm_name(base, tag)
    txt = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert base not in code and "msexp" not in code                            # fully renamed
    assert f'--job-name="tch-{arm}"' in txt and f"teacher-{arm}-%j.out" in txt
    assert f"omomo_teacher_{arm}.yaml" in code and f"smplx_teacher_{arm}/nn/" in code
    # the guard: flight block + BOTH constants, the motion-scale guard gone
    assert "objectFlightReward:" in code and f"rootFlight:\\s*0.6" in code and f"flatFlight:\\s*{flat}" in code
    assert "motionScaleReward" not in code
    assert code.index("objectFlightReward:") < code.index("python -u -m intermimic.run")
    assert "OBJECT-FLIGHT EXPONENT" in code                                      # the recipe echo
    # the twin's other guards survive (full episodes; the gen4 body files)
    assert "rolloutLength:\\s*1000" in code and "minEpisodeFrames:\\s*30" in code
    assert "_inertial.xml" in code
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{arm}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_flight_and_flight2_launchers_refuse_each_others_cfg():
    """The guards grep the exact flatFlight, so the two variants cannot be mixed up."""
    base = mk.BASES[2]                                                          # chairs
    for tag, other in (("flight", "flight2"), ("flight2", "flight")):
        sh = (REPO / f"slurm_teacher_{mk.arm_name(base, tag)}.sh").read_text()
        g = re.search(r"if ! grep -qE '\^\\s\*objectFlightReward:'.*?\nfi\n", sh, re.S).group(0)
        cfg_ok = CFG / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml"
        cfg_other = CFG / f"omomo_teacher_{mk.arm_name(base, other)}.yaml"
        for cfg, expect in ((cfg_ok, 0), (cfg_other, 1)):
            r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"\n{g}'], capture_output=True, text=True)
            assert r.returncode == expect, (tag, cfg.name, r.stderr)


def test_generator_is_idempotent():
    before = {p: p.read_text() for base, tag, _ in PAIRS for p in (
        CFG / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml",
        CFG / "train/rlg" / f"omomo_teacher_{mk.arm_name(base, tag)}.yaml",
        REPO / f"slurm_teacher_{mk.arm_name(base, tag)}.sh")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_flight_teacher_cfgs.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert p.read_text() == txt, p

"""The InterMimic BASELINE teachers (scripts/make_intermimic_baseline_cfgs.py): upstream's own
per-subject recipe, one arm per training source, changed in exactly one block -- our free-flight
gate -- plus the data dir. Base body = upstream's smplx/omomo.xml (Subject 9), no subjectBodies.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_intermimic_baseline_cfgs.py -q
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
spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_intermimic_baseline_cfgs.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
IDS = [f"src{n}" for n in mk.SOURCES]
PAPER_WEIGHTS = {"p": 30.0, "r": 2.5, "ig": 5.0, "or": 0.1, "op": 5.0, "cg_hand": 5.0, "cg_other": 5.0, "cg_all": 3.0,
                 "eg1": 2e-5, "eg2": 2e-5, "eg3": 1e-9}     # supp. Table C: (30, 2.5, 5, 0.1, 5, 5, 5, 3, 2e-5, 2e-5, 1e-9)


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


def test_upstream_bases_are_checked_in_and_are_the_recipe():
    e = yaml.safe_load(open(mk.ENV_BASE))["env"]
    assert e["robotType"] == "smplx/omomo.xml" and "subjectBodies" not in e and "retargetedMotionDir" not in e
    assert e["stateInit"] == "Hybrid" and e["hybridInitProb"] == 0.1 and e["physicalBufferSize"] == 3 and e["rolloutLength"] == 300
    assert e["numObs"] == 3198 and e["numEnvs"] == 4096 and "rewardShape" not in e and "rewardTerms" not in e
    for k, v in PAPER_WEIGHTS.items():
        assert e["rewardWeights"][k] == pytest.approx(v), k
    t = yaml.safe_load(open(mk.TRAIN_BASE))["params"]
    assert t["network"]["name"] == "intermimic" and t["network"]["mlp"]["units"] == [1024, 1024, 512]
    assert t["config"]["lr_schedule"] == "constant" and t["config"]["normalize_value"] is False
    assert t["config"]["resume_from"] == "omomo/init/nn/mimic.pth"      # upstream's shared init, which we lack
    assert len(mk.SOURCES) == 13 and not ({4, 10, 13, 16} & set(mk.SOURCES))


@pytest.mark.parametrize("n", mk.SOURCES, ids=IDS)
def test_env_is_upstreams_plus_exactly_the_gate_and_the_data_dir(n):
    base = yaml.safe_load(open(mk.ENV_BASE))
    new = yaml.safe_load(open(CFG / f"omomo_teacher_{mk.arm_name(n)}.yaml"))
    want = {"env.motion_file", "env.rewardTerms.freeFlightGate.resets", "env.rewardTerms.freeFlightGate.reward"}
    if n != 2:
        want.add("env.dataSub")
    assert _diff(base, new) == want
    e = new["env"]
    assert e["dataSub"] == [f"sub{n}"] and e["motion_file"] == "InterAct/OMOMO_new"
    assert e["rewardTerms"] == {"freeFlightGate": {"resets": True, "reward": False}}
    assert e["robotType"] == "smplx/omomo.xml" and "subjectBodies" not in e


@pytest.mark.parametrize("n", mk.SOURCES, ids=IDS)
def test_train_cfg_renames_and_trains_from_scratch(n):
    base = yaml.safe_load(open(mk.TRAIN_BASE))
    new = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{mk.arm_name(n)}.yaml"))
    assert _diff(base, new) == {"params.config.full_experiment_name", "params.config.resume_from"}
    assert new["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{mk.arm_name(n)}"
    assert new["params"]["config"]["resume_from"] == "None"


@pytest.mark.parametrize("n", mk.SOURCES, ids=IDS)
def test_launcher(n):
    arm = mk.arm_name(n)
    txt = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert f'--job-name="tch-{arm}"' in txt and f"teacher-{arm}-%j.out" in txt and "--time=7-00:00:00" in txt
    assert f"omomo_teacher_{arm}.yaml" in code and f"smplx_teacher_{arm}/nn/" in code
    assert f"dataSub:\\s*\\['sub{n}'\\]" in code                                  # one source
    assert "(subjectBodies|retargetedMotionDir|humanoidAssetSuffix|motionScaleReward|objectFlightReward|rewardShape):" in code
    assert 'robotType:\\s*"smplx/omomo\\.xml"' in code and "freeFlightGate:" in code
    assert "intermimic_transformer" in code                                       # refused
    assert 'NUM_ENVS="${NUM_ENVS:-4096}"' in code
    assert "python -u -m intermimic.run" in code and "--task InterMimic" in code
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{arm}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_guards_refuse_a_wrong_cfg(tmp_path):
    """The launcher's guards, exercised: our chairs flight cfg (subjectBodies, retarget, exponent) is refused."""
    sh = (REPO / f"slurm_teacher_{mk.arm_name(6)}.sh").read_text()
    guards = re.search(r"^# Baseline guards:.*?(?=\n\necho )", sh, re.M | re.S).group(0)
    good = CFG / f"omomo_teacher_{mk.arm_name(6)}.yaml"
    bad = CFG / "omomo_teacher_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_flight_fullep__f0.yaml"
    train = CFG / "train/rlg" / f"omomo_teacher_{mk.arm_name(6)}.yaml"
    for cfg, expect in ((good, 0), (bad, 1)):
        r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"; CFG_TRAIN="{train}"\n{guards}'], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == expect, (cfg.name, r.stderr)


@pytest.mark.parametrize("n", mk.SOURCES, ids=IDS)
def test_eval_cfg_mirrors_the_arm_on_the_base_body(n):
    arm = mk.arm_name(n)
    c = yaml.safe_load(open(CFG / f"omomo_eval_{arm}.yaml"))
    assert c["evalFor"] == [arm]
    e = c["env"]
    assert e["stateInit"] == "Start" and e["enableEvaluation"] is True and e["rolloutLength"] == 1000
    assert e["physicalBufferSize"] == 1 and e["numEnvs"] == 1024 and e["dataSub"] == [f"sub{n}"]
    assert "subjectBodies" not in e and e["robotType"] == "smplx/omomo.xml"   # BODY=generic at render time
    r = subprocess.run([sys.executable, str(REPO / "scripts/check_eval_cfg.py"), "--arm", arm], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_generator_is_idempotent():
    files = [p for n in mk.SOURCES for p in (CFG / f"omomo_teacher_{mk.arm_name(n)}.yaml",
                                              CFG / "train/rlg" / f"omomo_teacher_{mk.arm_name(n)}.yaml",
                                              REPO / f"slurm_teacher_{mk.arm_name(n)}.sh", CFG / f"omomo_eval_{mk.arm_name(n)}.yaml")]
    before = {p: p.read_text() for p in files}
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_intermimic_baseline_cfgs.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert p.read_text() == txt, p

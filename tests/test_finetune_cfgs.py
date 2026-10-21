"""Fine-tune arms (scripts/make_finetune_cfgs.py): a trained gen4 teacher continued under a
variant reward / episode rule as a NEW arm starting from the trained checkpoint.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_finetune_cfgs.py -q
"""
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_finetune_cfgs.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
CASES = [(short, tag) for short in sorted(mk.TRAINED) for tag in mk.VARIANTS]
IDS = [f"{s}-{t}" for s, t in CASES]


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


def test_names_and_checkpoints():
    assert mk.arm_name("gen4_x_msexp__f0", 44000, "flightfootw") == "gen4_x_ft44k_flightfootw_fullep__f0"
    assert mk.ckpt_path("gen4_x_msexp__f0", 44000) == "checkpoints/smplx_teacher_gen4_x_msexp__f0/nn/mimic_00044000.pth"
    assert set(mk.VARIANTS) == {"flight", "flightfootw", "flightfootwswing", "footw"}
    assert mk.TRAINED["chairs"] == ("gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0", 44000)
    assert len(CASES) == 24


@pytest.mark.parametrize("short,tag", CASES, ids=IDS)
def test_env_is_the_base_plus_exactly_the_variants_keys(short, tag):
    base, epoch = mk.TRAINED[short]
    arm = mk.arm_name(base, epoch, tag)
    b = yaml.safe_load(open(CFG / f"omomo_teacher_{base}.yaml"))
    n = yaml.safe_load(open(CFG / f"omomo_teacher_{arm}.yaml"))
    assert _diff(b, n) == mk.expected_diff(tag)
    e = n["env"]
    fl, fw, fs = mk.VARIANTS[tag]
    assert e["rolloutLength"] == 1000 and e["minEpisodeFrames"] == 30
    assert ("objectFlightReward" in e) == fl and ("motionScaleReward" in e) == (not fl)
    if fl:
        assert e["objectFlightReward"] == {"enable": True, "rootFlight": 0.6, "flatFlight": 0.0}
    assert ("footWeight" in e["rewardTerms"]) == fw and ("footSwing" in e["rewardTerms"]) == fs
    if fw:
        assert e["rewardTerms"]["footWeight"] == {"enable": True, "weight": 5.0}
    if fs:
        assert e["rewardTerms"]["footSwing"] == {"enable": True, "lambda": 50.0}
    assert e["rewardShape"] == "geometric_all"
    assert list(e["rewardTerms"]) == sorted(e["rewardTerms"])              # key order kept


@pytest.mark.parametrize("short,tag", CASES, ids=IDS)
def test_train_cfg_renames_and_starts_from_the_base_checkpoint(short, tag):
    base, epoch = mk.TRAINED[short]
    arm = mk.arm_name(base, epoch, tag)
    b = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"))
    n = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml"))
    assert _diff(b, n) == {"params.config.full_experiment_name", "params.config.resume_from"}
    assert n["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"
    assert n["params"]["config"]["resume_from"] == mk.ckpt_path(base, epoch)   # the BASE's path, not the arm's
    assert arm not in n["params"]["config"]["resume_from"]


@pytest.mark.parametrize("short,tag", CASES, ids=IDS)
def test_launcher(short, tag):
    base, epoch = mk.TRAINED[short]
    arm = mk.arm_name(base, epoch, tag)
    fl, fw, fs = mk.VARIANTS[tag]
    txt = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert f'--job-name="tch-{arm}"' in txt and f"smplx_teacher_{arm}/nn/" in code
    assert f'BASE_CKPT="{mk.ckpt_path(base, epoch)}"' in code                 # the base's checkpoint, by name
    assert "rolloutLength:\\s*1000" in code and "minEpisodeFrames:\\s*30" in code
    assert ("objectFlightReward:" in code) == fl and ("motionScaleReward:" in code) == (not fl)
    assert ("footWeight:" in code) == fw and ("footSwing:" in code) == fs
    assert "RESUMING the fine-tune from its own" in code and "FINE-TUNE from the base checkpoint" in code
    assert "fresh start" not in code                                           # a fine-tune never starts fresh
    assert code.index("BASE_CKPT=") < code.index("\npython -u -m intermimic.run")   # the invocation, not the echo of it
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{arm}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_resume_block_prefers_own_checkpoint_and_refuses_a_missing_base(tmp_path):
    base, epoch = mk.TRAINED["chairs"]
    arm = mk.arm_name(base, epoch, "flightfootw")
    sh = (REPO / f"slurm_teacher_{arm}.sh").read_text()
    block = re.search(r"^# --- fine-tune resume:.*?\nfi\n", sh, re.M | re.S).group(0)
    (tmp_path / "block.sh").write_text(block)
    (tmp_path / "train.yaml").write_text((CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml").read_text())
    def run():
        return subprocess.run(["bash", "-c", 'CFG_TRAIN=train.yaml; source block.sh; grep resume_from "$CFG_TRAIN"'],
                              cwd=tmp_path, capture_output=True, text=True)
    r = run()
    assert r.returncode == 1 and "base checkpoint not found" in r.stdout + r.stderr
    basedir = tmp_path / os.path.dirname(mk.ckpt_path(base, epoch)); basedir.mkdir(parents=True)
    (tmp_path / mk.ckpt_path(base, epoch)).write_text("")
    r = run()
    assert r.returncode == 0 and "FINE-TUNE from the base checkpoint" in r.stdout and mk.ckpt_path(base, epoch) in r.stdout
    own = tmp_path / f"checkpoints/smplx_teacher_{arm}/nn"; own.mkdir(parents=True)
    (own / "mimic.pth").write_text("")
    r = run()
    assert r.returncode == 0 and "RESUMING the fine-tune from its own" in r.stdout
    assert f"resume_from: 'checkpoints/smplx_teacher_{arm}/nn/mimic.pth'" in r.stdout


@pytest.mark.parametrize("short,tag", CASES, ids=IDS)
def test_eval_cfg_mirrors_the_arm(short, tag):
    base, epoch = mk.TRAINED[short]
    arm = mk.arm_name(base, epoch, tag)
    r = subprocess.run([sys.executable, str(REPO / "scripts/check_eval_cfg.py"), "--arm", arm], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith(f"omomo_eval_{arm}.yaml")


def test_generator_is_idempotent():
    files = []
    for short, tag in CASES:
        base, epoch = mk.TRAINED[short]
        arm = mk.arm_name(base, epoch, tag)
        files += [CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml",
                  REPO / f"slurm_teacher_{arm}.sh", CFG / f"omomo_eval_{arm}.yaml"]
    before = {p: p.read_text() for p in files}
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_finetune_cfgs.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for p, txt in before.items():
        assert p.read_text() == txt, p


def test_weight_2p5_fine_tunes_exist_and_differ_only_in_the_weight():
    """--weight 2.5 writes the footw2p5 / flightfootw2p5 / flightfootw2p5swing fine-tunes."""
    for short in ("chairs", "boxes"):
        base, epoch = mk.TRAINED[short]
        for tag in ("footw", "flightfootw", "flightfootwswing"):
            a5 = mk.arm_name(base, epoch, tag)
            a2 = a5.replace("footw", "footw2p5")
            e5 = yaml.safe_load(open(CFG / f"omomo_teacher_{a5}.yaml"))
            e2 = yaml.safe_load(open(CFG / f"omomo_teacher_{a2}.yaml"))
            assert _diff(e5, e2) == {"env.rewardTerms.footWeight.weight"}
            assert e2["env"]["rewardTerms"]["footWeight"]["weight"] == 2.5
            t2 = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{a2}.yaml"))["params"]["config"]
            assert t2["full_experiment_name"] == f"smplx_teacher_{a2}" and t2["resume_from"] == mk.ckpt_path(base, epoch)
            sh = (REPO / f"slurm_teacher_{a2}.sh").read_text()
            assert "weight:\\s*2.5" in sh and f'BASE_CKPT="{mk.ckpt_path(base, epoch)}"' in sh
            r = subprocess.run([sys.executable, str(REPO / "scripts/check_eval_cfg.py"), "--arm", a2], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr

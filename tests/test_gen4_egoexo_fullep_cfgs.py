#!/usr/bin/env python3
"""The gen4 EgoExo FULL-EPISODE arms (make_gen4_egoexo_teacher_cfgs.py --full-episode):
bball7 / soccer15 / act22 _msexp_fullep = their msexp twins with exactly two env
edits (rolloutLength 1000, minEpisodeFrames 30), train cfg = name only, launcher
guards both keys. The longest activity clip (soccer, 677 frames) must fit the window.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_egoexo_fullep_cfgs.py -q
"""
import csv
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
NAMES = ["bball7", "soccer15", "act22"]


def _gen():
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts/make_gen4_egoexo_teacher_cfgs.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    return mk


mk = _gen()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def differing_keys(a, b):
    fa, fb = flatten(a), flatten(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


@pytest.mark.parametrize("name", NAMES)
def test_env_is_the_msexp_twin_plus_two_keys(name):
    a = yaml.safe_load(open(CFG / f"omomo_teacher_{mk.arm(name)}.yaml"))
    b = yaml.safe_load(open(CFG / f"omomo_teacher_{mk.arm(name, True)}.yaml"))
    assert differing_keys(a, b) == {"env.rolloutLength", "env.minEpisodeFrames"}, name
    assert b["env"]["rolloutLength"] == mk.FULLEP_ROLLOUT == 1000
    assert b["env"]["minEpisodeFrames"] == mk.FULLEP_MIN_FRAMES == 30
    assert b["env"]["physicalBufferSize"] == 3 and b["env"]["stateInit"] == "Hybrid"
    assert b["env"]["rewardTerms"]["freeFlightGate"] == a["env"]["rewardTerms"]["freeFlightGate"]   # the ball gate untouched
    assert not {"sub4", "sub10", "sub13", "sub16"} & set(b["env"]["subjectBodies"])


@pytest.mark.parametrize("name", NAMES)
def test_train_cfg_name_only(name):
    a = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{mk.arm(name)}.yaml"))
    b = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{mk.arm(name, True)}.yaml"))
    assert differing_keys(a, b) == {"params.config.full_experiment_name"}, name
    assert b["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{mk.arm(name, True)}"


@pytest.mark.parametrize("name", NAMES)
def test_launcher(name):
    txt = (REPO / f"slurm_teacher_{mk.arm(name, True)}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert f"omomo_teacher_{mk.arm(name, True)}.yaml" in code and mk.arm(name) + ".yaml" not in code
    assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(mk.arm(name, True)), txt, flags=re.M)
    assert "rolloutLength:\\s*1000" in code and "minEpisodeFrames:\\s*30" in code
    assert "FULL EPISODES" in code and "rollout 50" not in code
    assert "motionScaleReward" in code and "_inertial.xml" in code                       # msexp + gen4 guards kept
    assert 'HELDOUT="sub4 sub10 sub13 sub16"' in txt
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{mk.arm(name, True)}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_longest_activity_clip_fits_the_window():
    """From the local manifests (the clip lengths the converters installed)."""
    longest = 0
    for m in (Path.home() / "cari4d_bball7/manifest.csv", Path.home() / "cari4d_soccer/manifest.csv"):
        if not m.is_file():
            pytest.fail(f"manifest not found: {m}")
        longest = max(longest, *(int(r["n_frames"]) for r in csv.DictReader(open(m))))
    assert longest == 677 < mk.FULLEP_ROLLOUT

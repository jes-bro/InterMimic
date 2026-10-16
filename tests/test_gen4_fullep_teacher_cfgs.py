#!/usr/bin/env python3
"""The gen4 FULL-EPISODE teacher arms (make_gen4_activity_teacher_cfgs.py
--motion-scale --full-episode [--single-object]): each is its msexp twin with
exactly two env edits -- rolloutLength 700 (past the longest OMOMO clip) and
minEpisodeFrames 30 (starts anywhere, episodes to the clip's end, PSI live) --
the train cfg unchanged but for the name, and a launcher that guards both keys.

Also pins, on the REAL chair clip lengths (~/new_one/OMOMO_new or the cluster
clone), what the two settings do to PSI and start coverage: the 50-window keeps
every clip eligible; 700 alone kills PSI; 700 + 30 keeps every clip eligible
with more cells than the 50-window and pins no clip to frame 0.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gen4_fullep_teacher_cfgs.py -q
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
GROUPS = ["boxes", "stands", "chairs", "tables", "all"]
OBJECTS = ["clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
           "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


mk = _load("mk", "scripts/make_gen4_activity_teacher_cfgs.py")


def twin(name): return f"gen4_omomo_geoall_{name}_xf_nvadlr_nopose_msexp__f0"
def fullep(name): return f"gen4_omomo_geoall_{name}_xf_nvadlr_nopose_msexp_fullep__f0"


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


@pytest.mark.parametrize("name", GROUPS + OBJECTS)
def test_env_is_the_msexp_twin_plus_exactly_two_keys(name):
    a = yaml.safe_load(open(CFG / f"omomo_teacher_{twin(name)}.yaml"))
    b = yaml.safe_load(open(CFG / f"omomo_teacher_{fullep(name)}.yaml"))
    assert differing_keys(a, b) == {"env.rolloutLength", "env.minEpisodeFrames"}, name
    assert b["env"]["rolloutLength"] == mk.FULLEP_ROLLOUT == 1000      # = the eval window
    assert b["env"]["minEpisodeFrames"] == mk.FULLEP_MIN_FRAMES == 30
    assert b["env"]["physicalBufferSize"] == 3 and b["env"]["stateInit"] == "Hybrid"      # PSI + hybrid init kept


@pytest.mark.parametrize("name", GROUPS + OBJECTS)
def test_train_cfg_name_only(name):
    a = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{twin(name)}.yaml"))
    b = yaml.safe_load(open(CFG / "train/rlg" / f"omomo_teacher_{fullep(name)}.yaml"))
    assert differing_keys(a, b) == {"params.config.full_experiment_name"}, name
    assert b["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{fullep(name)}"


@pytest.mark.parametrize("name", GROUPS + OBJECTS)
def test_launcher_guards_both_keys_and_names_itself(name):
    txt = (REPO / f"slurm_teacher_{fullep(name)}.sh").read_text()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert f"omomo_teacher_{fullep(name)}.yaml" in code and twin(name) + ".yaml" not in code
    assert re.search(r'^#SBATCH --job-name="tch-' + re.escape(fullep(name)), txt, flags=re.M)
    assert "rolloutLength:\\s*1000" in code and "minEpisodeFrames:\\s*30" in code
    assert "FULL EPISODES" in code and "rollout 50" not in code
    assert "motionScaleReward" in code and "raggedMotionData" in code                    # the msexp guards kept
    r = subprocess.run(["bash", "-n", str(REPO / f"slurm_teacher_{fullep(name)}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_full_episode_requires_motion_scale():
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_gen4_activity_teacher_cfgs.py"), "--full-episode"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "motion-scale" in (r.stdout + r.stderr)


def test_longest_omomo_clip_fits_the_window():
    """rolloutLength must exceed every training clip or the window would still cut episodes."""
    cands = [os.environ.get("OMOMO_NEW"), "~/new_one/OMOMO_new", str(REPO / "InterAct/OMOMO_new")]
    root = next((Path(os.path.expanduser(c)) for c in cands if c and Path(os.path.expanduser(c)).is_dir()
                 and sum(1 for x in os.listdir(os.path.expanduser(c)) if x.endswith(".pt")) == 4421), None)
    if root is None:
        pytest.fail("real OMOMO_new (4421 clips) not found; this test runs on the real data")
    import torch
    longest = 0
    for f in os.listdir(root):
        if f.endswith(".pt") and int(f.split("_")[0][3:]) in (1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17):
            longest = max(longest, torch.load(root / f, map_location="cpu", weights_only=False).shape[0])
    assert longest < mk.FULLEP_ROLLOUT, (longest, mk.FULLEP_ROLLOUT)


def test_psi_and_start_coverage_on_the_real_chair_clips():
    pu = _load("pu", "isaacgym/src/intermimic/utils/psi_update.py")
    import torch
    cands = [os.environ.get("OMOMO_NEW"), "~/new_one/OMOMO_new", str(REPO / "InterAct/OMOMO_new")]
    root = next((Path(os.path.expanduser(c)) for c in cands if c and Path(os.path.expanduser(c)).is_dir()
                 and sum(1 for x in os.listdir(os.path.expanduser(c)) if x.endswith(".pt")) == 4421), None)
    if root is None:
        pytest.fail("real OMOMO_new (4421 clips) not found; this test runs on the real data")
    mel = torch.tensor([torch.load(root / f, map_location="cpu", weights_only=False).shape[0]
                        for f in os.listdir(root) if f.endswith(".pt") and f.split("_")[-2] in ("whitechair", "woodchair")
                        and int(f.split("_")[0][3:]) in (2, 3, 5, 6, 7, 12, 14, 15, 17)])
    assert len(mel) == 566
    c50 = pu.psi_eligible_cells(mel, 50)
    cR = pu.psi_eligible_cells(mel, mk.FULLEP_ROLLOUT)
    cR_30 = pu.psi_eligible_cells(mel, mk.FULLEP_ROLLOUT, 30)
    assert int((c50 > 0).sum()) == 566 and int((cR > 0).sum()) == 0 and int((cR_30 > 0).sum()) == 566
    assert int(cR_30.sum()) > int(c50.sum())
    assert sum(1 for m in mel if pu.n_valid_starts(int(m), 30) == 1) == 0                        # nothing pinned to frame 0
    assert sum(1 for m in mel if pu.n_valid_starts(int(m), mk.FULLEP_ROLLOUT) == 1) == 566      # the window alone pins everything

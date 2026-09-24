"""retarget_bball7_local.sh: the plan and the guards, without running a solve.

This script builds a reference tree, and a reference tree is what an arm trained
against -- so the failure that matters is not a crash, it is quietly writing the
WRONG tree or over an existing one. DRY=1 prints the plan and exits, which is
what these tests read.

Pinned here:
  * the ablation (W_CONTACT=0) and the real tree (default 10) get DIFFERENT
    default paths, so one can never be mistaken for the other
  * an existing merged tree is refused unless FORCE=1
  * missing motion dir / cfg fail loudly, before hours of CPU
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("scripts", "retarget_bball7_local.sh")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(os.path.join(REPO, SCRIPT), tmp_path / SCRIPT)
    (tmp_path / "InterAct" / "behave_cari4d_bball7_cf2").mkdir(parents=True)
    cfg = tmp_path / "isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_bball7_geoall__f0.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("env:\n  subjectBodies: ['sub1']\n")
    return tmp_path


def run(repo, **env):
    e = dict(os.environ, DRY="1", **env)
    return subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)


def test_default_is_the_real_tree(repo):
    out = run(repo).stdout
    assert "w_contact=10" in out
    assert "merged : InterAct/behave_cari4d_bball7_f0_bodymajor" in out
    assert "ABLATION" not in out


def test_ablation_gets_its_own_path_and_says_so(repo):
    """The whole risk is writing the ablation over the real tree."""
    out = run(repo, W_CONTACT="0").stdout
    assert "w_contact=0" in out
    assert "behave_cari4d_bball7_f0_bodymajor_w0" in out
    assert "ABLATION" in out


def test_existing_merged_tree_is_refused(repo):
    (repo / "InterAct" / "behave_cari4d_bball7_f0_bodymajor_w0").mkdir(parents=True)
    r = run(repo, W_CONTACT="0")
    assert r.returncode == 2
    assert "already exists" in r.stderr


def test_force_overrides_the_refusal(repo):
    (repo / "InterAct" / "behave_cari4d_bball7_f0_bodymajor_w0").mkdir(parents=True)
    r = run(repo, W_CONTACT="0", FORCE="1")
    assert r.returncode == 0
    assert "(DRY=1: not running)" in r.stdout


def test_explicit_merged_name_is_honoured(repo):
    out = run(repo, W_CONTACT="0", MERGED="InterAct/my_tree").stdout
    assert "merged : InterAct/my_tree" in out


def test_missing_motion_dir_fails_loudly(repo):
    shutil.rmtree(repo / "InterAct" / "behave_cari4d_bball7_cf2")
    r = run(repo, W_CONTACT="0")
    assert r.returncode == 2 and "no motion dir" in r.stderr


def test_missing_cfg_fails_loudly(repo):
    os.remove(repo / "isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_bball7_geoall__f0.yaml")
    r = run(repo, W_CONTACT="0")
    assert r.returncode == 2 and "no cfg" in r.stderr


def test_real_tree_refuses_regressed_solves(repo):
    """allow-worse 0: a regression means an under-converged solve, and writing it
    would hand training a reference worse than no retargeting at all."""
    out = run(repo).stdout
    assert "--allow-worse-cm 0" in out


def test_ablation_keeps_regressed_solves(repo):
    """The ablation is SUPPOSED to solve contacts badly. Refusing those pairs would
    keep only the ones uniform weighting happened to do well on -- a cherry-picked
    tree that understates the cost of dropping contact weighting."""
    out = run(repo, W_CONTACT="0").stdout
    assert "--allow-worse-cm 1000" in out
    assert "cherry-pick" in out


def test_allow_worse_can_be_overridden(repo):
    out = run(repo, W_CONTACT="0", ALLOW_WORSE_CM="0.05").stdout
    assert "--allow-worse-cm 0.05" in out

"""retarget_heldout_bodies.sh: the guards, via DRY=1.

This script adds bodies to the tree the STUDENTS ALREADY USE, so the failure that
matters is touching that tree wrongly -- or discovering a missing input nine
sources into a many-hour run. Pinned here:

  * a missing MJCF for any target body fails BEFORE any solve, naming the body
  * a missing tree or motion dir fails loudly
  * BODIES is required (no accidental default set of people)
  * the worker count is capped, so `nproc` on a 255-core box cannot spawn 253
    torch processes and take the machine down
  * allow-worse stays 0: this is a real tree, where a regression means an
    under-converged solve, unlike the ablation tree
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("scripts", "retarget_heldout_bodies.sh")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(os.path.join(REPO, SCRIPT), tmp_path / SCRIPT)
    (tmp_path / "InterAct" / "OMOMO_retarget_contact_srcall13" / "sub2").mkdir(parents=True)
    motion = tmp_path / "InterAct" / "OMOMO_new"
    motion.mkdir(parents=True)
    for s in ["sub1", "sub2"]:
        (motion / f"{s}_largetable_000.pt").write_text("")
    mjcf = tmp_path / "isaacgym/src/intermimic/data/assets/smplx"
    mjcf.mkdir(parents=True)
    for b in ["sub300", "sub301"]:
        (mjcf / f"smplx_omomo_{b}.xml").write_text("")
    return tmp_path


def run(repo, **env):
    e = dict(os.environ, DRY="1", **env)
    return subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)


def test_plan_lists_bodies_sources_and_existing_counts(repo):
    out = run(repo, BODIES="sub300 sub301").stdout
    assert "sub300 sub301" in out
    assert "sub1 sub2 sub3" in out                 # the 13 OMOMO sources by default
    assert "0 reference file(s) already present" in out


def test_missing_mjcf_fails_before_any_solve(repo):
    r = run(repo, BODIES="sub300 sub999")
    assert r.returncode == 2
    assert "sub999" in r.stderr and "generate_per_subject_mjcfs" in r.stderr


def test_bodies_is_required(repo):
    e = dict(os.environ, DRY="1")
    e.pop("BODIES", None)
    r = subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)
    assert r.returncode != 0 and "BODIES" in r.stderr


def test_missing_tree_fails_loudly(repo):
    shutil.rmtree(repo / "InterAct" / "OMOMO_retarget_contact_srcall13")
    r = run(repo, BODIES="sub300")
    assert r.returncode == 2 and "no tree" in r.stderr


def test_missing_motion_dir_fails_loudly(repo):
    shutil.rmtree(repo / "InterAct" / "OMOMO_new")
    r = run(repo, BODIES="sub300")
    assert r.returncode == 2 and "no motion dir" in r.stderr


def test_workers_are_capped(repo):
    """253 torch processes on a 255-core box is how you take the machine down."""
    out = run(repo, BODIES="sub300").stdout
    workers = int(out.split("workers :")[1].split()[0])
    assert workers <= 128


def test_workers_overridable(repo):
    out = run(repo, BODIES="sub300", WORKERS="8").stdout
    assert "workers : 8" in out


def test_allow_worse_defaults_to_zero(repo):
    """A real tree refuses regressions; only the ablation tree keeps them."""
    out = run(repo, BODIES="sub300").stdout
    assert "allow-worse: 0 cm" in out


def test_sources_overridable(repo):
    out = run(repo, BODIES="sub300", SOURCES="sub1").stdout
    assert "sources : sub1\n" in out


# --- source MJCF resolution -------------------------------------------------
# retarget_contact resolves a bare source id to smplx_omomo_<id>.xml. That is
# right for OMOMO sources and wrong for the CARI4D ones (sub401 is
# smplh_behave_sub401.xml), so the driver passes --source-mjcf when the
# convention's file is absent -- and refuses when neither exists, rather than
# solving every clip against the wrong source body.

def test_cari4d_source_passes_its_smplh_mjcf_explicitly(repo):
    """sub401's body is smplh_behave_sub401.xml; without --source-mjcf the solver
    would look for smplx_omomo_sub401.xml. Run without DRY so the loop is reached
    (it then fails on the absent retarget_contact.py, which is fine -- what is
    asserted is the flag, not the solve)."""
    motion = repo / "InterAct" / "behave_cari4d_act"
    motion.mkdir(parents=True)
    (motion / "sub401_ball_000.pt").write_text("")
    mjcf = repo / "isaacgym/src/intermimic/data/assets/smplx"
    (mjcf / "smplh_behave_sub401.xml").write_text("")
    e = dict(os.environ, BODIES="sub300", SOURCES="sub401",
             MOTION_DIR="InterAct/behave_cari4d_act")
    e.pop("DRY", None)
    r = subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)
    assert "--source-mjcf" in r.stdout
    assert "smplh_behave_sub401.xml" in r.stdout
    assert "no MJCF for SOURCE" not in r.stderr


def test_omomo_source_passes_no_source_mjcf_flag(repo):
    """sub1 IS smplx_omomo_sub1.xml, so the convention is right and no flag is
    added -- the OMOMO path must not change."""
    mjcf = repo / "isaacgym/src/intermimic/data/assets/smplx"
    (mjcf / "smplx_omomo_sub1.xml").write_text("")
    e = dict(os.environ, BODIES="sub300", SOURCES="sub1")
    e.pop("DRY", None)
    r = subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)
    assert "--source-mjcf" not in r.stdout


def test_source_with_no_mjcf_at_all_fails_loudly(repo, tmp_path):
    """Fails at the source, naming it -- not silently retargeting from the wrong body."""
    motion = repo / "InterAct" / "behave_cari4d_act"
    motion.mkdir(parents=True)
    (motion / "sub999_ball_000.pt").write_text("")
    e = dict(os.environ, BODIES="sub300", SOURCES="sub999",
             MOTION_DIR="InterAct/behave_cari4d_act")
    r = subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)
    assert r.returncode == 2
    assert "SOURCE sub999" in r.stderr

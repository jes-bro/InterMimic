"""run_vanilla_indist_local.sh: the plan it prints, without running any eval.

The script deals 13 bodies onto N GPUs and launches one eval per body. What is
worth pinning is the DEALING and the SKIPPING -- the parts that decide whether a
sweep does the right work, and the parts that quietly ruin a sweep if wrong:

  * every body is assigned exactly once (none dropped, none scored twice)
  * the bodies spread evenly over the GPUs rather than piling on one
  * a body whose CSV already exists is SKIPPED, so a re-run after a crash fills
    gaps instead of redoing finished work (and cannot double-write a CSV)
  * a missing input fails loudly instead of launching jobs that die in a rollout

DRY=1 makes this testable: the script prints its plan and exits before any GPU
work, so these run anywhere, no Isaac Gym needed.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("scripts", "run_vanilla_indist_local.sh")
OMOMO13 = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
           "sub11", "sub12", "sub14", "sub15", "sub17"]


@pytest.fixture
def repo(tmp_path):
    """A skeleton repo holding only what the script checks for."""
    for rel in ["checkpoints/vanilla/student.pth",
                "isaacgym/src/intermimic/data/cfg/vanilla_intermimic_eval.yaml",
                "isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all.yaml",
                "slurm_eval_curriculum.sh"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(os.path.join(REPO, SCRIPT), tmp_path / SCRIPT)
    return tmp_path


def run(repo, **env):
    e = dict(os.environ, DRY="1", **env)
    r = subprocess.run(["sh", SCRIPT], cwd=repo, env=e,
                       capture_output=True, text=True)
    return r


def plan_lines(out):
    """{body: (gpu, skipped)} from the printed plan."""
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "gpu":
            got[parts[2]] = (parts[1], "SKIP" in line)
    return got


def test_every_body_assigned_exactly_once(repo):
    got = plan_lines(run(repo, GPUS="4 5 6 7").stdout)
    assert sorted(got) == sorted(OMOMO13)


def test_bodies_spread_evenly_over_gpus(repo):
    got = plan_lines(run(repo, GPUS="4 5 6 7").stdout)
    counts = {}
    for gpu, _ in got.values():
        counts[gpu] = counts.get(gpu, 0) + 1
    assert sorted(counts) == ["4", "5", "6", "7"]
    # 13 bodies over 4 GPUs: 4/3/3/3, never 13/0/0/0
    assert max(counts.values()) - min(counts.values()) <= 1


def test_one_gpu_takes_everything(repo):
    got = plan_lines(run(repo, GPUS="4").stdout)
    assert {g for g, _ in got.values()} == {"4"}
    assert len(got) == 13


def test_existing_csv_is_skipped(repo):
    (repo / "eval_results").mkdir(exist_ok=True)
    (repo / "eval_results" / "vanilla_indist_sub5.csv").write_text("body,source\n")
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is True                      # skipped
    assert got["sub1"][1] is False                     # others still run


def test_missing_input_fails_loudly(repo):
    os.remove(repo / "checkpoints/vanilla/student.pth")
    r = run(repo, GPUS="4")
    assert r.returncode == 2
    assert "missing checkpoints/vanilla/student.pth" in r.stderr


def test_body_and_source_overrides(repo):
    r = run(repo, GPUS="4", BODIES="sub1 sub2", SOURCES="sub3")
    got = plan_lines(r.stdout)
    assert sorted(got) == ["sub1", "sub2"]
    assert "sources    : sub3" in r.stdout


def test_dry_run_launches_nothing(repo):
    r = run(repo, GPUS="4 5 6 7")
    assert "(DRY=1: not running)" in r.stdout
    assert not (repo / "vanilla_indist-sub1.log").exists()

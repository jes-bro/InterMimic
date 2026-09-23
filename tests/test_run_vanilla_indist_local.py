"""run_vanilla_indist_local.sh: the plan it prints, without running any eval.

The script deals 13 bodies onto N GPUs and launches one eval per body. What is
worth pinning is the DEALING and the SKIPPING -- the parts that decide whether a
sweep does the right work, and the parts that quietly ruin a sweep if wrong:

  * every body is assigned exactly once (none dropped, none scored twice)
  * the bodies spread evenly over the GPUs rather than piling on one
  * a body whose CSV is COMPLETE is SKIPPED, so a re-run after a crash (or to add
    GPUs as they free up) fills gaps instead of redoing finished work -- while a
    startup failure's full CSV of exit_code=1 rows, and a killed sweep's partial
    CSV, are both redone rather than taken as results
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


HEADER = ("body,source,is_identity,avg_steps,human_pose_error,object_pose_error,"
          "success_rate,success_count,success_total,exit_code,timed_out,checkpoint\n")
SCORED = "sub5,sub1,False,120.0,0.11,0.09,75.0,3,4,0,False,ckpt.pth\n"
FAILED = "sub5,sub1,False,,,,,,,1,False,ckpt.pth\n"


def _csv(repo, body, rows):
    (repo / "eval_results").mkdir(exist_ok=True)
    (repo / "eval_results" / f"vanilla_indist_{body}.csv").write_text(HEADER + rows)


def test_complete_csv_is_skipped(repo):
    """13 sources -> 13 rows, at least one scored."""
    _csv(repo, "sub5", SCORED * 13)
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is True                      # skipped
    assert got["sub1"][1] is False                     # others still run


def test_partial_csv_from_a_killed_sweep_is_redone(repo):
    """A sweep killed mid-body leaves a FEW scored rows. That body is not done:
    skipping it would freeze it at 1/13 pairs and the mean would quietly be over
    one source instead of thirteen."""
    _csv(repo, "sub5", SCORED)
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is False


def test_completeness_follows_the_source_count(repo):
    """With one source, one row IS complete -- the bar is per-sweep, not a constant."""
    _csv(repo, "sub5", SCORED)
    got = plan_lines(run(repo, GPUS="4", SOURCES="sub1").stdout)
    assert got["sub5"][1] is True


def test_all_failure_csv_is_not_treated_as_done(repo):
    """A job that dies at startup writes a FULL csv (13 rows) of exit_code=1 rows
    with empty metrics -- what a missing MJCF did. Row count alone would accept it,
    so the scored-row half of the check is what rejects it."""
    _csv(repo, "sub5", FAILED * 13)
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is False                     # re-run, not skipped


def test_finished_body_with_some_failed_pairs_counts_as_done(repo):
    """A COMPLETE body whose individual pairs partly failed is done: those failures
    are recorded as exit_code=1 rows and are visible to the summary, so re-running
    would only redo work and could half-overwrite the file."""
    _csv(repo, "sub5", FAILED * 12 + SCORED)
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is True


def test_empty_csv_is_not_treated_as_done(repo):
    _csv(repo, "sub5", "")
    got = plan_lines(run(repo, GPUS="4 5").stdout)
    assert got["sub5"][1] is False


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

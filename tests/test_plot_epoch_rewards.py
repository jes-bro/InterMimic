"""Fixture tests for scripts/plot_epoch_rewards.py.

The bug these pin down: the summary line formatted the epoch bounds with `:d`,
which raises on a float array. Only the .out-log path yields ints -- the
tensorboard path and --auto-epochs both produce floats -- so the crash appeared
only once a run of the other kind happened to come second in the list, after a
row had already printed successfully.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "plot_epoch_rewards.py"


def write_log(path, epochs, reward=1.0, step=100):
    """A slurm .out log in the format the parser expects."""
    lines = [f"epoch_num:{e} mean_rewards:[{reward + e / 10000:.3f}] fps step: 12345"
             for e in epochs]
    path.write_text("\n".join(lines) + "\n")
    return path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          cwd=REPO, capture_output=True, text=True)


def test_summary_prints_for_a_single_run(tmp_path):
    write_log(tmp_path / "teacher-alpha-1.out", range(100, 5100, 100))
    r = run("--glob", str(tmp_path / "*.out"), "--out", tmp_path / "o.png")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.png").exists()
    assert "alpha" in r.stdout


def test_two_runs_both_print(tmp_path):
    """The crash only showed up on the SECOND row, after one had succeeded."""
    write_log(tmp_path / "teacher-alpha-1.out", range(100, 5100, 100))
    write_log(tmp_path / "teacher-beta-2.out", range(100, 5100, 100), reward=2.0)
    r = run("--glob", str(tmp_path / "*.out"), "--out", tmp_path / "o.png")
    assert r.returncode == 0, r.stderr
    assert "alpha" in r.stdout and "beta" in r.stdout


def test_float_epochs_do_not_crash_the_summary(tmp_path):
    """--auto-epochs divides by a stride, making the epoch array float."""
    write_log(tmp_path / "teacher-alpha-1.out", range(65536, 65536 * 60, 65536))
    r = run("--glob", str(tmp_path / "*.out"), "--auto-epochs",
            "--out", tmp_path / "o.png")
    assert r.returncode == 0, r.stderr
    assert "Unknown format code" not in r.stderr
    assert "alpha" in r.stdout


def test_relative_mode_runs(tmp_path):
    write_log(tmp_path / "teacher-alpha-1.out", range(13000, 18000, 100))
    r = run("--glob", str(tmp_path / "*.out"), "--relative",
            "--out", tmp_path / "o.png")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.png").exists()


def test_no_matching_logs_is_reported(tmp_path):
    r = run("--glob", str(tmp_path / "nothing-*.out"), "--out", tmp_path / "o.png")
    assert r.returncode != 0
    assert not (tmp_path / "o.png").exists()

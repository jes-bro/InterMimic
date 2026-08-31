"""Fixture tests for scripts/checkpoint_epoch.py.

A rolling mimic.pth carries no epoch in its name, so two runs' "finals" can be
40,000 epochs apart and look identical on disk. This reads the number out of the
file. What matters is that a checkpoint it cannot read says so instead of
printing a plausible zero.
"""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "checkpoint_epoch.py"


def run(*paths):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, paths)],
                          capture_output=True, text=True)


def save(path, **kw):
    torch.save(kw, path)
    return path


def test_reads_epoch_frame_and_reward(tmp_path):
    p = save(tmp_path / "mimic.pth", epoch=54600, frame=3578265600,
             last_mean_rewards=7.421, model={})
    r = run(p)
    assert r.returncode == 0, r.stderr
    assert "54600" in r.stdout
    assert "3,578,265,600" in r.stdout
    assert "7.421" in r.stdout


def test_tensor_scalars_are_unwrapped(tmp_path):
    """rl_games stores last_mean_rewards as a 0-d tensor in some versions."""
    p = save(tmp_path / "mimic.pth", epoch=100, frame=200,
             last_mean_rewards=torch.tensor(3.5))
    r = run(p)
    assert r.returncode == 0, r.stderr
    assert "3.500" in r.stdout
    # Check the value columns only -- tmp_path carries the test name, which
    # itself contains "tensor", so scanning the whole line proves nothing.
    data = [l for l in r.stdout.splitlines() if l.strip().startswith("100")][0]
    assert data.split()[:3] == ["100", "200", "3.500"]


def test_missing_file_is_reported_not_crashed(tmp_path):
    r = run(tmp_path / "nope.pth")
    assert r.returncode == 1
    assert "[MISSING]" in r.stdout


def test_checkpoint_without_the_keys_says_so(tmp_path):
    p = save(tmp_path / "weights.pth", model={}, optimizer={})
    r = run(p)
    assert r.returncode == 1
    assert "none of" in r.stdout
    assert "model" in r.stdout          # shows what the file does contain


def test_partial_keys_print_dashes_not_zeros(tmp_path):
    """A missing field must not be filled in with a plausible number."""
    p = save(tmp_path / "mimic.pth", epoch=1200, model={})
    r = run(p)
    assert r.returncode == 0, r.stderr
    assert "1200" in r.stdout
    assert "--" in r.stdout


def test_several_checkpoints_in_one_call(tmp_path):
    a = save(tmp_path / "a.pth", epoch=10, frame=1, last_mean_rewards=1.0)
    b = save(tmp_path / "b.pth", epoch=20, frame=2, last_mean_rewards=2.0)
    r = run(a, b)
    assert r.returncode == 0, r.stderr
    assert "10" in r.stdout and "20" in r.stdout
    assert r.stdout.count("mimic") == 0          # both named a/b, sanity on paths

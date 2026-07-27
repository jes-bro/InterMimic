"""Fixture test for scripts/plot_tb_rewards.py.

Writes tiny synthetic TensorBoard event files with KNOWN scalar values (no
Isaac Gym / rl_games needed), then checks that the loader reads back exactly
what was written and that main() produces a PNG end-to-end.

Run:  python -m pytest tests/test_plot_tb_rewards.py -q
"""

import os
import sys

import numpy as np
import pytest

# The scripts/ dir is not a package; import by path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import plot_tb_rewards  # noqa: E402

from tensorboard.compat.proto.event_pb2 import Event  # noqa: E402
from tensorboard.compat.proto.summary_pb2 import Summary  # noqa: E402
from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: E402


def write_run(run_dir, tag, points):
    """Write an event file with (step, wall_time, value) triples for one tag."""
    os.makedirs(run_dir, exist_ok=True)
    w = EventFileWriter(run_dir)
    for step, wall, value in points:
        ev = Event(wall_time=wall, step=step)
        ev.summary.value.append(Summary.Value(tag=tag, simple_value=value))
        w.add_event(ev)
    w.close()


@pytest.fixture
def two_runs(tmp_path):
    """Two runs with distinct, known reward curves (like the normval A/B)."""
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    # run_a: rewards 1,2,3 at frames 100,200,300; wall times 10s apart
    write_run(str(a), "rewards0/frame",
              [(100, 1000.0, 1.0), (200, 1010.0, 2.0), (300, 1020.0, 3.0)])
    # run_b: shorter run, higher rewards
    write_run(str(b), "rewards0/frame",
              [(100, 2000.0, 2.0), (200, 2015.0, 4.0)])
    return str(a), str(b)


def test_load_scalar_roundtrip(two_runs):
    run_a, _ = two_runs
    steps, walls, vals = plot_tb_rewards.load_scalar(run_a, "rewards0/frame")
    np.testing.assert_array_equal(steps, [100, 200, 300])
    np.testing.assert_array_equal(walls, [1000.0, 1010.0, 1020.0])
    np.testing.assert_array_equal(vals, [1.0, 2.0, 3.0])


def test_load_scalar_fails_loud_on_missing(two_runs, tmp_path):
    run_a, _ = two_runs
    with pytest.raises(FileNotFoundError):
        plot_tb_rewards.load_scalar(str(tmp_path / "nope"), "rewards0/frame")
    with pytest.raises(ValueError, match="not in"):
        plot_tb_rewards.load_scalar(run_a, "typo0/frame")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((ValueError, Exception)):
        plot_tb_rewards.load_scalar(str(empty), "rewards0/frame")


def test_ema_matches_tensorboard_formula():
    vals = np.array([1.0, 2.0, 3.0])
    out = plot_tb_rewards.ema(vals, 0.5)
    # acc starts at vals[0]: 1.0 -> .5*1+.5*1=1.0 -> .5*1+.5*2=1.5 -> .5*1.5+.5*3=2.25
    np.testing.assert_allclose(out, [1.0, 1.5, 2.25])
    # weight 0 = no smoothing
    np.testing.assert_array_equal(plot_tb_rewards.ema(vals, 0.0), vals)


def test_main_end_to_end(two_runs, tmp_path, capsys):
    run_a, run_b = two_runs
    out = tmp_path / "plot.png"
    rc = plot_tb_rewards.main([
        "--run", f"A={run_a}", "--run", f"B={run_b}",
        "--smoothing", "0.0", "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    text = capsys.readouterr().out
    # common frame budget is min(300, 200) = 200; unsmoothed values there: A=2, B=4
    assert "0.00B" in text  # 200 frames prints as 0.00B
    assert "2.000" in text and "4.000" in text

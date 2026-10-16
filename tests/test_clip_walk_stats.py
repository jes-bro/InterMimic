"""clip_walk_stats.py: walking metrics from a pelvis track, on synthetic clips
with known motion, plus the file-level collector on a tiny 591-channel fixture.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_clip_walk_stats.py -q
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("cws", os.path.join(REPO, "scripts", "clip_walk_stats.py"))
cws = importlib.util.module_from_spec(spec); spec.loader.exec_module(cws)


def _track(speed_mps, seconds, fps=30.0):
    """Straight-line pelvis track at a constant speed along x."""
    T = int(seconds * fps)
    t = np.arange(T) / fps
    return np.stack([speed_mps * t, np.zeros(T)], axis=1)


def test_standing_clip_has_no_walking():
    s = cws.walk_stats(np.zeros((90, 2)))
    assert s["frames"] == 90 and s["path_m"] == 0 and s["net_m"] == 0
    assert s["walk_share"] == 0 and s["walk_s"] == 0 and s["longest_walk_s"] == 0


def test_constant_walk_is_all_walking_and_path_equals_net():
    s = cws.walk_stats(_track(1.0, 4.0))        # 1 m/s for 4 s
    assert s["frames"] == 120
    assert s["net_m"] == pytest.approx(119 / 30, abs=1e-9)
    assert s["path_m"] == pytest.approx(s["net_m"], abs=1e-9)   # straight line
    assert s["walk_share"] == 1.0 and s["walk_s"] == pytest.approx(4.0)
    assert s["longest_walk_s"] == pytest.approx(4.0)
    assert s["peak_speed"] == pytest.approx(1.0, rel=1e-6)


def test_walk_then_stand_then_walk_gives_longest_run_not_total():
    walk, stand = _track(1.0, 2.0), np.zeros((60, 2))
    xy = np.concatenate([walk, walk[-1] + stand, walk[-1] + _track(1.0, 1.0)], axis=0)
    s = cws.walk_stats(xy)
    assert s["walk_s"] == pytest.approx(3.0, abs=0.2)          # 2 s + 1 s, +-window edges
    assert s["longest_walk_s"] == pytest.approx(2.0, abs=0.2)  # the first run, not 3 s
    assert 0.5 < s["walk_share"] < 0.7


def test_slow_drift_below_threshold_is_not_walking():
    s = cws.walk_stats(_track(0.1, 3.0), walk_speed=0.3)
    assert s["walk_share"] == 0 and s["path_m"] > 0.2         # it moved, but did not walk


def test_collect_reads_591_channel_clips_and_filters(tmp_path):
    def clip(name, speed, seconds):
        x = torch.zeros(int(seconds * 30), 591)
        x[:, 0:2] = torch.tensor(_track(speed, seconds), dtype=torch.float32)
        torch.save(x, tmp_path / f"{name}.pt")
    clip("sub6_woodchair_000", 0.0, 2.0)
    clip("sub6_woodchair_001", 1.0, 2.0)
    clip("sub2_largetable_000", 1.0, 2.0)
    (tmp_path / "notes.txt").write_text("ignored")
    rows = cws.collect(str(tmp_path), objects=["woodchair"], sources=["sub6"])
    assert [r["clip"] for r in rows] == ["sub6_woodchair_000", "sub6_woodchair_001"]
    assert rows[0]["walk_share"] == 0 and rows[1]["walk_share"] == 1.0
    assert rows[1]["source"] == "sub6" and rows[1]["object"] == "woodchair"
    assert len(cws.collect(str(tmp_path))) == 3                   # no filter: every stem


def test_bad_channel_count_fails_loudly(tmp_path):
    torch.save(torch.zeros(10, 590), tmp_path / "sub1_box_000.pt")
    with pytest.raises(SystemExit):
        cws.root_xy(str(tmp_path / "sub1_box_000.pt"))

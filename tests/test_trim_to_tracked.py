#!/usr/bin/env python3
"""Tests for trim_to_tracked.py -- run-finding and mask trimming.

The dangerous failure here is silent misalignment: an off-by-one in the run
bounds or the renumbering pairs every mask with the wrong video frame, and the
reconstruction fails in a way that looks like a tracking problem.
"""
import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import trim_to_tracked as T  # noqa: E402


# ------------------------------------------------------------------ find_runs

def test_simple_runs():
    good = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    assert T.find_runs(good) == [(1, 3), (6, 7)]


def test_run_at_both_edges():
    """A run touching frame 0 or the last frame must not be dropped."""
    assert T.find_runs(np.array([1, 1, 0, 1, 1], dtype=bool)) == [(0, 1), (3, 4)]


def test_all_good_is_one_run():
    assert T.find_runs(np.ones(5, dtype=bool)) == [(0, 4)]


def test_none_good_is_no_runs():
    assert T.find_runs(np.zeros(5, dtype=bool)) == []


def test_gap_tolerance_bridges_short_dropout():
    good = np.array([1, 1, 0, 1, 1], dtype=bool)          # single-frame hole
    assert T.find_runs(good, gap_tolerance=0) == [(0, 1), (3, 4)]
    assert T.find_runs(good, gap_tolerance=1) == [(0, 4)]


def test_gap_tolerance_does_not_bridge_too_long():
    good = np.array([1, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert T.find_runs(good, gap_tolerance=2) == [(0, 1), (5, 6)]
    assert T.find_runs(good, gap_tolerance=3) == [(0, 6)]


def test_run_never_ends_on_a_bad_frame():
    """A bridged run must end on a GOOD frame, not on trailing dropout."""
    good = np.array([1, 1, 0, 0], dtype=bool)
    assert T.find_runs(good, gap_tolerance=5) == [(0, 1)]


# ------------------------------------------------------- h5 reading / trimming

def _make_h5(path, cam, n, person_areas, object_areas, H=4, W=5):
    with h5py.File(path, "w") as f:
        g = f.create_group(cam)
        for i in range(n):
            for kind, areas in ((T.PERSON, person_areas), (T.OBJECT, object_areas)):
                m = np.zeros((H, W), dtype=bool)
                m.reshape(-1)[:areas[i]] = True     # exact pixel count
                g.create_dataset(f"{i:06d}-k0.{kind}", data=m)


def test_read_areas_counts_pixels(tmp_path):
    p = tmp_path / "c.h5"
    _make_h5(str(p), "cam04", 3, [1, 2, 3], [0, 5, 0])
    cam, frames, per, obj, shape = T.read_areas(str(p))
    assert cam == "cam04"
    assert list(frames) == [0, 1, 2]
    assert list(per) == [1, 2, 3] and list(obj) == [0, 5, 0]
    assert shape == (4, 5)


def test_holes_in_frame_numbering_fail_loudly(tmp_path):
    """A missing frame would shift every later mask -- must not be guessed at."""
    p = tmp_path / "c.h5"
    _make_h5(str(p), "cam04", 3, [1, 1, 1], [1, 1, 1])
    with h5py.File(str(p), "a") as f:
        del f["cam04"]["000001-k0.person_mask.png"]
        del f["cam04"]["000001-k0.obj_rend_mask.png"]
    with pytest.raises(ValueError, match="holes"):
        T.read_areas(str(p))


def test_trim_masks_renumbers_from_zero_and_preserves_content(tmp_path):
    src, dst = tmp_path / "s.h5", tmp_path / "d.h5"
    # frames 0..5, object present only on 2,3,4
    _make_h5(str(src), "cam04", 6, [3]*6, [0, 0, 7, 8, 9, 0])
    cam, frames, per, obj, _ = T.read_areas(str(src))
    good = (per >= 1) & (obj >= 1)
    runs = T.find_runs(good)
    assert runs == [(2, 4)]
    n = T.trim_masks(str(src), str(dst), cam, frames, 2, 4)
    assert n == 3
    with h5py.File(str(dst), "r") as f:
        keys = sorted(f["cam04"].keys())
        assert keys == sorted([f"{i:06d}-k0.{k}" for i in range(3)
                               for k in (T.PERSON, T.OBJECT)])
        # content must follow the source frames 2,3,4 in order
        got = [f["cam04"][f"{i:06d}-k0.{T.OBJECT}"][()].sum() for i in range(3)]
        assert got == [7, 8, 9]


def test_thresholds_reject_tiny_blobs(tmp_path):
    """A 2px spurious detection must not count as a tracked object."""
    p = tmp_path / "c.h5"
    _make_h5(str(p), "cam04", 4, [5]*4, [2, 9, 9, 2])
    _, _, per, obj, _ = T.read_areas(str(p))
    assert T.find_runs((per >= 1) & (obj >= 1)) == [(0, 3)]      # no threshold
    assert T.find_runs((per >= 1) & (obj >= 5)) == [(1, 2)]      # thresholded


def test_longest_run_selection_is_by_length(tmp_path):
    good = np.array([1,1,0,1,1,1,1,0,1], dtype=bool)
    runs = sorted(T.find_runs(good), key=lambda r: -(r[1]-r[0]+1))
    assert runs[0] == (3, 6)          # 4 frames, the longest
    assert runs[-1] == (8, 8)         # 1 frame, the shortest

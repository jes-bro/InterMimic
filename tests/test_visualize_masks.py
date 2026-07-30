#!/usr/bin/env python3
"""Tests for visualize_masks.py -- the pure compositing/geometry helpers.

These cover what a rendered frame CANNOT be eyeballed for: that the blend uses
CARI4D's colours and alpha, that an empty mask leaves the frame untouched, that
the bbox is right, and that a one-frame dropout is not averaged away in the
timeline strip. Whether the result LOOKS right is for a human to judge.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import visualize_masks as V  # noqa: E402


def _frame(h=6, w=8, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ------------------------------------------------------------------ composite

def test_empty_masks_leave_frame_untouched():
    f = _frame()
    out = V.composite(f, np.zeros(f.shape[:2], bool), np.zeros(f.shape[:2], bool))
    assert np.array_equal(out, f)


def test_person_blends_toward_red_at_half_alpha():
    f = _frame(val=100)
    m = np.zeros(f.shape[:2], bool); m[0, 0] = True
    out = V.composite(f, m, np.zeros(f.shape[:2], bool))
    # 100*0.5 + 255*0.5 = 177.5 -> 177 ; other channels 100*0.5 + 0 = 50
    assert tuple(out[0, 0]) == (177, 50, 50)
    assert tuple(out[1, 1]) == (100, 100, 100)      # untouched elsewhere


def test_object_blends_toward_blue():
    f = _frame(val=100)
    m = np.zeros(f.shape[:2], bool); m[2, 3] = True
    out = V.composite(f, np.zeros(f.shape[:2], bool), m)
    assert tuple(out[2, 3]) == (50, 50, 177)


def test_object_wins_on_overlap():
    """Overlap must read as the object -- that is the track under suspicion."""
    f = _frame(val=100)
    m = np.zeros(f.shape[:2], bool); m[1, 1] = True
    out = V.composite(f, m.copy(), m.copy())
    # person: 100*.5 + 255*.5 = (177.5, 50, 50); then object on top of that:
    # 177.5*.5 = 88.75, 50*.5 = 25, 50*.5 + 255*.5 = 152.5
    assert tuple(out[1, 1]) == (88, 25, 152)
    assert out[1, 1][2] > out[1, 1][0]         # unambiguously blue-dominant


# ---------------------------------------------------------------- object_bbox

def test_bbox_none_when_empty():
    assert V.object_bbox(np.zeros((5, 5), bool)) is None


def test_bbox_covers_mask_and_pads_within_bounds():
    m = np.zeros((20, 20), bool); m[8:12, 9:13] = True
    x0, y0, x1, y1 = V.object_bbox(m, pad=3)
    assert (x0, y0, x1, y1) == (6, 5, 16, 15)
    m2 = np.zeros((20, 20), bool); m2[0, 0] = True     # clamps at the edge
    assert V.object_bbox(m2, pad=5) == (0, 0, 6, 6)


# -------------------------------------------------------------------- strip

def test_strip_all_good_is_green_all_bad_is_red():
    g = V.timeline_strip(np.ones(50, bool), width=25)
    assert set(map(tuple, g.reshape(-1, 3))) == {(60, 190, 90)}
    r = V.timeline_strip(np.zeros(50, bool), width=25)
    assert set(map(tuple, r.reshape(-1, 3))) == {(150, 30, 30)}


def test_single_frame_dropout_is_not_averaged_away():
    """A column is green only if EVERY frame in it is tracked."""
    good = np.ones(100, bool); good[50] = False
    strip = V.timeline_strip(good, width=10)          # 10 frames per column
    cols = {tuple(strip[0, i]) for i in range(10)}
    assert (150, 30, 30) in cols, "a lost frame vanished from the timeline"


def test_cursor_is_drawn_and_moves():
    good = np.ones(100, bool)
    a = V.timeline_strip(good, width=50, cursor=0)
    b = V.timeline_strip(good, width=50, cursor=99)
    assert (255, 220, 0) in set(map(tuple, a[0]))
    assert not np.array_equal(a, b), "cursor did not move with the frame index"


def test_strip_dimensions():
    s = V.timeline_strip(np.ones(10, bool), width=64, height=7)
    assert s.shape == (7, 64, 3) and s.dtype == np.uint8


# ---------------------------------------------------------------- zoom inset

def test_zoom_inset_magnifies_and_is_square():
    f = _frame(h=40, w=40)
    f[10:14, 10:14] = 200
    ins = V.zoom_inset(f, (10, 10, 14, 14), size=40)
    assert ins.shape == (40, 40, 3)
    assert ins[0, 0].tolist() == [200, 200, 200]   # magnified content, not padding


def test_zoom_inset_handles_degenerate_bbox():
    assert V.zoom_inset(_frame(), (2, 2, 2, 2), size=20) is None

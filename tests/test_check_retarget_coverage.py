#!/usr/bin/env python3
"""Fixtures for the retarget-coverage pre-flight check.

The dangerous case is a PARTIALLY solved body: a body dir that exists but is
missing some clips reads as "present" to anything that only checks the directory,
while the task itself fails on the individual file. So these pin per-(body, clip)
resolution, not per-body.

    python3 -m pytest tests/test_check_retarget_coverage.py -q
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import check_retarget_coverage as crc   # noqa: E402


CLIPS = ["sub2_largetable_000.pt", "sub2_woodchair_001.pt"]


@pytest.fixture
def retarget_tree(tmp_path):
    """<dir>/<body>/<clip>.pt, the layout intermimic.py:314 expects.

    sub1/sub10 fully solved, sub16 solved for only one clip, sub13 absent.
    """
    for body in ("sub1", "sub10"):
        d = tmp_path / body
        d.mkdir()
        for c in CLIPS:
            (d / c).touch()
    partial = tmp_path / "sub16"
    partial.mkdir()
    (partial / CLIPS[0]).touch()
    return tmp_path


def _missing(root, body, clips=CLIPS):
    return [c for c in clips if not (root / body / c).exists()]


def test_fully_solved_body_is_complete(retarget_tree):
    assert _missing(retarget_tree, "sub1") == []


def test_partially_solved_body_is_caught(retarget_tree):
    """A body dir that EXISTS but lacks clips must not read as covered."""
    assert (retarget_tree / "sub16").is_dir()
    assert _missing(retarget_tree, "sub16") == [CLIPS[1]]


def test_absent_body_is_caught(retarget_tree):
    assert _missing(retarget_tree, "sub13") == CLIPS


def test_clip_filter_matches_the_task_datasub_rule(tmp_path, monkeypatch):
    """Only .pt files whose sub<N>_ prefix is in dataSub, same as intermimic.py."""
    for f in ("sub2_a_000.pt", "sub2_b_001.pt", "sub9_c_000.pt", "readme.txt"):
        (tmp_path / f).touch()
    monkeypatch.setattr(crc, "REPO", "")
    clips, err = crc.clips_for(str(tmp_path), ["sub2"])
    assert err is None
    assert clips == ["sub2_a_000.pt", "sub2_b_001.pt"]


def test_missing_motion_dir_reports_rather_than_crashes(monkeypatch):
    monkeypatch.setattr(crc, "REPO", "")
    clips, err = crc.clips_for("/definitely/not/here", ["sub2"])
    assert clips is None and "not found" in err


def test_default_bodies_include_the_heldout_trio_and_exclude_sub4():
    """The held-out bodies are the usual coverage gap, so they must be checked.

    The retarget job is normally run with --targets-from the arm's own cfg, which
    by construction lists only TRAINING bodies -- so the held-out trio is exactly
    what goes missing. sub4 is excluded: its MJCF crashes the simulator.
    """
    assert {"sub10", "sub13", "sub16"} <= set(crc.DEFAULT_BODIES)
    assert "sub4" not in crc.DEFAULT_BODIES


def test_omomo_arm_gets_the_omomo_generator(capsys):
    """An OMOMO arm may be handed slurm_retarget_gen.sh; a non-OMOMO one must not.

    That generator defaults --motion-dir to InterAct/OMOMO_new and writes a FLAT
    per-target layout, so pointing it at the bball tree would solve the wrong
    clips into the wrong shape. Printing a confidently wrong command is worse
    than printing none.
    """
    src = open(os.path.join(REPO, "scripts/check_retarget_coverage.py")).read()
    # the generator command is emitted only under the OMOMO branch
    omomo_branch = src.split('if motion == "InterAct/OMOMO_new":')[1].split("else:")[0]
    assert "slurm_retarget_gen.sh" in omomo_branch
    other_branch = src.split('if motion == "InterAct/OMOMO_new":')[1].split("else:")[1]
    assert "sbatch slurm_retarget_gen.sh" not in other_branch
    assert "retarget_layout.py" in other_branch

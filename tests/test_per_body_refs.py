"""per_body_reference_files: which reference files the InterMimic baseline reads.

Option A scores the paper's student (InterMimic_All) against OUR per-body
contact-retargeted reference, on the SAME clips as the g3 students, so the two
sit in one table. The picking rule is what these tests pin:

  * clips come from motion_file (OMOMO_new), filtered by dataSub -- not from the
    retarget tree, whose authors' subset is smaller
  * every clip resolves to <tree>/<body>/<clip>.pt
  * one body only (an eval runs one (body, source) pair)
  * a missing (body, clip) file is a loud error, never a silent fallback
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "isaacgym", "src"))

from intermimic.utils.per_body_refs import per_body_reference_files  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    """motion dir with 3 sources, and a retarget tree holding 2 bodies."""
    motion = tmp_path / "OMOMO_new"
    motion.mkdir()
    for name in ["sub1_largetable_000.pt", "sub1_largetable_001.pt",
                 "sub2_chair_000.pt", "sub3_box_000.pt", "notes.txt"]:
        (motion / name).write_text("")
    rt = tmp_path / "retarget"
    for body in ["sub10", "sub13"]:
        (rt / body).mkdir(parents=True)
        for name in ["sub1_largetable_000.pt", "sub1_largetable_001.pt",
                     "sub2_chair_000.pt", "sub3_box_000.pt"]:
            (rt / body / name).write_text("")
    return tmp_path, str(motion), str(rt)


def _call(tmp, motion, rt, data_sub, bodies):
    # resolve is identity here: the fixture already uses absolute paths
    return per_body_reference_files(rt, motion, data_sub, bodies, resolve=lambda p: p)


def test_clips_come_from_motion_dir_filtered_by_datasub(tree):
    tmp, motion, rt = tree
    files, clips = _call(tmp, motion, rt, ["sub1"], ["sub10"])
    assert clips == ["sub1_largetable_000.pt", "sub1_largetable_001.pt"]   # sorted, sub1 only
    assert all(f.startswith(os.path.join(rt, "sub10")) for f in files)     # this body's copies
    assert len(files) == len(clips)


def test_paths_are_strings_not_path_objects(tree):
    """The caller splits these as filenames (object name out of the stem), which a
    PosixPath cannot do -- resolve_repo_path returns one, so the helper converts."""
    import pathlib
    tmp, motion, rt = tree
    files, _ = per_body_reference_files(rt, motion, ["sub1"], ["sub10"],
                                        resolve=lambda p: pathlib.Path(p))
    assert all(isinstance(f, str) for f in files)
    assert files[0].split("_")[-2] == "largetable"


def test_non_pt_files_are_ignored(tree):
    tmp, motion, rt = tree
    _, clips = _call(tmp, motion, rt, ["sub1", "sub2", "sub3"], ["sub13"])
    assert "notes.txt" not in clips and len(clips) == 4


def test_body_selects_that_bodys_copy(tree):
    tmp, motion, rt = tree
    f10, _ = _call(tmp, motion, rt, ["sub2"], ["sub10"])
    f13, _ = _call(tmp, motion, rt, ["sub2"], ["sub13"])
    assert f10 != f13
    assert os.path.basename(f10[0]) == os.path.basename(f13[0])            # same clip
    assert os.path.basename(os.path.dirname(f10[0])) == "sub10"
    assert os.path.basename(os.path.dirname(f13[0])) == "sub13"


def test_more_than_one_body_is_refused(tree):
    tmp, motion, rt = tree
    with pytest.raises(ValueError, match="exactly ONE body"):
        _call(tmp, motion, rt, ["sub1"], ["sub10", "sub13"])
    with pytest.raises(ValueError, match="exactly ONE body"):
        _call(tmp, motion, rt, ["sub1"], [])


def test_missing_reference_is_loud(tree):
    tmp, motion, rt = tree
    os.remove(os.path.join(rt, "sub10", "sub1_largetable_001.pt"))
    with pytest.raises(FileNotFoundError) as e:
        _call(tmp, motion, rt, ["sub1"], ["sub10"])
    assert "1 of 2" in str(e.value) and "sub1_largetable_001.pt" in str(e.value)


def test_clip_set_is_the_students_not_the_authors_subset(tree):
    """A clip present in the motion dir AND the tree is scored even if the authors'
    (smaller) release never had it: the list is driven by motion_file."""
    tmp, motion, rt = tree
    open(os.path.join(motion, "sub1_largetable_002.pt"), "w").close()
    for body in ["sub10", "sub13"]:
        open(os.path.join(rt, body, "sub1_largetable_002.pt"), "w").close()
    _, clips = _call(tmp, motion, rt, ["sub1"], ["sub10"])
    assert "sub1_largetable_002.pt" in clips and len(clips) == 3

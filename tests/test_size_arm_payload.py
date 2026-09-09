"""Fixture-based tests for scripts/size_arm_payload.py.

The two behaviours that matter for provisioning a paid VM:

  1. A MISSING path is reported, never counted as zero. An under-reported total
     is how you provision a disk that fills up mid-run.
  2. Symlinks are not counted. The merged trees (src2src6, src1src12src14) are
     symlink farms pointing INTO the single-source trees, so counting the link
     targets would double-count data that is already being staged separately.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import size_arm_payload as S  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    """A tiny retarget tree: 2 bodies x 2 clips of 1000 bytes."""
    root = tmp_path / "OMOMO_retarget_contact_srcX"
    for body in ("sub1", "sub2"):
        d = root / body
        d.mkdir(parents=True)
        for clip in ("subX_obj_000.pt", "subX_obj_001.pt"):
            (d / clip).write_bytes(b"\0" * 1000)
    return root


class TestDirBytes:
    def test_counts_every_regular_file(self, tree):
        total, n = S.dir_bytes(str(tree))
        assert (total, n) == (4000, 4)

    def test_missing_dir_is_none_not_zero(self, tmp_path):
        # None is what lets the caller say MISSING; 0 would silently pass.
        assert S.dir_bytes(str(tmp_path / "nope")) is None

    def test_symlinks_are_not_counted(self, tree, tmp_path):
        merged = tmp_path / "merged"
        (merged / "sub1").mkdir(parents=True)
        os.symlink(tree / "sub1" / "subX_obj_000.pt",
                   merged / "sub1" / "subX_obj_000.pt")
        total, n = S.dir_bytes(str(merged))
        assert (total, n) == (0, 0), "merged-tree symlinks must not double-count"

    def test_real_file_beside_a_symlink_still_counts(self, tree, tmp_path):
        merged = tmp_path / "mixed"
        (merged / "sub1").mkdir(parents=True)
        os.symlink(tree / "sub1" / "subX_obj_000.pt", merged / "sub1" / "linked.pt")
        (merged / "sub1" / "real.pt").write_bytes(b"\0" * 500)
        assert S.dir_bytes(str(merged)) == (500, 1)


class TestClipsBytes:
    @pytest.fixture
    def motion(self, tmp_path):
        d = tmp_path / "OMOMO_new"
        d.mkdir()
        for name, size in [("sub5_a_000.pt", 100), ("sub5_a_001.pt", 100),
                           ("sub6_a_000.pt", 999), ("sub15_a_000.pt", 777)]:
            (d / name).write_bytes(b"\0" * size)
        return d

    def test_only_the_arms_own_sources(self, motion):
        assert S.clips_bytes(str(motion), ["sub5"]) == (200, 2)

    def test_multi_source_sums(self, motion):
        assert S.clips_bytes(str(motion), ["sub5", "sub6"]) == (1199, 3)

    def test_sub5_prefix_does_not_swallow_sub15(self, motion):
        # the classic sub1-vs-sub10 substring trap: the glob is "<src>_*"
        total, n = S.clips_bytes(str(motion), ["sub5"])
        assert n == 2 and total == 200

    def test_missing_motion_dir_is_none(self, tmp_path):
        assert S.clips_bytes(str(tmp_path / "nope"), ["sub5"]) is None


class TestFilesBytes:
    def test_reports_missing_separately(self, tmp_path):
        ok = tmp_path / "there.xml"
        ok.write_bytes(b"\0" * 42)
        total, missing = S.files_bytes([str(ok), str(tmp_path / "gone.xml")])
        assert total == 42
        assert missing == [str(tmp_path / "gone.xml")]

    def test_all_present(self, tmp_path):
        ps = []
        for i in range(3):
            p = tmp_path / f"b{i}.xml"
            p.write_bytes(b"\0" * 10)
            ps.append(str(p))
        assert S.files_bytes(ps) == (30, [])


class TestSizeArm:
    def test_missing_tree_is_flagged_not_silently_zero(self, tmp_path, monkeypatch):
        cfgdir = tmp_path / "cfg"
        cfgdir.mkdir()
        (cfgdir / "omomo_teacher_fake__f0.yaml").write_text(yaml.safe_dump({
            "env": {"retargetedMotionDir": str(tmp_path / "absent_tree"),
                    "motion_file": str(tmp_path / "absent_motion"),
                    "subjectBodies": ["sub1"], "dataSub": ["sub1"]}}))
        monkeypatch.setattr(S, "CFG_DIR", str(cfgdir))
        monkeypatch.setattr(S, "MJCF_DIR", str(tmp_path / "assets"))
        row = S.size_arm("fake__f0", want_checkpoints=False)
        assert any("tree MISSING" in m for m in row["missing"])
        assert row["total"] == 0        # nothing counted, and it SAID so

    def test_unknown_arm_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "CFG_DIR", str(tmp_path))
        assert "error" in S.size_arm("does_not_exist", want_checkpoints=False)

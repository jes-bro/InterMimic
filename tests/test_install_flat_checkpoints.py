"""install_flat_checkpoints.py: name split, tree layout, refusals.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_install_flat_checkpoints.py -q
"""
import importlib.util
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "install_flat_checkpoints.py")
spec = importlib.util.spec_from_file_location("install_flat_checkpoints", SCRIPT)
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)


def test_split_name():
    assert ic.split_name("smplx_teacher_g3_omomo_geoall_src11__f0_mimic.pth") == \
        ("smplx_teacher_g3_omomo_geoall_src11__f0", "mimic.pth")
    assert ic.split_name("smplx_teacher_g3_omomo_geoall__f0_mimic_00030000.pth") == \
        ("smplx_teacher_g3_omomo_geoall__f0", "mimic_00030000.pth")
    assert ic.split_name("mimic.pth") is None            # no experiment prefix
    assert ic.split_name("something.pth") is None


def test_install_copy_and_move(tmp_path):
    src = tmp_path / "dl"
    src.mkdir()
    a = src / "smplx_teacher_g3_omomo_geoall_src3__f0_mimic.pth"
    b = src / "smplx_teacher_g3_omomo_geoall__f0_mimic_00030000.pth"
    a.write_bytes(b"A"); b.write_bytes(b"B")
    root = tmp_path / "ck"
    ic.main([str(a), "--root", str(root)])
    assert (root / "smplx_teacher_g3_omomo_geoall_src3__f0" / "nn" / "mimic.pth").read_bytes() == b"A"
    assert a.exists()                                   # copy keeps the source
    ic.main([str(b), "--root", str(root), "--move"])
    assert (root / "smplx_teacher_g3_omomo_geoall__f0" / "nn" / "mimic_00030000.pth").read_bytes() == b"B"
    assert not b.exists()                               # move removes it
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ic.main([str(a), "--root", str(root)])


def test_bad_name_refused(tmp_path):
    f = tmp_path / "weights.pth"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit, match="refusing to guess"):
        ic.main([str(f), "--root", str(tmp_path / "ck"), "--dry-run"])


def test_dry_run_writes_nothing(tmp_path):
    f = tmp_path / "smplx_teacher_g3_omomo_geoall_src9__f0_mimic.pth"
    f.write_bytes(b"x")
    ic.main([str(f), "--root", str(tmp_path / "ck"), "--dry-run"])
    assert not (tmp_path / "ck").exists()

"""Fixture tests for scripts/trim_pt_start.py: row-slice correctness, the
no-overwrite guard, and the fail-loud paths."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "trim_pt_start.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    # 10-frame clip whose frame index is written into channel 0, so any
    # off-by-one in the slice is visible in the values, not just the length.
    t = torch.zeros(10, 591)
    t[:, 0] = torch.arange(10, dtype=torch.float32)
    torch.save(t, d / "sub100_bball_000.pt")
    (d / "manifest.json").write_text("{}")
    return d


def test_trims_and_copies(src, tmp_path):
    dst = tmp_path / "dst"
    r = run("--src-dir", str(src), "--dst-dir", str(dst), "--start", "4")
    assert r.returncode == 0, r.stderr
    out = torch.load(dst / "sub100_bball_000.pt")
    assert out.shape == (6, 591)
    assert out[0, 0].item() == 4.0          # new frame 0 is old frame 4
    assert out[-1, 0].item() == 9.0         # last frame preserved
    assert (dst / "manifest.json").exists()  # non-.pt files come along


def test_refuses_existing_dst(src, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    r = run("--src-dir", str(src), "--dst-dir", str(dst), "--start", "4")
    assert r.returncode != 0
    assert "refusing to overwrite" in r.stderr


def test_refuses_bad_start(src, tmp_path):
    # start >= clip length
    r = run("--src-dir", str(src), "--dst-dir", str(tmp_path / "a"), "--start", "10")
    assert r.returncode != 0 and "clip length" in r.stderr
    # start 0 is a pure copy, not a trim
    r = run("--src-dir", str(src), "--dst-dir", str(tmp_path / "b"), "--start", "0")
    assert r.returncode != 0

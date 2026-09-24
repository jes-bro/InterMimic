"""add_body_heights.py: additive, and never silently changes a body's height.

bodyNormalizedReward divides the reward by the subject's height, so this file is
part of what every arm using it trained and was scored under. The failure modes
worth pinning are both silent:

  * changing an existing body's height would alter what an already-finished run
    means, so a differing value is REFUSED, not overwritten
  * a missing body is a hard KeyError at env startup (140/140 zero-shot pairs
    crashed on "KeyError: 300"), so adding must actually land in the file
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "add_body_heights.py"
MODELS = Path.home() / "Downloads" / "models" / "smplx"

pytestmark = pytest.mark.skipif(
    not (MODELS / "SMPLX_NEUTRAL.npz").exists(),
    reason="SMPL-X neutral model not present (laptop-only)")


def _betas(path, mapping):
    np.savez(path, **{k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()},
             _genders=np.array([f"{k}:neutral" for k in mapping]))


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT)] + args,
                          capture_output=True, text=True)


def test_adds_missing_bodies(tmp_path):
    b = tmp_path / "b.npz"
    _betas(b, {"sub900": np.zeros(16)})
    h = tmp_path / "h.json"
    h.write_text(json.dumps({"100": 1.77}))
    r = _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS)])
    assert r.returncode == 0, r.stderr
    got = json.loads(h.read_text())
    assert "900" in got and 1.5 < got["900"] < 2.0        # zero betas = the mean body
    assert got["100"] == 1.77                             # untouched


def test_zero_betas_is_the_canonical_height(tmp_path):
    """Sanity on the computation itself: the mean SMPL-X body is ~1.7 m, and the
    task's own fallback for 'no per-subject body' is 1.585."""
    b = tmp_path / "b.npz"
    _betas(b, {"sub900": np.zeros(16)})
    h = tmp_path / "h.json"
    h.write_text("{}")
    _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS)])
    assert 1.6 < json.loads(h.read_text())["900"] < 1.8


def test_existing_body_with_a_different_height_is_refused(tmp_path):
    b = tmp_path / "b.npz"
    _betas(b, {"sub900": np.zeros(16)})
    h = tmp_path / "h.json"
    h.write_text(json.dumps({"900": 1.2345}))
    r = _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS)])
    assert r.returncode == 2
    assert "refusing to overwrite" in r.stderr
    assert json.loads(h.read_text())["900"] == 1.2345     # unchanged


def test_dry_run_writes_nothing(tmp_path):
    b = tmp_path / "b.npz"
    _betas(b, {"sub900": np.zeros(16)})
    h = tmp_path / "h.json"
    h.write_text("{}")
    r = _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS),
              "--dry-run"])
    assert r.returncode == 0 and "dry run" in r.stdout
    assert json.loads(h.read_text()) == {}


def test_metadata_keys_are_skipped(tmp_path):
    """_genders / _source live in the same npz and are not bodies."""
    b = tmp_path / "b.npz"
    _betas(b, {"sub900": np.zeros(16)})
    h = tmp_path / "h.json"
    h.write_text("{}")
    _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS)])
    assert list(json.loads(h.read_text())) == ["900"]


def test_taller_betas_give_a_taller_body(tmp_path):
    """The first SMPL-X shape coefficient scales overall size; if it did not move
    the height, the betas would not be reaching the mesh at all."""
    b = tmp_path / "b.npz"
    tall = np.zeros(16); tall[0] = 2.0
    short = np.zeros(16); short[0] = -2.0
    _betas(b, {"sub900": tall, "sub901": short})
    h = tmp_path / "h.json"
    h.write_text("{}")
    _run(["--betas", str(b), "--heights", str(h), "--models-dir", str(MODELS)])
    got = json.loads(h.read_text())
    assert got["900"] != got["901"]

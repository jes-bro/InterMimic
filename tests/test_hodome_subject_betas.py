"""hodome_subject_betas.py: one body per HODome person, in the generator's format.

What these pin is the part that is silently wrong rather than loud: which vector
becomes a person's body. A wrong betas vector still builds an MJCF, still trains,
still evaluates -- and measures a person who does not exist.

  * shape is per PERSON, so several sequences of one subject collapse to one
    vector (their mean), with the spread reported
  * a per-frame (T, n) fit collapses over time rather than taking frame 0
  * ids are offset away from OMOMO / synthetic / CARI4D ranges, and the mapping
    back to the HODome subject is recorded
  * the npz carries '_genders' in the exact form generate_per_subject_mjcfs.py
    parses ('sub<N>:<gender>')
  * an unknown betas key fails loudly instead of guessing
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hodome_subject_betas.py"


def _seq(path, betas, gender=None):
    d = {"betas": np.asarray(betas, dtype=np.float32)}
    if gender is not None:
        d["gender"] = np.array(gender)
    np.savez(path, **d)


@pytest.fixture
def src(tmp_path):
    s = tmp_path / "smplx"
    s.mkdir()
    # subject01: two sequences, slightly different fits of the same person
    _seq(s / "subject01_box.npz", [1.0] * 10, "male")
    _seq(s / "subject01_chair.npz", [1.2] * 10, "male")
    # subject02: one sequence, per-frame fit
    _seq(s / "subject02_box.npz", np.tile(np.array([[-0.5] * 10]), (7, 1)), "female")
    return s


def _run(args):
    r = subprocess.run([sys.executable, str(SCRIPT)] + args,
                       capture_output=True, text=True)
    return r


def test_one_body_per_subject_is_the_mean(src, tmp_path):
    out = tmp_path / "b.npz"
    r = _run(["--src", str(src), "--out", str(out)])
    assert r.returncode == 0, r.stderr
    d = np.load(out, allow_pickle=True)
    assert "sub300" in d.files and "sub301" in d.files
    assert np.allclose(d["sub300"][:10], 1.1, atol=1e-5)      # mean of 1.0 and 1.2


def test_per_frame_fit_collapses_over_time(src, tmp_path):
    out = tmp_path / "b.npz"
    _run(["--src", str(src), "--out", str(out)])
    d = np.load(out, allow_pickle=True)
    assert d["sub301"].shape == (16,)
    assert np.allclose(d["sub301"][:10], -0.5, atol=1e-5)


def test_betas_are_padded_to_sixteen(src, tmp_path):
    """The generator indexes 16; a 10-vector fit pads with zeros, which are the
    fit's actual values for those coefficients, not unknowns."""
    out = tmp_path / "b.npz"
    _run(["--src", str(src), "--out", str(out)])
    d = np.load(out, allow_pickle=True)
    assert d["sub300"].shape == (16,)
    assert np.allclose(d["sub300"][10:], 0.0)


def test_genders_in_the_form_the_generator_parses(src, tmp_path):
    out = tmp_path / "b.npz"
    _run(["--src", str(src), "--out", str(out)])
    d = np.load(out, allow_pickle=True)
    got = {str(e).split(":", 1)[0]: str(e).split(":", 1)[1] for e in d["_genders"]}
    assert got == {"sub300": "male", "sub301": "female"}


def test_gender_override_when_the_npz_has_none(tmp_path):
    s = tmp_path / "smplx"
    s.mkdir()
    _seq(s / "subject01_box.npz", [0.3] * 10)                 # no gender key
    out = tmp_path / "b.npz"
    _run(["--src", str(s), "--out", str(out), "--gender", "subject01=female"])
    d = np.load(out, allow_pickle=True)
    assert str(d["_genders"][0]) == "sub300:female"


def test_default_gender_is_neutral(tmp_path):
    s = tmp_path / "smplx"
    s.mkdir()
    _seq(s / "subject01_box.npz", [0.3] * 10)
    out = tmp_path / "b.npz"
    _run(["--src", str(s), "--out", str(out)])
    d = np.load(out, allow_pickle=True)
    assert str(d["_genders"][0]) == "sub300:neutral"


def test_ids_avoid_the_existing_ranges_and_record_the_source(src, tmp_path):
    """sub1-17 OMOMO, sub100+ synthetic, sub204/sub401+ CARI4D: a collision would
    silently overwrite another dataset's MJCF."""
    out = tmp_path / "b.npz"
    _run(["--src", str(src), "--out", str(out)])
    d = np.load(out, allow_pickle=True)
    ids = sorted(int(k[3:]) for k in d.files if k.startswith("sub"))
    assert ids == [300, 301]
    src_map = {str(e).split(":")[0]: str(e).split(":")[1] for e in d["_source"]}
    assert src_map == {"sub300": "subject01", "sub301": "subject02"}


def test_first_id_is_settable(src, tmp_path):
    out = tmp_path / "b.npz"
    _run(["--src", str(src), "--out", str(out), "--first-id", "350"])
    d = np.load(out, allow_pickle=True)
    assert "sub350" in d.files


def test_unknown_betas_key_fails_loudly(tmp_path):
    s = tmp_path / "smplx"
    s.mkdir()
    np.savez(s / "subject01_box.npz", something_else=np.zeros(10, np.float32))
    r = _run(["--src", str(s), "--out", str(tmp_path / "b.npz")])
    assert r.returncode != 0
    assert "none of" in r.stderr and "--inspect" in r.stderr


def test_inspect_prints_keys(src):
    r = _run(["--inspect", str(src / "subject01_box.npz")])
    assert r.returncode == 0
    assert "betas" in r.stdout and "shape=" in r.stdout


def test_no_subject_files_fails_loudly(tmp_path):
    s = tmp_path / "empty"
    s.mkdir()
    r = _run(["--src", str(s), "--out", str(tmp_path / "b.npz")])
    assert r.returncode != 0 and "no subject" in r.stderr

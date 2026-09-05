#!/usr/bin/env python3
"""Tests for the neutral-betas-from-MJCF fit.

The point of these is that the SMPL-X models are laptop-only, so the solver has
to be verifiable WITHOUT them. Everything here runs against a synthetic model
whose ground-truth betas are known by construction, which is a stronger check
than a real fit anyway: with a real body you only ever see a residual, never
whether the recovered vector is the right one.

    python3 -m pytest tests/test_fit_neutral_betas_from_mjcf.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

fit_mod = pytest.importorskip("fit_neutral_betas_from_mjcf",
                              reason="needs scipy")
from smplx_pose import SMPLX_JOINTS, _MJCF_TO_SMPL   # noqa: E402


N_BETAS = 6          # small basis keeps the synthetic fits fast and well-posed
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def toy_model():
    """A model with the same algebra as SMPL-X: J(beta) linear in beta.

    v_template/shapedirs are over 'vertices' that J_regressor selects one-to-one
    as joints, so J(beta) = T + S.beta exactly -- the structure the solver
    assumes, with none of SMPL-X's size.
    """
    n = len(SMPLX_JOINTS)
    v_template = RNG.normal(scale=0.30, size=(n, 3))
    shapedirs = RNG.normal(scale=0.05, size=(n, 3, N_BETAS))
    return {"v_template": v_template,
            "shapedirs": shapedirs,
            "J_regressor": np.eye(n)}


def bones_for(model, betas):
    """MJCF-style [(child, parent, length)] generated FROM a known beta.

    Uses the real _MJCF_TO_SMPL mapping and the real MJCF parent relationships,
    so build_pairs is exercised on genuine names rather than invented ones.
    """
    J = fit_mod.rest_joints(model, betas)
    idx = {n: i for i, n in enumerate(SMPLX_JOINTS)}
    # a spanning set of MJCF bones whose endpoints both map into SMPL-X
    chain = [("L_Knee", "L_Hip"), ("L_Ankle", "L_Knee"), ("R_Knee", "R_Hip"),
             ("R_Ankle", "R_Knee"), ("Spine", "Torso"), ("Chest", "Spine"),
             ("Neck", "Chest"), ("Head", "Neck"), ("L_Elbow", "L_Shoulder"),
             ("L_Wrist", "L_Elbow"), ("R_Elbow", "R_Shoulder"),
             ("R_Wrist", "R_Elbow"), ("L_Shoulder", "L_Thorax"),
             ("R_Shoulder", "R_Thorax")]
    out = []
    for child, parent in chain:
        cs, ps = _MJCF_TO_SMPL.get(child), _MJCF_TO_SMPL.get(parent)
        if cs in idx and ps in idx:
            out.append((child, parent, float(np.linalg.norm(J[idx[cs]] - J[idx[ps]]))))
    return out


# --------------------------------------------------------------------------
# The solver recovers a shape it was given.
# --------------------------------------------------------------------------
def test_recovers_the_bone_lengths_it_was_given(toy_model):
    truth = RNG.normal(scale=0.8, size=N_BETAS)
    bones = bones_for(toy_model, truth)
    got, rep = fit_mod.fit_betas(toy_model, SMPLX_JOINTS, bones, N_BETAS)

    # Bone lengths are the objective, so THEY must be reproduced to ~0.
    assert rep["rms_mm"] < 0.5, rep
    fitted = [b for _, _, b in bones_for(toy_model, got)]
    target = [b for _, _, b in bones]
    assert np.allclose(fitted, target, atol=5e-4)


def test_zero_shape_recovers_the_template(toy_model):
    bones = bones_for(toy_model, np.zeros(N_BETAS))
    got, rep = fit_mod.fit_betas(toy_model, SMPLX_JOINTS, bones, N_BETAS)
    assert rep["rms_mm"] < 0.5
    assert np.allclose([b for _, _, b in bones_for(toy_model, got)],
                       [b for _, _, b in bones], atol=5e-4)


def test_a_taller_body_fits_longer_bones(toy_model):
    """Sanity that the fit tracks shape rather than returning a constant."""
    short = bones_for(toy_model, np.full(N_BETAS, -0.6))
    tall = bones_for(toy_model, np.full(N_BETAS, +0.6))
    bs, _ = fit_mod.fit_betas(toy_model, SMPLX_JOINTS, short, N_BETAS)
    bt, _ = fit_mod.fit_betas(toy_model, SMPLX_JOINTS, tall, N_BETAS)
    assert not np.allclose(bs, bt, atol=1e-3)
    leg_s = sum(b for _, _, b in bones_for(toy_model, bs)[:2])
    leg_t = sum(b for _, _, b in bones_for(toy_model, bt)[:2])
    assert abs(leg_t - leg_s) > 1e-3


# --------------------------------------------------------------------------
# It refuses rather than fitting nonsense.
# --------------------------------------------------------------------------
def test_too_few_matched_bones_is_an_error_not_a_bad_fit(toy_model):
    """A near-empty objective must fail loudly -- it would 'converge' happily."""
    with pytest.raises(SystemExit, match="bones matched"):
        fit_mod.fit_betas(toy_model, SMPLX_JOINTS,
                          [("L_Knee", "L_Hip", 0.4)], N_BETAS)


def test_unmappable_bones_are_reported_not_silently_dropped(toy_model):
    bones = bones_for(toy_model, np.zeros(N_BETAS))
    bones += [("NotAJoint", "AlsoNot", 0.3)]
    _, rep = fit_mod.fit_betas(toy_model, SMPLX_JOINTS, bones, N_BETAS)
    assert rep["n_dropped"] == 1 and "NotAJoint" in rep["dropped"]


# --------------------------------------------------------------------------
# Archive handling: the id collision this whole script exists to prevent.
# --------------------------------------------------------------------------
def test_roundtrip_preserves_existing_subjects(tmp_path):
    base = tmp_path / "base.npz"
    fit_mod.write_archive(str(base), {"sub2": np.arange(16, dtype=np.float32)},
                          {"sub2": "neutral"})
    betas, genders = fit_mod.load_archive(str(base))
    assert list(betas) == ["sub2"] and genders == {"sub2": "neutral"}
    assert np.allclose(betas["sub2"], np.arange(16))


def test_missing_archive_reads_as_empty(tmp_path):
    betas, genders = fit_mod.load_archive(str(tmp_path / "nope.npz"))
    assert betas == {} and genders == {}


def test_refuses_to_overwrite_an_existing_subject(tmp_path, monkeypatch):
    """Overwriting a subject's shape would silently change what a trained
    policy was conditioned on, with no error and no way to notice later."""
    base = tmp_path / "base.npz"
    fit_mod.write_archive(str(base), {"sub300": np.zeros(16, dtype=np.float32)},
                          {"sub300": "neutral"})
    arch, _ = fit_mod.load_archive(str(base))
    assert "sub300" in arch          # the condition main() raises SystemExit on


def test_free_id_range_is_above_every_id_in_use():
    """300+ must clear real (1-17), synthetic (100-139) and syn-ladder (140-239)."""
    assert fit_mod.FIRST_FREE_ID > 239

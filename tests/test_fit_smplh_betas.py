#!/usr/bin/env python3
"""Tests for scripts/fit_smplh_betas.py.

The failure this guards against is silent and figure-destroying: if the bone
matching breaks, the fit falls back to too few bones, every subject converges to
something near the template, and a "retargeting to multiple bodies" figure renders
panels that all look the same -- the exact claim the figure exists to make.

The SMPL-H model files are laptop-only, so the model-dependent tests skip
cleanly on the cluster; the name-mapping tests run everywhere.

Run:  python tests/test_fit_smplh_betas.py   (exit 0 = all green)
  or: pytest tests/test_fit_smplh_betas.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import fit_smplh_betas as fb  # noqa: E402
from smplx_pose import SMPLH_JOINTS, _MJCF_TO_SMPL  # noqa: E402

MJCF = os.path.join(REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub2.xml")
MODEL = os.path.expanduser("~/Downloads/SMPLH_MALE.pkl")


def _model():
    if not os.path.isfile(MODEL):
        return None
    return fb._load_model_file(MODEL)


def test_mjcf_bones_are_parsed_with_parents():
    if not os.path.isfile(MJCF):
        print("skip: MJCF not present")
        return
    bones = fb.mjcf_bones(MJCF)
    # 52 bodies, root has no parent, so at most 51 bones; none may be zero-length
    assert 40 <= len(bones) <= 51, len(bones)
    assert all(L > 0 for _, _, L in bones)
    names = {c for c, _, _ in bones}
    assert "Pelvis" not in names, "the root has no parent and must be skipped"
    for child, parent, _ in bones:
        assert child != parent
    print(f"ok: {len(bones)} MJCF bones parsed, all with parents and positive length")


def test_almost_every_bone_maps_into_smplh():
    """The mapping is the thing that silently degrades. If it breaks, the fit
    still 'succeeds' on a handful of bones and every body comes out generic."""
    if not os.path.isfile(MJCF):
        print("skip: MJCF not present")
        return
    bones = fb.mjcf_bones(MJCF)
    pairs, tgt, used, missing = fb.build_pairs(SMPLH_JOINTS, bones)
    assert len(pairs) >= 45, f"only {len(pairs)} bones matched -- mapping degraded"
    assert len(pairs) == len(tgt) == len(used)
    assert all(0 <= c < len(SMPLH_JOINTS) and 0 <= p < len(SMPLH_JOINTS)
               for c, p in pairs)
    assert all(c != p for c, p in pairs)
    print(f"ok: {len(pairs)} of {len(bones)} bones map into SMPL-H "
          f"({len(missing)} unmapped)")


def test_unmapped_bones_are_reported_not_dropped():
    fake = [("Pelvis_to_nowhere", "NoSuchParent", 0.3),
            ("L_Knee", "L_Hip", 0.4)]
    pairs, tgt, used, missing = fb.build_pairs(SMPLH_JOINTS, fake)
    assert used == ["L_Knee"]
    assert missing == ["Pelvis_to_nowhere"], missing
    print("ok: a bone with no SMPL-H counterpart is reported, not silently dropped")


def test_round_trip_recovers_known_betas():
    """Generate bone lengths FROM a known shape, fit, and check we get it back.
    This is the fit's own correctness, independent of any MJCF."""
    model = _model()
    if model is None:
        print(f"skip: no model at {MODEL}")
        return
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1.2, size=model["shapedirs"].shape[-1])

    # a plausible bone set: SMPL-H's own kintree, expressed as MJCF-style triples
    kin = model["kintree_table"]
    parent_of = {int(kin[1, i]): int(kin[0, i]) for i in range(kin.shape[1])}
    inv = {v: k for k, v in _MJCF_TO_SMPL.items() if v in SMPLH_JOINTS}
    J = fb.smplh_rest_joints(model, truth)
    bones = []
    n = len(SMPLH_JOINTS)
    for ci, pi in parent_of.items():
        # SMPL kintrees store the root's parent as uint32 -1 (4294967295), so a
        # plain `pi < 0` does not catch it -- bound both ends.
        if not (0 <= pi < n and 0 <= ci < n) or pi == ci:
            continue
        cn, pn = inv.get(SMPLH_JOINTS[ci]), inv.get(SMPLH_JOINTS[pi])
        if cn and pn:
            bones.append((cn, pn, float(np.linalg.norm(J[ci] - J[pi]))))
    assert len(bones) >= 30, len(bones)

    pairs, tgt, used, _ = fb.build_pairs(SMPLH_JOINTS, bones)
    from scipy.optimize import least_squares
    sol = least_squares(lambda b: fb.smplh_bone_lengths(model, b, pairs) - tgt,
                        np.zeros(len(truth)), method="lm", max_nfev=20000)
    resid = fb.smplh_bone_lengths(model, sol.x, pairs) - tgt
    rms_mm = float(np.sqrt((resid ** 2).mean()) * 1000)
    # bone lengths do not pin every shape direction (girth is invisible to them),
    # so betas need not match exactly -- the LENGTHS must.
    assert rms_mm < 1.0, f"round-trip rms {rms_mm:.3f} mm"
    print(f"ok: round trip reproduces its own bone lengths to {rms_mm:.3f} mm rms")


def test_fitted_subjects_are_actually_different():
    """The whole point. sub2 (m, 1.775 m) and sub10 (f, 1.587 m) must come out
    visibly different, or the figure shows the same body twice."""
    out = os.path.join(REPO, "scripts/omomo_betas_smplh.npz")
    if not os.path.isfile(out):
        print("skip: omomo_betas_smplh.npz not generated yet")
        return
    d = np.load(out, allow_pickle=True)
    for s in ("sub2", "sub10"):
        assert s in d.files, s
    b2, b10 = d["sub2"], d["sub10"]
    assert b2.shape == b10.shape
    assert np.linalg.norm(b2 - b10) > 1.0, "the two shapes are nearly identical"
    g = {x.split(":")[0]: x.split(":")[1] for x in d["_genders"]}
    assert g["sub2"] == "male" and g["sub10"] == "female", g
    print(f"ok: sub2 and sub10 differ (||db|| = {np.linalg.norm(b2 - b10):.2f}), "
          f"genders preserved")


def test_gender_lookup_refuses_to_guess():
    genders = fb.subject_genders(os.path.join(REPO, "scripts/omomo_betas.npz"))
    assert genders["sub2"] == "male"
    assert genders["sub10"] == "female"
    assert "sub999" not in genders          # main() raises rather than defaulting
    print("ok: gender comes from the archive, with no default")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")

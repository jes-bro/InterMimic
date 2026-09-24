#!/usr/bin/env python3
"""BEHAVE's SMPL-H shapes -> SMPL-X betas, by matching rest-pose bone lengths.

WHY. generate_per_subject_mjcfs.py builds bodies with SMPL-X. BEHAVE ships
SMPL-H fits, and the two shape spaces are NOT interchangeable: different
templates, different topology (6890 vs 10475 vertices), so feeding SMPL-H betas
to an SMPL-X model gives a body that is nobody in particular. The closed-form
projection used by refit_betas_to_neutral.py does not apply either -- that one
works because source and target are both SMPL-X, where V(beta) is linear in a
shared basis.

WHAT IS MATCHED. Bone LENGTHS, the same objective fit_smplh_betas.py uses in the
other direction. Rest joints are a smooth function of shape,

    J(beta) = J_regressor @ (v_template + shapedirs . beta)

so for every parent->child bone in the MJCF skeleton we take its length under the
SMPL-H model with BEHAVE's betas, and solve for the SMPL-X betas whose bones are
as close as possible. Lengths, not joint positions: the two rest poses differ, and
a length is invariant to that.

GENDER IS PRESERVED. Each subject is fit from its own SMPL-H model to the SMPL-X
model of the same gender (BEHAVE is 5 male / 3 female). Crossing genders would
make beta absorb a template difference on top of the shape difference.

WHAT THIS IS NOT. Not smplx's transfer_model (a cross-topology mesh fit with
vertex correspondences, the official route). This matches a skeleton, which is
what the simulator actually uses -- an MJCF is bone offsets and capsules. The
report prints the residual per subject so the quality is stated, not assumed.

    python3 scripts/fit_smplx_from_smplh_betas.py \\
        --betas scripts/behave_subject_betas.npz \\
        --models /simurgh2/projects/ret-hoi/InterAct/models \\
        --out scripts/behave_subject_betas_smplx.npz

CLUSTER: needs BOTH model families. SMPL-H lives only on simurgh
(InterAct/models/smplh); the laptop has SMPL-X alone.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import (SMPLH_JOINTS, SMPLX_JOINTS, _MJCF_TO_SMPL,  # noqa: E402
                        _load_model_file, _parse_mjcf_tree)

DEFAULT_MJCF = "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub2.xml"


def rest_joints(model, betas):
    """J(beta) = J_regressor @ (v_template + shapedirs . beta), in metres."""
    n = min(len(betas), model["shapedirs"].shape[2])
    v = model["v_template"] + np.einsum("vcb,b->vc", model["shapedirs"][:, :, :n], betas[:n])
    return model["J_regressor"] @ v


def bone_pairs(mjcf_path, names):
    """[(child_idx, parent_idx)] for every MJCF bone, in one model's joint order.

    Parents come from the MJCF tree, not from a model's kintree, so both models
    are measured on exactly the same bones. A bone whose endpoints have no
    counterpart is returned separately rather than dropped silently.
    """
    idx = {n: i for i, n in enumerate(names)}
    pairs, used, missing = [], [], []
    for child, parent, off in _parse_mjcf_tree(mjcf_path):
        if parent is None or np.linalg.norm(np.asarray(off, dtype=np.float64)) < 1e-9:
            continue                                  # root, or a pure re-orientation frame
        cs, ps = _MJCF_TO_SMPL.get(child), _MJCF_TO_SMPL.get(parent)
        ci, pi = (idx.get(cs) if cs else None), (idx.get(ps) if ps else None)
        if ci is None or pi is None or ci == pi:
            missing.append(child)
            continue
        pairs.append((ci, pi))
        used.append(child)
    return pairs, used, missing


def lengths(model, betas, pairs):
    J = rest_joints(model, betas)
    return np.array([np.linalg.norm(J[c] - J[p]) for c, p in pairs])


def fit_one(src_model, src_betas, src_pairs, tgt_model, tgt_pairs, n_betas=16):
    """-> (smplx_betas, report). Least squares on bone-length residuals."""
    from scipy.optimize import least_squares

    target = lengths(src_model, src_betas, src_pairs)

    def residual(b):
        return lengths(tgt_model, b, tgt_pairs) - target

    sol = least_squares(residual, np.zeros(n_betas), method="lm", max_nfev=20000)
    r = residual(sol.x)
    return sol.x, dict(n_bones=len(tgt_pairs),
                       rms_mm=float(np.sqrt((r ** 2).mean()) * 1000),
                       max_mm=float(np.abs(r).max() * 1000))


def genders_of(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for item in np.atleast_1d(d.get("_genders", [])):
        s = item.decode() if isinstance(item, bytes) else str(item)
        if ":" in s:
            k, v = s.split(":", 1)
            out[k] = v.strip().lower()
    return out


def model_path(models_dir, family, gender):
    """SMPL-H ships .pkl, SMPL-X .npz; both are tried, either layout."""
    g = gender.upper()
    cands = [os.path.join(models_dir, family.lower(), f"{family.upper()}_{g}.npz"),
             os.path.join(models_dir, family.lower(), f"{family.upper()}_{g}.pkl"),
             os.path.join(models_dir, family.lower(), g.lower(), "model.npz"),
             os.path.join(models_dir, family.lower(), g.lower(), "model.pkl"),
             os.path.join(models_dir, f"{family.upper()}_{g}.npz"),
             os.path.join(models_dir, f"{family.upper()}_{g}.pkl")]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit(f"no {family} {gender} model under {models_dir}; tried:\n  " +
                     "\n  ".join(cands))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", required=True, help="npz of SMPL-H betas (+_genders)")
    ap.add_argument("--models", required=True, help="dir holding smplh/ and smplx/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mjcf", default=DEFAULT_MJCF,
                    help="any per-subject MJCF; only its TREE is used, for the bone list")
    ap.add_argument("--src-family", default="smplh")
    ap.add_argument("--tgt-family", default="smplx")
    a = ap.parse_args()

    src_names = SMPLH_JOINTS if a.src_family == "smplh" else SMPLX_JOINTS
    tgt_names = SMPLX_JOINTS if a.tgt_family == "smplx" else SMPLH_JOINTS

    d = np.load(a.betas, allow_pickle=True)
    genders = genders_of(a.betas)
    bodies = sorted(k for k in d.files if k.startswith("sub"))
    if not bodies:
        raise SystemExit(f"no sub* entries in {a.betas}")

    src_pairs, used, missing = bone_pairs(a.mjcf, src_names)
    tgt_pairs, _, _ = bone_pairs(a.mjcf, tgt_names)
    if len(src_pairs) != len(tgt_pairs) or len(src_pairs) < 10:
        raise SystemExit(f"bone lists disagree: {len(src_pairs)} vs {len(tgt_pairs)} "
                         f"-- the two joint orders do not cover the same bones")
    print(f"{len(src_pairs)} bones matched on both models"
          + (f"; {len(missing)} MJCF bone(s) have no counterpart and are dropped"
             if missing else ""))

    cache, out, meta = {}, {}, []
    for b in bodies:
        g = genders.get(b, "neutral")
        for fam in (a.src_family, a.tgt_family):
            if (fam, g) not in cache:
                cache[(fam, g)] = _load_model_file(model_path(a.models, fam, g))
        betas_x, rep = fit_one(cache[(a.src_family, g)], np.asarray(d[b], dtype=np.float64),
                               src_pairs, cache[(a.tgt_family, g)], tgt_pairs)
        out[b] = betas_x.astype(np.float32)
        meta.append(f"{b}:{g}")
        print(f"  {b} ({g}): |betas| {np.linalg.norm(betas_x):.2f}  "
              f"rms {rep['rms_mm']:.1f} mm  max {rep['max_mm']:.1f} mm over {rep['n_bones']} bones")

    np.savez(a.out, _genders=np.array(meta),
             _source=np.array([f"fit from {a.src_family} betas in {os.path.basename(a.betas)}"]),
             **out)
    print(f"\nwrote {a.out}: {len(out)} bodies")
    print("A few mm rms is a good fit (bones are 10-50 cm). Tens of mm means the")
    print("shape does not live in the target model's space -- say so rather than")
    print("reporting those bodies as the BEHAVE subjects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

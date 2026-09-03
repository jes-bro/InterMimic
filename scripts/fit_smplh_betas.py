#!/usr/bin/env python3
"""Fit SMPL-H betas that reproduce an OMOMO subject's proportions, for figures.

WHY THIS EXISTS, AND WHY IT IS NOT A GENERAL NEED. Within a dataset everything is
same-family: OMOMO bodies are SMPL-X end to end, CARI4D/BEHAVE are SMPL-H end to
end, and no conversion is wanted. The mismatch appears in exactly one place --
rendering an OMOMO body through CARI4D's figure renderer
(tools/render_behave_style.py), which calls lib_smpl.get_smpl(.., True) and so
builds an SMPL-H body from whatever betas the bundle carries. There is no SMPL-X
path in that renderer. Without SMPL-H betas per body, every panel of a
"retargeting to multiple bodies" figure renders the SAME body, which is precisely
the claim the figure is making.

So: this is a rendering aid. It is NOT part of the training pipeline, and nothing
should condition a policy on its output -- for that, use the neutral-space betas
(scripts/omomo_betas_neutral_aug.npz, see refit_betas_to_neutral.py), because
betas from different gendered models are different bases and are not comparable.

WHAT IT FITS. The subject's proportions live in their MJCF as parent->child body
offsets. SMPL-H's rest joints are

    J(beta) = J_regressor @ (v_template + shapedirs . beta)

so bone lengths are a smooth function of beta. This solves, per subject,

    argmin_beta  sum_bones ( ||J_child(beta) - J_parent(beta)|| - mjcf_bone )^2

Bone LENGTHS rather than joint positions because that is what a figure shows --
height, limb proportions, build -- and lengths are invariant to the rest pose
differing between the two skeletons.

GENDER IS PRESERVED. Each subject is fit against its own SMPL-H model
(SMPLH_MALE / SMPLH_FEMALE), read from the gendered archive scripts/omomo_betas.npz.
Fitting a female subject to the male template would force beta to absorb a
template difference on top of the shape difference -- strictly more residual. The
"one shared basis" argument that motivates the neutral refit is about
CONDITIONING, where betas are compared across subjects; nothing compares betas
across panels of a figure.

    python3 scripts/fit_smplh_betas.py --subjects sub2 sub10
    python3 scripts/fit_smplh_betas.py --subjects sub2 sub10 --out scripts/omomo_betas_smplh.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import (SMPLH_JOINTS, _MJCF_TO_SMPL, _load_model_file,  # noqa: E402
                        _parse_mjcf_tree)

DEFAULT_MODELS = os.path.expanduser("~/Downloads")
DEFAULT_MJCF_DIR = "isaacgym/src/intermimic/data/assets/smplx"
DEFAULT_GENDERS = "scripts/omomo_betas.npz"


def subject_genders(path):
    """-> {'sub2': 'male', ...} from a betas archive's _genders array."""
    d = np.load(path, allow_pickle=True)
    if "_genders" not in d.files:
        raise SystemExit(f"{path} has no _genders entry")
    out = {}
    for item in np.atleast_1d(d["_genders"]):
        s = item.decode() if isinstance(item, bytes) else str(item)
        if ":" in s:
            k, v = s.split(":", 1)
            out[k] = v
    return out


def mjcf_bones(mjcf_path):
    """-> [(child_mjcf_name, parent_mjcf_name, length)] for every MJCF bone.

    The root has no parent and is skipped; a zero-length offset (a pure
    re-orientation frame) carries no shape information and is skipped too,
    rather than dragging a zero into the objective.
    """
    bones = []
    for name, parent, off in _parse_mjcf_tree(mjcf_path):
        if parent is None:
            continue
        L = float(np.linalg.norm(np.asarray(off, dtype=np.float64)))
        if L > 1e-9:
            bones.append((name, parent, L))
    return bones


def smplh_rest_joints(model, betas):
    """SMPL-H rest-pose joint positions for a shape. J = Jreg @ (T + S . beta)."""
    v = model["v_template"] + np.einsum("vcb,b->vc", model["shapedirs"], betas)
    return model["J_regressor"] @ v


def smplh_bone_lengths(model, betas, pairs):
    """Bone lengths for the (child_idx, parent_idx) pairs, in SMPL-H order."""
    J = smplh_rest_joints(model, betas)
    return np.array([np.linalg.norm(J[c] - J[p]) for c, p in pairs])


def build_pairs(names, bones):
    """Map MJCF bones onto SMPL-H (child_idx, parent_idx) index pairs.

    Uses the MJCF's OWN parent, translated through _MJCF_TO_SMPL, rather than
    SMPL-H's kintree -- so both sides are guaranteed to be measuring the same
    bone even where the two skeletons order or parent things differently.

    A bone whose child or parent has no SMPL-H counterpart is reported, not
    silently dropped: a silent drop would quietly shrink the objective.
    """
    idx = {n: i for i, n in enumerate(names)}
    pairs, tgt, used, missing = [], [], [], []
    for child, parent, L in bones:
        cs, ps = _MJCF_TO_SMPL.get(child), _MJCF_TO_SMPL.get(parent)
        ci, pi = (idx.get(cs) if cs else None), (idx.get(ps) if ps else None)
        if ci is None or pi is None or ci == pi:
            missing.append(child)
            continue
        pairs.append((ci, pi))
        tgt.append(L)
        used.append(child)
    return pairs, np.array(tgt), used, missing


def fit_subject(mjcf_path, model, names, n_betas, verbose=True):
    """-> (betas, report dict). Least squares on bone-length residuals."""
    from scipy.optimize import least_squares

    pairs, tgt, used, missing = build_pairs(names, mjcf_bones(mjcf_path))
    if len(pairs) < 10:
        raise SystemExit(f"only {len(pairs)} bones matched for {mjcf_path} -- "
                         f"the MJCF body names and SMPLH_JOINTS disagree")
    if missing and verbose:
        print(f"    {len(missing)} MJCF bone(s) absent from SMPL-H, dropped: "
              f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")

    def residual(b):
        return smplh_bone_lengths(model, b, pairs) - tgt

    sol = least_squares(residual, np.zeros(n_betas), method="lm", max_nfev=20000)
    r = residual(sol.x)
    return sol.x, dict(n_bones=len(pairs), rms_mm=float(np.sqrt((r ** 2).mean()) * 1000),
                       max_mm=float(np.abs(r).max() * 1000),
                       worst=used[int(np.argmax(np.abs(r)))])


def chain_check(model, betas, bones, names, chain):
    """Target vs fitted length of a named MJCF bone chain, in cm.

    An apples-to-apples check: both sides sum the SAME bones. (Comparing the
    shaped MESH's vertical extent against the MJCF's capsule-hull height is NOT
    comparable -- the surface reaches past the capsules at head and sole, a
    systematic ~12 cm offset that says nothing about fit quality.)
    """
    idx = {n: i for i, n in enumerate(names)}
    by_child = {c: (pn, L) for c, pn, L in bones}
    tgt = fit = 0.0
    J = smplh_rest_joints(model, betas)
    for child in chain:
        if child not in by_child:
            continue
        parent, L = by_child[child]
        cs, ps = _MJCF_TO_SMPL.get(child), _MJCF_TO_SMPL.get(parent)
        if cs is None or ps is None:
            continue
        tgt += L
        fit += float(np.linalg.norm(J[idx[cs]] - J[idx[ps]]))
    return tgt * 100, fit * 100


LEG_CHAIN = ["L_Hip", "L_Knee", "L_Ankle"]
SPINE_CHAIN = ["Torso", "Spine", "Chest", "Neck", "Head"]
ARM_CHAIN = ["L_Shoulder", "L_Elbow", "L_Wrist"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--mjcf-dir", default=DEFAULT_MJCF_DIR)
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="dir holding SMPLH_MALE.pkl / SMPLH_FEMALE.pkl")
    ap.add_argument("--genders-from", default=DEFAULT_GENDERS,
                    help="betas archive whose _genders says male/female per subject")
    ap.add_argument("--n-betas", type=int, default=16,
                    help="SMPL-H shapedirs here carry 16 components")
    ap.add_argument("--out", default=None, help="npz to write (default: report only)")
    args = ap.parse_args(argv)

    genders = subject_genders(args.genders_from)
    cache, out, out_genders = {}, {}, []

    for sub in args.subjects:
        mjcf = os.path.join(args.mjcf_dir, f"smplx_omomo_{sub}.xml")
        if not os.path.isfile(mjcf):
            raise SystemExit(f"no MJCF at {mjcf}")
        g = genders.get(sub)
        if g is None:
            raise SystemExit(f"{args.genders_from} does not give a gender for {sub} -- "
                             f"refusing to guess, a wrong template inflates the fit")
        if g not in cache:
            path = os.path.join(args.models, f"SMPLH_{g.upper()}.pkl")
            cache[g] = _load_model_file(path)
            n = cache[g]["shapedirs"].shape[-1]
            print(f"loaded {path}  ({n} shape components)")
        model = cache[g]
        nb = min(args.n_betas, model["shapedirs"].shape[-1])

        print(f"\n{sub} ({g})")
        betas, rep = fit_subject(mjcf, model, SMPLH_JOINTS, nb)
        print(f"    {rep['n_bones']} bones | rms {rep['rms_mm']:.2f} mm | "
              f"max {rep['max_mm']:.2f} mm ({rep['worst']})")
        bones = mjcf_bones(mjcf)
        for label, chain in (("leg", LEG_CHAIN), ("spine", SPINE_CHAIN), ("arm", ARM_CHAIN)):
            t, f = chain_check(model, betas, bones, SMPLH_JOINTS, chain)
            print(f"    {label:>5s} chain  MJCF {t:6.2f} cm  ->  fitted {f:6.2f} cm"
                  f"   ({f - t:+.2f} cm)")
        out[sub] = betas.astype(np.float32)
        out_genders.append(f"{sub}:{g}")

    if args.out:
        np.savez(args.out, _genders=np.array(out_genders), **out)
        print(f"\nwrote {args.out}  ({len(out)} subject(s))")
        print("NOTE: rendering aid only -- do NOT condition a policy on these; "
              "they are per-gender bases and not comparable across subjects.")
    else:
        print("\n(no --out given, nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

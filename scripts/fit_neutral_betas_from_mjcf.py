#!/usr/bin/env python3
"""Fit NEUTRAL SMPL-X betas that reproduce a body's proportions, from its MJCF.

WHY THIS EXISTS. Betas conditioning requires every subject's shape to live in ONE
shared basis -- the neutral SMPL-X space (see refit_betas_to_neutral.py and
project_betas_gendered_not_shared). OMOMO subjects get there by a closed-form
vertex projection: source and target are both SMPL-X, identical 10475-vertex
topology, so it is a plain least-squares reprojection.

That route is CLOSED for a CARI4D / EgoExo4D subject. Those bodies are SMPL-H
(6890 vertices), so there is no common vertex space to project through. Without
this script the only options were to leave such a subject out of the betas space
entirely -- which is why the bball arm trains with betas off -- or to let the
observation silently pick up a DIFFERENT person's betas, which is the trap the
g3_bball config header documents: the clip is sub100_bball_000.pt, so a lookup
finds synthetic OMOMO "sub100", succeeds, and conditions on the wrong identity.

WHAT IT FITS. A subject's proportions are already in its MJCF, as parent->child
body offsets -- and the MJCF is what Isaac Gym actually simulates, so it is the
authoritative statement of the body being driven. SMPL-X rest joints are a smooth
function of beta,

    J(beta) = J_regressor @ (v_template + shapedirs . beta)

so this solves, over the neutral model,

    argmin_beta  sum_bones ( ||J_child(beta) - J_parent(beta)|| - mjcf_bone )^2

Bone LENGTHS, not joint positions: lengths are invariant to the two skeletons
resting in different poses, which they do. This is the same objective
fit_smplh_betas.py uses -- that script fits SMPL-H betas for RENDERING; this one
fits NEUTRAL SMPL-X betas for CONDITIONING. Opposite directions, shared method,
and the bone-matching helpers are imported from there rather than re-written.

CROSS-TOPOLOGY IS THE POINT, AND ITS LIMIT. Nothing here requires the source body
to be SMPL-X: the MJCF is just a set of measured bone lengths. That is what lets
an SMPL-H subject enter the neutral space at all. But the fit can only capture
what bone lengths encode -- limb proportions and height, not girth or soft-shape.
The reported residual says how well those lengths were matched; it does NOT
certify that two bodies with equal bone lengths look alike.

    # report only, writes nothing
    python3 scripts/fit_neutral_betas_from_mjcf.py \\
        --mjcf isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml \\
        --id sub300 --dry-run

    # fit and write a NEW archive, carrying an existing one forward
    python3 scripts/fit_neutral_betas_from_mjcf.py \\
        --mjcf .../smplh_behave_sub100.xml --id sub300 \\
        --base scripts/omomo_betas_neutral_aug.npz \\
        --out  scripts/omomo_betas_neutral_aug_bball.npz

NEEDS THE SMPL-X MODELS, which are laptop-only (~/Downloads/models/smplx) -- run
this locally and commit the npz, do not expect it to work on the cluster.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import SMPLX_JOINTS, _load_model_file          # noqa: E402
from fit_smplh_betas import (build_pairs, chain_check, mjcf_bones,  # noqa: E402
                             ARM_CHAIN, LEG_CHAIN, SPINE_CHAIN)

DEFAULT_MODELS = os.path.expanduser("~/Downloads/models/smplx")
# 1-239 are taken: 1-17 real OMOMO, 100-139 synthetic, 140-239 the syn-ladder
# roster. 300+ is clear, and leaving the 200s alone keeps room for that ladder to
# grow without colliding with reconstructed subjects.
FIRST_FREE_ID = 300


def rest_joints(model, betas):
    """J(beta) = J_regressor @ (v_template + shapedirs . beta)."""
    v = model["v_template"] + np.einsum("vcb,b->vc", model["shapedirs"], betas)
    return model["J_regressor"] @ v


def bone_lengths(model, betas, pairs):
    J = rest_joints(model, betas)
    return np.array([np.linalg.norm(J[c] - J[p]) for c, p in pairs])


def fit_betas(model, names, bones, n_betas=16):
    """-> (betas, report). Least squares on bone-length residuals.

    `bones` is [(child_mjcf_name, parent_mjcf_name, length_m)], i.e. the output
    of mjcf_bones -- passed in rather than read from a file so the solver can be
    tested without an MJCF or a model on disk.
    """
    from scipy.optimize import least_squares

    pairs, tgt, used, missing = build_pairs(names, bones)
    if len(pairs) < 10:
        raise SystemExit(
            f"only {len(pairs)} bones matched -- the MJCF body names and "
            f"SMPL-X joint names disagree, so the fit would be meaningless")

    def residual(b):
        return bone_lengths(model, b, pairs) - tgt

    sol = least_squares(residual, np.zeros(n_betas), method="lm", max_nfev=20000)
    r = residual(sol.x)
    return sol.x, dict(
        n_bones=len(pairs),
        n_dropped=len(missing),
        dropped=missing,
        rms_mm=float(np.sqrt((r ** 2).mean()) * 1000),
        max_mm=float(np.abs(r).max() * 1000),
        worst=used[int(np.argmax(np.abs(r)))],
    )


def load_archive(path):
    """-> (dict of subject->betas, dict of subject->gender). Missing file = empty."""
    if not path or not os.path.isfile(path):
        return {}, {}
    d = np.load(path, allow_pickle=True)
    betas = {k: np.asarray(d[k]) for k in d.files if not k.startswith("_")}
    genders = {}
    if "_genders" in d.files:
        for item in np.atleast_1d(d["_genders"]):
            s = item.decode() if isinstance(item, bytes) else str(item)
            if ":" in s:
                k, v = s.split(":", 1)
                genders[k] = v
    return betas, genders


def write_archive(path, betas, genders):
    arr = np.array([f"{k}:{v}" for k, v in sorted(genders.items())])
    np.savez(path, _genders=arr, **betas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mjcf", required=True, help="the subject's MJCF")
    p.add_argument("--id", required=True,
                   help=f"subject id to store, e.g. sub{FIRST_FREE_ID}. "
                        f"1-239 are taken; use {FIRST_FREE_ID}+ for reconstructed "
                        f"subjects so they cannot collide with a synthetic body "
                        f"that is simultaneously a TARGET in the same roster.")
    p.add_argument("--models-dir", default=DEFAULT_MODELS)
    p.add_argument("--model-file", default="SMPLX_NEUTRAL.npz")
    p.add_argument("--n-betas", type=int, default=16)
    p.add_argument("--base", help="existing npz to carry forward (not modified)")
    p.add_argument("--out", help="npz to write; omit with --dry-run")
    p.add_argument("--dry-run", action="store_true",
                   help="fit and report, write nothing")
    args = p.parse_args(argv)

    if not args.dry_run and not args.out:
        raise SystemExit("--out is required unless --dry-run")

    n = int(args.id[3:]) if args.id.startswith("sub") else None
    if n is not None and n < FIRST_FREE_ID:
        print(f"WARNING: id {args.id} is below {FIRST_FREE_ID}; ids 1-239 are in "
              f"use by real, synthetic and syn-ladder bodies. If this id is also "
              f"a TARGET body in the same roster, the source lookup will find the "
              f"WRONG person and succeed silently.", file=sys.stderr)

    model = _load_model_file(os.path.join(args.models_dir, args.model_file))
    bones = mjcf_bones(args.mjcf)
    betas, rep = fit_betas(model, SMPLX_JOINTS, bones, args.n_betas)

    print(f"mjcf   : {args.mjcf}")
    print(f"model  : {os.path.join(args.models_dir, args.model_file)}")
    print(f"bones  : {rep['n_bones']} matched"
          + (f", {rep['n_dropped']} dropped ({', '.join(rep['dropped'][:5])})"
             if rep["n_dropped"] else ""))
    print(f"residual: rms {rep['rms_mm']:.2f} mm | max {rep['max_mm']:.2f} mm "
          f"(worst bone: {rep['worst']})")
    for label, chain in (("leg", LEG_CHAIN), ("spine", SPINE_CHAIN), ("arm", ARM_CHAIN)):
        t, f = chain_check(model, betas, bones, SMPLX_JOINTS, chain)
        print(f"  {label:<6} target {t:6.2f} cm | fitted {f:6.2f} cm | diff {f - t:+5.2f} cm")
    print(f"betas  : {np.array2string(betas, precision=4, max_line_width=100)}")

    if rep["rms_mm"] > 20:
        print("\nWARNING: rms residual > 20 mm. The neutral shape space could not "
              "reproduce these proportions; conditioning on this vector would be "
              "conditioning on a body you are not simulating.", file=sys.stderr)

    if args.dry_run:
        print("\n(--dry-run: nothing written)")
        return 0

    arch, genders = load_archive(args.base)
    if args.id in arch:
        raise SystemExit(
            f"ERROR: {args.id} already exists in {args.base}. Refusing to "
            f"overwrite a subject's shape -- pick a free id, or drop --base.")
    arch[args.id] = betas.astype(np.float32)
    genders[args.id] = "neutral"
    write_archive(args.out, arch, genders)
    print(f"\nwrote {args.out}  ({len(arch)} subjects, {args.id} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

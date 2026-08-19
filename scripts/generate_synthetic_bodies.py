#!/usr/bin/env python3
"""Generate synthetic SMPL-X bodies for TRAINING augmentation, in shared NEUTRAL space.

These are target-only bodies: the policy is trained to drive them with REAL source
motions (no ground truth of their own needed). Interpolating betas is only valid
because everyone's already in the neutral frame (see project_betas_gendered_not_shared)
-- so this consumes omomo_betas_neutral.npz, NOT the gendered omomo_betas.npz.

Two kinds, tagged so the fold-in experiments can use them separately:
  inhull  -- blends of 2-3 real (non-held-out) subjects (realistic interpolation)
  extrap  -- a real subject pushed AWAY from the population centroid (extrapolation,
             to test whether training on out-of-hull shapes helps the held-out bodies)

Held-out subjects {sub4,sub10,sub16} are EXCLUDED from the blend basis, and any
synthetic body landing within --min-heldout-dist of a held-out subject is rejected
and resampled -- so training on these can't contaminate the OOD test.

Output npz: syn0..syn{N-1} (16,) neutral betas + _genders (all 'neutral') +
_kinds ('inhull'/'extrap'). Feed to generate_per_subject_mjcfs.py (gender=neutral).
"""
import argparse
from pathlib import Path

import numpy as np


# SMPL-X body joint order (standard): 0 pelvis, 1/2 L/R hip, 4/5 L/R knee,
# 7/8 L/R ankle, 12 neck, 15 head, 16/17 L/R shoulder, 18/19 L/R elbow,
# 20/21 L/R wrist.
_J = {"L_hip": 1, "R_hip": 2, "L_knee": 4, "L_ankle": 7, "neck": 12, "head": 15,
      "L_shoulder": 16, "R_shoulder": 17, "L_elbow": 18, "L_wrist": 20}


def load_model(models_dir):
    z = np.load(models_dir / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    return (z["v_template"].astype(np.float64),
            z["shapedirs"].astype(np.float64),
            z["J_regressor"].astype(np.float64))


def proportions(model, betas):
    """Anthropometric ratios for a betas vector, from the shaped mesh + joints.
    All normalized by height so the bands are scale-free."""
    v_template, shapedirs, J_reg = model
    V = v_template + np.einsum("vni,i->vn", shapedirs[:, :, :len(betas)], betas)
    J = J_reg @ V
    height = V[:, 1].max() - V[:, 1].min()
    seg = lambda a, b: np.linalg.norm(J[_J[a]] - J[_J[b]])
    return {
        "height_m": height,
        "leg/h":      (seg("L_hip", "L_knee") + seg("L_knee", "L_ankle")) / height,
        "arm/h":      (seg("L_shoulder", "L_elbow") + seg("L_elbow", "L_wrist")) / height,
        "shoulder/h": seg("L_shoulder", "R_shoulder") / height,
        "hip/h":      seg("L_hip", "R_hip") / height,
        "head/h":     (V[:, 1].max() - J[_J["neck"]][1]) / height,
    }


def proportion_bands(model, real_betas, margin):
    """[lo, hi] per metric = real population range widened by `margin` of its width
    (height band widened in meters). Every real body passes its own bands by
    construction; synthetics must look proportion-plausible to be admitted."""
    rows = [proportions(model, b) for b in real_betas]
    bands = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * margin if k != "height_m" else 0.05 + (hi - lo) * margin
        bands[k] = (lo - pad, hi + pad)
    return bands


def within_bands(model, betas, bands):
    pr = proportions(model, betas)
    return all(bands[k][0] <= pr[k] <= bands[k][1] for k in bands)


def neutral_height(models_dir, betas):
    z = np.load(models_dir / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    sd = z["shapedirs"][:, :, :len(betas)].astype(np.float64)
    V = z["v_template"].astype(np.float64) + np.einsum("vni,i->vn", sd, betas)
    return float(V[:, 1].max() - V[:, 1].min())   # SMPL-X up = +y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--betas", type=Path, default=Path("scripts/omomo_betas_neutral.npz"))
    ap.add_argument("--models-dir", type=Path,
                    default=Path.home() / "Downloads" / "models" / "smplx")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--frac-extrap", type=float, default=0.3,
                    help="fraction of bodies that are extrapolated past the hull")
    ap.add_argument("--extrap-scale", type=float, nargs=2, default=[1.15, 1.45],
                    help="push factor range (1.0 = on the subject, >1 = beyond it)")
    ap.add_argument("--frac-extrap-dir", type=float, default=0.0,
                    help="fraction of bodies from RANDOM-DIRECTION extrapolation: "
                         "centroid + random unit dir * radius. Opens the whole "
                         "out-of-hull shell (ray extrap only covers 13 lines); "
                         "realism is enforced by the proportion bands")
    ap.add_argument("--dir-radius", type=float, nargs=2, default=None,
                    help="radius range for --frac-extrap-dir (default: [1.0, 1.6] x "
                         "the reals' max distance from their centroid)")
    ap.add_argument("--proportion-margin", type=float, default=0.3,
                    help="widen each real-population proportion band by this fraction "
                         "of its width; bodies outside ANY band are rejected")
    # sub13 added 2026-07-21: the FIRST synthetic set (sub100-139) was generated
    # with only {sub4,sub10,sub16} held out, so sub121 landed 0.34 from sub13 and
    # contaminated it as a held-out test (see project_synthetic_sub13_leak). Any
    # regeneration now protects the full held-out set so this can't recur.
    ap.add_argument("--held-out", nargs="+", default=["sub4", "sub10", "sub13", "sub16"])
    ap.add_argument("--min-heldout-dist", type=float, default=2.0,
                    help="reject synthetic bodies closer than this (L2 in betas) to any held-out subject")
    ap.add_argument("--start-id", type=int, default=100,
                    help="synthetic bodies are named sub<start-id>.. so the env's "
                         "int(sub[3:]) machinery + smplx_omomo_sub<N>.xml loading just work")
    ap.add_argument("--min-pairwise-dist", type=float, default=0.0,
                    help="reject synthetic bodies closer than this (L2 betas) to ANY "
                         "other training body: real basis, --existing-syn, or an "
                         "already-accepted new body. 0 = off (legacy behavior)")
    ap.add_argument("--existing-syn", type=Path, default=None,
                    help="npz of already-in-use synthetic bodies: they join the "
                         "pairwise-spacing avoid set and the new ids start after them")
    ap.add_argument("--max-tries", type=int, default=500000,
                    help="total sampling attempts before FAILING LOUDLY (a stall "
                         "means the spacing/floor constraints don't fit this many "
                         "bodies -- relax deliberately, never silently)")
    ap.add_argument("--heights-out", type=Path, default=None,
                    help="heights json path (default: synthetic_heights.json beside --out)")
    ap.add_argument("--merge-heights", type=Path, default=None,
                    help="existing heights json to merge into --heights-out (so one "
                         "file covers old + new synthetics for subjectHeightsFile)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("scripts/synthetic_bodies_neutral.npz"))
    ap.add_argument("--combined-out", type=Path,
                    default=Path("scripts/omomo_betas_neutral_aug.npz"),
                    help="real neutral betas + synthetic, in one file for the env's betas_file")
    args = ap.parse_args()
    args.models_dir = args.models_dir.expanduser()

    src = np.load(args.betas, allow_pickle=True)
    held = set(args.held_out)
    basis_names = [k for k in src.files if k != "_genders" and k not in held]
    B = np.stack([src[s].astype(np.float64) for s in basis_names])   # (M,16) blend basis
    H = np.stack([src[s].astype(np.float64) for s in args.held_out]) # held-out, to avoid
    centroid = B.mean(0)
    rng = np.random.default_rng(args.seed)

    model = load_model(args.models_dir)
    bands = proportion_bands(model, B, args.proportion_margin)
    print("[proportions] acceptance bands (real range +/- margin):")
    for k, (lo, hi) in bands.items():
        print(f"    {k:11s} {lo:.3f} .. {hi:.3f}")
    radii = np.linalg.norm(B - centroid, axis=1)
    dir_radius = args.dir_radius or [radii.max() * 1.0, radii.max() * 1.6]

    # Pairwise-spacing avoid set: reals (basis) + existing synthetics + accepted new.
    existing = {}
    if args.existing_syn is not None:
        ez = np.load(args.existing_syn, allow_pickle=True)
        existing = {k: ez[k].astype(np.float64) for k in ez.files if not k.startswith("_")}
        print(f"[spacing] existing synthetics in avoid set: {len(existing)} from {args.existing_syn}")
    avoid = [b for b in B] + list(existing.values())

    n_extrap = int(round(args.n * args.frac_extrap))
    n_dir = int(round(args.n * args.frac_extrap_dir))
    n_inhull = args.n - n_extrap - n_dir
    assert n_inhull >= 0, "frac_extrap + frac_extrap_dir must be <= 1"

    def far_from_heldout(b):
        return np.linalg.norm(H - b, axis=1).min() >= args.min_heldout_dist

    def far_from_roster(b):
        if args.min_pairwise_dist <= 0:
            return True
        A = np.stack(avoid)
        return np.linalg.norm(A - b, axis=1).min() >= args.min_pairwise_dist

    def sample_inhull():
        k = rng.integers(2, 4)                              # blend 2-3 real subjects
        idx = rng.choice(len(B), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        return w @ B[idx]

    def sample_extrap():
        i = rng.integers(len(B))
        s = rng.uniform(*args.extrap_scale)
        return centroid + s * (B[i] - centroid)            # push past subject i

    def sample_extrap_dir():
        d = rng.standard_normal(B.shape[1])
        d /= np.linalg.norm(d)
        return centroid + rng.uniform(*dir_radius) * d     # anywhere on the shell

    out, kinds, tries = {}, [], 0
    for kind, sampler, count in [("inhull", sample_inhull, n_inhull),
                                 ("extrap", sample_extrap, n_extrap),
                                 ("extrap_dir", sample_extrap_dir, n_dir)]:
        made = 0
        while made < count:
            tries += 1
            if tries > args.max_tries:
                import sys
                sys.exit(f"FATAL: only placed {len(out)}/{args.n} bodies in {args.max_tries} "
                         f"tries -- min-pairwise-dist {args.min_pairwise_dist} does not fit "
                         f"this many bodies. Relax the spacing DELIBERATELY or lower --n.")
            b = sampler()
            if not far_from_heldout(b):
                continue                                    # too close to a test body -> resample
            if not far_from_roster(b):
                continue                                    # too close to another training body
            if not within_bands(model, b, bands):
                continue                                    # implausible human proportions
            name = f"sub{args.start_id + len(out)}"          # env-compatible naming
            out[name] = b.astype(np.float32)
            avoid.append(b.astype(np.float64))
            kinds.append(kind)
            made += 1
    if args.min_pairwise_dist > 0:
        print(f"[spacing] placed {len(out)} bodies with pairwise >= {args.min_pairwise_dist} "
              f"in {tries} tries")

    out["_genders"] = np.array([f"{k}:neutral" for k in out if not k.startswith("_")], dtype=object)
    out["_kinds"] = np.array(kinds, dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    # combined file the env reads: real neutral betas + synthetic, all gender=neutral
    combined = {k: src[k] for k in src.files if k != "_genders"}
    combined.update({k: v.astype(np.float32) for k, v in existing.items()})
    combined.update({k: out[k] for k in out if not k.startswith("_")})
    combined["_genders"] = np.array([f"{k}:neutral" for k in combined], dtype=object)
    np.savez(args.combined_out, **combined)
    print(f"wrote combined real+synthetic betas -> {args.combined_out} "
          f"({len(combined) - 1} bodies)")

    # report
    names = [k for k in out if not k.startswith("_")]
    heights = {n: neutral_height(args.models_dir, out[n].astype(np.float64)) for n in names}
    d_held = {n: float(np.linalg.norm(H - out[n], axis=1).min()) for n in names}
    d_basis = {n: float(np.linalg.norm(B - out[n], axis=1).min()) for n in names}
    print(f"generated {len(names)} bodies ({n_inhull} inhull + {n_extrap} extrap) -> {args.out}")
    print(f"  height:   {min(heights.values())*100:.0f}-{max(heights.values())*100:.0f} cm "
          f"(real basis range for sanity)")
    print(f"  nearest real subject (L2 betas):  min {min(d_basis.values()):.2f}  max {max(d_basis.values()):.2f}")
    print(f"  nearest HELD-OUT subject (L2):     min {min(d_held.values()):.2f}  "
          f"(must be >= {args.min_heldout_dist}; closer ones were rejected)")
    tall = [n for n, h in heights.items() if not (1.40 <= h <= 2.10)]
    print(f"  bodies outside 140-210cm (sanity flag): {tall if tall else 'none'}")

    # heights file for bodyNormalizedReward (synthetic bodies aren't in SUBJECT_HEIGHTS).
    # Approx = neutral mesh extent; measure_subject_bodies.py on the MJCFs is the exact version.
    import json
    hpath = args.heights_out or (args.out.parent / "synthetic_heights.json")
    hmap = {}
    if args.merge_heights is not None:
        hmap.update(json.load(open(args.merge_heights)))
    hmap.update({str(args.start_id + i): round(heights[n], 4) for i, n in enumerate(names)})
    json.dump(hmap, open(hpath, "w"), indent=2)
    print(f"  wrote heights -> {hpath} (for --subject-heights-file under body-norm)")


if __name__ == "__main__":
    main()

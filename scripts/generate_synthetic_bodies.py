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
    ap.add_argument("--held-out", nargs="+", default=["sub4", "sub10", "sub16"])
    ap.add_argument("--min-heldout-dist", type=float, default=2.0,
                    help="reject synthetic bodies closer than this (L2 in betas) to any held-out subject")
    ap.add_argument("--start-id", type=int, default=100,
                    help="synthetic bodies are named sub<start-id>.. so the env's "
                         "int(sub[3:]) machinery + smplx_omomo_sub<N>.xml loading just work")
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

    n_extrap = int(round(args.n * args.frac_extrap))
    n_inhull = args.n - n_extrap

    def far_from_heldout(b):
        return np.linalg.norm(H - b, axis=1).min() >= args.min_heldout_dist

    def sample_inhull():
        k = rng.integers(2, 4)                              # blend 2-3 real subjects
        idx = rng.choice(len(B), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        return w @ B[idx]

    def sample_extrap():
        i = rng.integers(len(B))
        s = rng.uniform(*args.extrap_scale)
        return centroid + s * (B[i] - centroid)            # push past subject i

    out, kinds = {}, []
    for kind, sampler, count in [("inhull", sample_inhull, n_inhull),
                                 ("extrap", sample_extrap, n_extrap)]:
        made = 0
        while made < count:
            b = sampler()
            if not far_from_heldout(b):
                continue                                    # too close to a test body -> resample
            name = f"sub{args.start_id + len(out)}"          # sub100, sub101, ... (env-compatible)
            out[name] = b.astype(np.float32)
            kinds.append(kind)
            made += 1

    out["_genders"] = np.array([f"{k}:neutral" for k in out if not k.startswith("_")], dtype=object)
    out["_kinds"] = np.array(kinds, dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    # combined file the env reads: real neutral betas + synthetic, all gender=neutral
    combined = {k: src[k] for k in src.files if k != "_genders"}
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


if __name__ == "__main__":
    main()

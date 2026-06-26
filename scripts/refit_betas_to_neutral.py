#!/usr/bin/env python3
"""Re-express each subject's SMPL-X shape in the NEUTRAL model's coordinate space.

WHY: omomo_betas.npz holds GENDERED betas (a male subject's 16 numbers are
male-model coefficients, a female's are female-model coefficients -- different
bases; see project_betas_gendered_not_shared). That makes the policy's betas
conditioning incoherent across genders and makes cross-gender interpolation
(synthetic bodies) invalid. This puts everyone in ONE shared neutral space.

HOW (exact closed-form -- NOT smplx's transfer_model, which is for cross-TOPOLOGY
SMPL<->SMPL-X; here source & target are both SMPL-X with identical 10475-vertex
topology, so it's a plain linear projection):

  SMPL-X rest shape is LINEAR in betas (Loper et al., "SMPL", SIGGRAPH Asia 2015;
  Pavlakos et al., "SMPL-X", CVPR 2019):
        V(beta) = v_template + shapedirs @ beta
  Build the subject's body with its gendered model:  V* = T_g + B_g @ beta_g
  Solve for the neutral betas that reproduce V*:
        beta_neutral = argmin || T_n + B_n @ beta - V* ||^2   (ordinary least squares)
  Closed form via lstsq; exact because the objective is quadratic in beta.

Residual (how far V* sits outside the 16-D neutral shape subspace) is ~2 mm here,
so the conversion is essentially lossless.

NOTE: this builds V* with the GENDERED model -- i.e. it reproduces the bodies you
have actually been training on. If you later confirm OMOMO's betas were neutral
all along, rebuild V* with the neutral model instead (one-line change).

Output: <out> npz with per-subject (16,) NEUTRAL betas + _genders all set to
'neutral'. Feed it to generate_per_subject_mjcfs.py with gender forced neutral.

Usage:
  python scripts/refit_betas_to_neutral.py \
      --betas scripts/omomo_betas.npz \
      --models-dir ~/Downloads/models/smplx \
      --out scripts/omomo_betas_neutral.npz
"""
import argparse
from pathlib import Path

import numpy as np


def load_model(models_dir: Path, gender: str):
    z = np.load(models_dir / f"SMPLX_{gender.upper()}.npz", allow_pickle=True)
    return z["v_template"].astype(np.float64), z["shapedirs"].astype(np.float64)


def gendered_verts(v_template, shapedirs, betas):
    """V(beta) = v_template + shapedirs[:,:,:K] @ beta  (rest-pose, linear in beta)."""
    k = len(betas)
    return v_template + np.einsum("vni,i->vn", shapedirs[:, :, :k], betas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--betas", type=Path, default=Path("scripts/omomo_betas.npz"))
    ap.add_argument("--models-dir", type=Path,
                    default=Path.home() / "Downloads" / "models" / "smplx",
                    help="dir holding SMPLX_{NEUTRAL,MALE,FEMALE}.npz")
    ap.add_argument("--out", type=Path, default=Path("scripts/omomo_betas_neutral.npz"))
    args = ap.parse_args()
    args.models_dir = args.models_dir.expanduser()

    src = np.load(args.betas, allow_pickle=True)
    gender_of = {x.split(":")[0]: x.split(":")[1] for x in src["_genders"]}
    subs = sorted((k for k in src.files if k != "_genders"),
                  key=lambda s: int(s[3:]))

    # Neutral target basis (reshaped to (3N, K)) + template, built once.
    Tn_full, Bn_full = load_model(args.models_dir, "neutral")
    K = src[subs[0]].shape[0]
    Bn = Bn_full[:, :, :K].reshape(-1, K)          # (3N, K)
    Tn = Tn_full.reshape(-1)                        # (3N,)

    gen_cache = {}
    out = {}
    print(f"{'subj':6}{'gender':8}{'|beta_g|':>9}{'|beta_neu|':>11}{'resid_mean':>12}{'resid_max':>11}")
    for s in subs:
        g = gender_of[s]
        if g not in gen_cache:
            gen_cache[g] = load_model(args.models_dir, g)
        Tg, Bg = gen_cache[g]
        beta_g = src[s].astype(np.float64)
        Vstar = gendered_verts(Tg, Bg, beta_g).reshape(-1)        # body we trained on
        beta_n, *_ = np.linalg.lstsq(Bn, Vstar - Tn, rcond=None)  # exact LS projection
        resid = np.linalg.norm((Tn + Bn @ beta_n - Vstar).reshape(-1, 3), axis=1)
        out[s] = beta_n.astype(np.float32)
        print(f"{s:6}{g:8}{np.linalg.norm(beta_g):>9.2f}{np.linalg.norm(beta_n):>11.2f}"
              f"{resid.mean()*1000:>10.2f}mm{resid.max()*1000:>9.2f}mm")

    out["_genders"] = np.array([f"{s}:neutral" for s in subs], dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"\nwrote {args.out}  ({len(subs)} subjects, all gender=neutral, shared space)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic SMPL-X bodies for TRAINING augmentation by sampling the
SMPL-X shape prior directly, spread evenly over it, in the shared NEUTRAL space.

These are target-only bodies: the policy is trained to drive them with REAL source
motions (no ground truth of their own needed). They live in the neutral frame
(see project_betas_gendered_not_shared), like the refit real subjects in
omomo_betas_neutral.npz.

WHAT THE AXES ARE. Each of the 16 betas is a coefficient on one principal
component of human shape, learned from ~3800 CAESAR body scans (SMPL-X paper,
Sec. 3.1). The authors scaled the space "for unit variance" (Sec. 3.2): on every
axis, a beta of 1.0 is one standard deviation of that scan population. Axis 0
moves height a lot per unit, axis 15 is a subtle shape change per unit, and
both are "one standard deviation of people". So the population prior over
betas is a standard normal on every axis.

HOW BODIES ARE PLACED. Latin hypercube sampling over the prior: on every axis
the population is cut into --n equal-probability slices and exactly one body
lands in each slice, axes shuffled independently, then each slice position is
mapped through the normal inverse CDF so values are in the model's unit-
variance scale. Every axis is therefore covered end to end evenly (one body
among the shortest 1/n of people, one among the tallest 1/n, likewise for each
other axis) instead of clumping the way n random draws do. scipy's
"random-cd" optimization additionally pushes the n points apart in the full
16-D space. The clip (default 4 std) touches 0.006% of draws per axis and
exists only in case a freak tail value makes the mesh self-intersect; it does
not shape the distribution.

NO SUBJECT IS INVOLVED. The set is a function of --seed alone. This script
never reads any real person's betas: not the training subjects (no blending,
no centroid, no bands calibrated on them) and not the test subjects (no
distance checks). The only use of --betas is to COPY the real subjects into
the combined file the env loads, unchanged and uninspected. The tests pin
this: replacing every real body's betas leaves the synthetics bit-identical.

History: earlier sets (sub100-239) were blends/extrapolations of the training
subjects, filtered by distance to the TEST subjects. That shaped the training
set with the test set and is invalid; this script cannot produce such sets.

Outputs:
  --out           sub<start-id>.. (16,) float32 betas + _genders ('<name>:neutral')
                  + _kinds (all 'prior'). Feed to generate_per_subject_mjcfs.py.
  --combined-out  every entry of --betas + the synthetics, for the env's betas_file
  --heights-out   {"<id>": height_m} for bodyNormalizedReward's subjectHeightsFile
                  (neutral mesh extent; measure_subject_bodies.py on the MJCFs is exact)

Naming: the env identifies a body only by its name sub<N> (MJCF
smplx_omomo_sub<N>.xml + betas key), so --start-id must be a free block.
In use: 1-17 real OMOMO subjects, 100-239 the retired synthetic sets, 401-580
CARI4D / soccer subjects. Default 600.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri            # standard normal inverse CDF
from scipy.stats import qmc                # Latin hypercube sampler


def load_model(models_dir):
    """v_template (V,3) and shapedirs (V,3,C) of the neutral model, for heights."""
    z = np.load(models_dir / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    return z["v_template"].astype(np.float64), z["shapedirs"].astype(np.float64)


def neutral_height(model, betas):
    """Standing height of the rest-pose mesh: extent along SMPL-X's up axis (+y)."""
    v_template, shapedirs = model
    V = v_template + np.einsum("vni,i->vn", shapedirs[:, :, :len(betas)], betas)
    return float(V[:, 1].max() - V[:, 1].min())


def sample_prior_lhs(n, n_betas, clip, seed):
    """n bodies spread evenly over the unit-variance shape prior.

    LatinHypercube gives u in (0,1)^(n x n_betas) with exactly one point per
    1/n slice on every axis; ndtri maps each slice position to the normal
    quantile, i.e. to betas in population-std units. Nothing else goes in."""
    u = qmc.LatinHypercube(d=n_betas, optimization="random-cd", seed=seed).random(n)
    return np.clip(ndtri(u), -clip, clip)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--betas", type=Path, default=Path("scripts/omomo_betas_neutral.npz"),
                    help="real subjects' neutral betas -- copied into --combined-out, "
                         "never read for sampling")
    ap.add_argument("--models-dir", type=Path,
                    default=Path.home() / "Downloads" / "models" / "smplx",
                    help="dir holding SMPLX_NEUTRAL.npz (only used to report heights)")
    ap.add_argument("--n", type=int, default=30, help="number of bodies")
    ap.add_argument("--n-betas", type=int, default=16,
                    help="shape components per body (the env conditions on 16)")
    ap.add_argument("--clip", type=float, default=4.0,
                    help="clip every beta to +/- this many population std devs "
                         "(4 keeps 99.994%% of people per axis)")
    ap.add_argument("--seed", type=int, default=0,
                    help="THE input: the set is a deterministic function of the seed")
    ap.add_argument("--start-id", type=int, default=600,
                    help="bodies are named sub<start-id>.. (see module docstring)")
    # "gen4" = this generation of synthetic bodies (the prior-sampled set), to keep
    # it apart from the gen2/g3 experiments and the retired sub100-239 sets.
    ap.add_argument("--out", type=Path, default=Path("scripts/synthetic_bodies_gen4.npz"))
    ap.add_argument("--combined-out", type=Path,
                    default=Path("scripts/omomo_betas_neutral_aug_gen4.npz"),
                    help="real neutral betas + synthetic, in one file for the env's betas_file")
    ap.add_argument("--heights-out", type=Path, default=None,
                    help="heights json (default: synthetic_heights_gen4.json beside --out)")
    args = ap.parse_args()
    args.models_dir = args.models_dir.expanduser()

    # ---- sample: seed in, bodies out. No person's betas are read here. ----
    S = sample_prior_lhs(args.n, args.n_betas, args.clip, args.seed)
    names = [f"sub{args.start_id + i}" for i in range(args.n)]
    print(f"[prior] {args.n} bodies x {args.n_betas} betas, Latin hypercube over N(0,1) "
          f"per axis, clipped at +/-{args.clip}, seed {args.seed}; "
          f"no real subject's betas were read")

    out = {n: S[i].astype(np.float32) for i, n in enumerate(names)}
    out["_genders"] = np.array([f"{n}:neutral" for n in names], dtype=object)
    out["_kinds"] = np.array(["prior"] * args.n, dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    # ---- combined file the env reads: the real subjects, copied untouched, + ours ----
    src = np.load(args.betas, allow_pickle=True)
    real_names = [k for k in src.files if k != "_genders"]
    clash = sorted(set(real_names) & set(names))
    if clash:
        raise SystemExit(f"ERROR: synthetic ids {clash} already exist in {args.betas}; "
                         f"pick another --start-id")
    combined = {k: src[k] for k in real_names}
    combined.update({n: out[n] for n in names})
    combined["_genders"] = np.array([f"{k}:neutral" for k in combined], dtype=object)
    np.savez(args.combined_out, **combined)
    print(f"wrote {args.out} ({args.n} synthetic) and {args.combined_out} "
          f"({len(real_names)} real + {args.n} synthetic)")

    # ---- report (sanity only, nothing is filtered) ----
    model = load_model(args.models_dir)
    heights = {n: neutral_height(model, S[i]) for i, n in enumerate(names)}
    print(f"  height: {min(heights.values())*100:.0f}-{max(heights.values())*100:.0f} cm")
    print(f"  per-axis min..max: {np.round(S.min(0), 2).tolist()} .. "
          f"{np.round(S.max(0), 2).tolist()}")
    print(f"  |beta| max: {np.abs(S).max():.2f}   mean ||beta||^2: {(S**2).sum(1).mean():.1f} "
          f"(a typical person in the prior scores ~{args.n_betas})")
    odd = [n for n, h in heights.items() if not (1.40 <= h <= 2.10)]
    print(f"  bodies outside 140-210cm (flag only, kept): {odd if odd else 'none'}")

    hpath = args.heights_out or (args.out.parent / "synthetic_heights_gen4.json")
    json.dump({str(args.start_id + i): round(heights[n], 4) for i, n in enumerate(names)},
              open(hpath, "w"), indent=2)
    print(f"  wrote heights -> {hpath} (subjectHeightsFile for bodyNormalizedReward)")


if __name__ == "__main__":
    main()

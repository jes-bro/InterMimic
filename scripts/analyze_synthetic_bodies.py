#!/usr/bin/env python3
"""Compare a synthetic augmentation set to the real TRAINING subjects.

All bodies live in SMPL-X NEUTRAL beta space (see project_betas_gendered_not_shared),
so betas are directly comparable and we can push each through the neutral SMPL-X
mesh to read real anthropometry (height, arm span, widths, volume) rather than
squinting at raw beta coefficients.

TRAINING BODIES ONLY. The tool is parametrized by the training reals (--reals);
every other real in the betas file (i.e. the test bodies) is dropped at load
time and never measured, plotted or reported. Knowing where the test bodies sit
relative to the synthetics is exactly the information that shaped the old,
invalid synthetic sets, so this tool refuses to produce it.

Inputs (all local, no cluster):
  --betas    combined neutral betas (reals + synthetics), default
             scripts/omomo_betas_neutral_aug_gen4.npz
  --kinds    the synthetic-only npz with the _kinds tag per body
  ~/Downloads/models/smplx/SMPLX_NEUTRAL.npz   neutral template + shapedirs (for mesh measures)
  --heights  stored synthetic heights -> used as a self-check

Outputs (to --out):
  betas_pca.png            16-D betas -> 2-D PCA scatter, training real vs synthetic
  anthropometry.png        height / arm span / shoulder / hip / leg / volume distributions
  betas_violin.png         per-beta-dim spread, real vs synthetic
  coverage.png             synthetic -> nearest training real distance histogram
  body_measurements.csv    every training body x every measurement
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

# The fold's real TRAINING subjects (the generator's blend basis). Default =
# fold 0: the 17 OMOMO subjects minus that fold's test trio. sub4 is included:
# its MJCF is broken so it is never a simulated body, but its betas are a real
# human shape and it IS a blend parent. Overridable with --reals for other folds.
TRAIN_REALS = ["sub1", "sub2", "sub3", "sub4", "sub5", "sub6", "sub7", "sub8",
               "sub9", "sub11", "sub12", "sub14", "sub15", "sub17"]


# ---------------------------------------------------------------- data loading
def sub_num(name):
    """'sub103' -> 103 (so we can split real <100 from synthetic >=100)."""
    return int(name[3:])


def load_bodies(betas_path, kinds_path):
    """Return dict name -> {betas(16,), group in {real,prior}}
    for the TRAINING bodies only: reals listed in TRAIN_REALS + every synthetic
    tagged in the kinds file. Any other real in the betas file (the test bodies)
    is dropped here and never seen again."""
    z = np.load(betas_path, allow_pickle=True)

    # per-synthetic kind tag (parallel to the synthetic key order in the kinds npz)
    kz = np.load(kinds_path, allow_pickle=True)
    syn_keys = [k for k in kz.files if k.startswith("sub") and k[3:].isdigit()]
    kinds = {k: str(t) for k, t in zip(syn_keys, kz["_kinds"])}

    missing = [r for r in TRAIN_REALS if r not in z.files]
    if missing:
        raise SystemExit(f"ERROR: training reals {missing} not in {betas_path}")
    bodies = {}
    for n in TRAIN_REALS:
        bodies[n] = {"betas": z[n].astype(np.float64), "group": "real"}
    for n in syn_keys:
        if n not in z.files:
            raise SystemExit(f"ERROR: synthetic {n} from {kinds_path} not in {betas_path}")
        bodies[n] = {"betas": z[n].astype(np.float64), "group": kinds[n]}
    return bodies


# ------------------------------------------------------- mesh-derived measures
# SMPL-X body joint indices (for shoulder width via the joint regressor)
J_L_SHOULDER, J_R_SHOULDER = 16, 17


def load_smplx(models_dir, n_betas=16):
    """v_template (V,3) + shapedirs (V,3,n_betas) + J_regressor (J,V) from the npz.

    J_regressor maps vertices -> joint centers, giving a pose-independent shoulder
    width (the rest-pose mesh has arms extended, so mesh bounding boxes can't).
    """
    z = np.load(Path(models_dir) / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    v_template = z["v_template"].astype(np.float64)          # (V,3)
    shapedirs = z["shapedirs"][:, :, :n_betas].astype(np.float64)  # (V,3,n_betas)
    j_reg = z["J_regressor"].astype(np.float64)              # (J,V)
    return v_template, shapedirs, j_reg


def shaped_vertices(v_template, shapedirs, betas):
    """Apply betas to the template: V = T + shapedirs . betas  (rest pose, no LBS)."""
    return v_template + np.einsum("vni,i->vn", shapedirs, betas)


def measure(V, joints):
    """Anthropometry from a rest-pose SMPL-X mesh. SMPL-X axes: x=left, y=up, z=fwd.

    Height/depth/arm-span are mesh bounding-box extents; shoulder width comes from
    the JOINT regressor (the rest pose has arms extended at shoulder height, so a
    mesh x-slice there would just re-measure the arm span -- joints avoid that).
    Hip width is a mesh band (arms don't reach hip height, so it's clean).
    """
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    shoulder_w = abs(joints[J_L_SHOULDER, 0] - joints[J_R_SHOULDER, 0])  # x gap of shoulder joints
    return {
        "height": ext[1],           # up extent, matches generate_synthetic_bodies.neutral_height
        "arm_span": ext[0],         # fingertip-to-fingertip (x) in the rest T-pose
        "depth": ext[2],            # front-back
        "shoulder_w": shoulder_w,   # joint-based, NOT a mesh band (see docstring)
        "hip_w": _band_x_extent(V, lo[1], hi[1], 0.48, 0.58),
        "volume": _convex_volume(V),   # proxy for mass (denser betas -> bigger body)
    }


def _band_x_extent(V, y_lo, y_hi, f0, f1):
    """x-extent of vertices in the vertical band [f0,f1] of total height."""
    y0, y1 = y_lo + f0 * (y_hi - y_lo), y_lo + f1 * (y_hi - y_lo)
    band = V[(V[:, 1] >= y0) & (V[:, 1] <= y1)]
    if len(band) < 3:
        return float("nan")
    return float(band[:, 0].max() - band[:, 0].min())


def _convex_volume(V):
    """Convex-hull volume as a cheap, robust mass proxy (avoids needing faces)."""
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(V).volume)
    except Exception:
        # no scipy: fall back to bounding-box volume (still monotonic-ish in size)
        ext = V.max(0) - V.min(0)
        return float(ext[0] * ext[1] * ext[2])


# --------------------------------------------------------------------- plots
GROUP_COLOR = {"real": "#1b6fb3", "prior": "#e08214"}
GROUP_LABEL = {"real": "real training people",
               "prior": "synthetic (sampled from the SMPL-X shape prior)"}
GROUPS = ["real", "prior"]


def plot_pca(bodies, out):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    names = list(bodies)
    X = np.stack([bodies[n]["betas"] for n in names])
    P = PCA(n_components=2).fit(X)
    Y = P.transform(X)
    var = P.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 8))
    # synthetics drawn first, reals drawn last so the reals' number labels are
    # not hidden under synthetic dots (draw order only, no effect on numbers)
    for g in ("prior", "real"):
        idx = [i for i, n in enumerate(names) if bodies[n]["group"] == g]
        ax.scatter(Y[idx, 0], Y[idx, 1], s=90 if g == "real" else 55,
                   c=GROUP_COLOR[g], label=GROUP_LABEL[g],
                   edgecolor="k", linewidth=0.4, alpha=0.9, zorder=3 if g == "real" else 2)
    for i, n in enumerate(names):
        if bodies[n]["group"] == "real":
            ax.annotate(n.replace("sub", ""), (Y[i, 0], Y[i, 1]),
                        fontsize=8, ha="center", va="center", color="white", zorder=4)
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}% var)")
    ax.set_title("Body shape space: real training people vs. the synthetic bodies")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_anthropometry(rows, out):
    import matplotlib.pyplot as plt
    metrics = [("height", "height (m)"), ("arm_span", "arm span (m)"),
               ("shoulder_w", "shoulder width (m)"), ("hip_w", "hip width (m)"),
               ("depth", "body depth (m)"), ("volume", "hull volume (m^3, mass proxy)")]
    groups = GROUPS                      # one violin column per category, real then synthetic
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        data = [[r[key] for r in rows if r["group"] == g and not np.isnan(r[key])]
                for g in groups]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(GROUP_COLOR[g]); pc.set_alpha(0.55)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([GROUP_LABEL[g] for g in groups], rotation=12, fontsize=8)
        ax.set_title(label); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Anthropometry from the SMPL-X mesh: real training people vs synthetic",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_betas_violin(bodies, out):
    import matplotlib.pyplot as plt
    real = np.stack([b["betas"] for b in bodies.values() if b["group"] == "real"])
    syn = np.stack([b["betas"] for b in bodies.values() if b["group"] != "real"])
    n = real.shape[1]
    fig, ax = plt.subplots(figsize=(14, 6))
    pos = np.arange(n)
    vp_r = ax.violinplot([real[:, i] for i in range(n)], positions=pos - 0.18,
                         widths=0.32, showmeans=True)
    vp_s = ax.violinplot([syn[:, i] for i in range(n)], positions=pos + 0.18,
                         widths=0.32, showmeans=True)
    for pc in vp_r["bodies"]:
        pc.set_facecolor(GROUP_COLOR["real"]); pc.set_alpha(0.6)
    for pc in vp_s["bodies"]:
        pc.set_facecolor(GROUP_COLOR["prior"]); pc.set_alpha(0.6)
    ax.set_xticks(pos); ax.set_xticklabels([f"b{i}" for i in range(n)])
    ax.set_xlabel("SMPL-X beta dimension"); ax.set_ylabel("value")
    ax.set_title("Per-beta spread: real (blue) vs synthetic (orange) -- "
                 "which shape axes the augmentation stretches")
    ax.grid(alpha=0.25, axis="y")
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=GROUP_COLOR["real"]), Patch(facecolor=GROUP_COLOR["prior"])],
              ["real", "synthetic"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_coverage(bodies, out):
    """One separation, in raw 16-D beta space (that's the conditioning input):
    each synthetic -> nearest real TRAINING body. Near 0 = a duplicate of a
    real person, large = a shape no training subject has. The reference line
    is the smallest distance between two TRAINING reals ("how close two
    different people in the training set get"). Only training bodies are
    involved."""
    import matplotlib.pyplot as plt
    names = list(bodies)
    B = {n: bodies[n]["betas"] for n in names}
    real = [n for n in names if bodies[n]["group"] == "real"]
    syn = [n for n in names if bodies[n]["group"] != "real"]

    def nn_dist(a, pool):
        return min(np.linalg.norm(B[a] - B[p]) for p in pool)

    syn_to_real = [nn_dist(s, real) for s in syn]
    train_floor = min(np.linalg.norm(B[a] - B[b])
                      for i, a in enumerate(real) for b in real[i + 1:])

    fig, ax1 = plt.subplots(figsize=(7, 5.5))
    ax1.hist(syn_to_real, bins=12, color=GROUP_COLOR["prior"], edgecolor="k", alpha=0.8)
    ax1.axvline(np.mean(syn_to_real), color="k", ls="--",
                label=f"mean {np.mean(syn_to_real):.2f}")
    ax1.axvline(train_floor, color="#1b6fb3", ls=":",
                label=f"closest two training reals = {train_floor:.2f}")
    ax1.set_xlabel("L2 betas distance to nearest REAL training body")
    ax1.set_ylabel("# synthetic bodies")
    ax1.set_title("Synthetic -> nearest real training body\n(0 = duplicate, large = novel shape)")
    ax1.legend(); ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return syn_to_real, train_floor


# ------------------------------------------------------------------- self-check
def selfcheck_heights(rows, heights_json):
    """Our mesh height for sub100..139 must match the stored synthetic_heights.json.

    This validates the whole measurement path (template + shapedirs + betas) against
    numbers produced independently at generation time. Fails loudly on drift.
    """
    stored = json.load(open(heights_json))
    worst = 0.0
    for r in rows:
        if sub_num(r["name"]) >= 100 and str(sub_num(r["name"])) in stored:
            worst = max(worst, abs(r["height"] - stored[str(sub_num(r["name"]))]))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--betas", default="scripts/omomo_betas_neutral_aug_gen4.npz")
    ap.add_argument("--kinds", default="scripts/synthetic_bodies_gen4.npz")
    ap.add_argument("--models-dir", default=str(Path.home() / "Downloads/models/smplx"))
    ap.add_argument("--heights", default="scripts/synthetic_heights_gen4.json")
    ap.add_argument("--out", default=str(Path.home() / "synthetic_body_analysis"))
    ap.add_argument("--reals", nargs="+", default=TRAIN_REALS,
                    help="the real TRAINING subjects; any other real in --betas "
                         "(the test bodies) is ignored entirely")
    a = ap.parse_args()
    TRAIN_REALS[:] = a.reals          # module-level list read by load_bodies

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bodies = load_bodies(a.betas, a.kinds)
    print(f"[bodies] {len(a.reals)} training reals + "
          f"{sum(b['group'] != 'real' for b in bodies.values())} synthetics "
          f"(test bodies are not loaded)")
    v_template, shapedirs, j_reg = load_smplx(
        a.models_dir, n_betas=len(next(iter(bodies.values()))["betas"]))

    # measure every body
    rows = []
    for n, b in bodies.items():
        V = shaped_vertices(v_template, shapedirs, b["betas"])
        joints = j_reg @ V                      # (J,V)@(V,3) -> (J,3) joint centers
        m = measure(V, joints)
        rows.append({"name": n, "group": b["group"], **m})

    # self-check before trusting any plot
    worst = selfcheck_heights(rows, a.heights)
    print(f"[selfcheck] max |mesh height - stored height| over synthetics = {worst*1000:.1f} mm")
    if worst > 0.01:   # 1 cm tolerance
        raise SystemExit(f"ERROR: height self-check failed ({worst*1000:.1f} mm > 10 mm) "
                         "-- measurement path disagrees with generation; not writing plots.")

    # figures
    plot_pca(bodies, out / "betas_pca.png")
    plot_anthropometry(rows, out / "anthropometry.png")
    plot_betas_violin(bodies, out / "betas_violin.png")
    syn_to_real, train_floor = plot_coverage(bodies, out / "coverage.png")

    # summary csv
    cols = ["name", "group", "height", "arm_span", "shoulder_w",
            "hip_w", "depth", "volume"]
    with open(out / "body_measurements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: sub_num(r["name"])):
            w.writerow({k: r[k] for k in cols})

    # console summary
    print(f"\n[coverage] synthetic -> nearest real training body: min {min(syn_to_real):.2f} "
          f"mean {np.mean(syn_to_real):.2f} max {max(syn_to_real):.2f}")
    print(f"[coverage] closest two training reals = {train_floor:.2f} (reference only)")
    print(f"\nwrote figures + body_measurements.csv to {out}")


if __name__ == "__main__":
    main()

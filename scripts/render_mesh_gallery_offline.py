#!/usr/bin/env python3
"""Render every subject's shaped SMPL-X SURFACE mesh side by side, offline (no
Isaac Gym). Static rest pose -> only body SHAPE varies, so subjects are easy to
tell apart. Runs anywhere the SMPL-X models are (this laptop).

  python3 scripts/render_mesh_gallery_offline.py --out mesh_gallery.png
  python3 scripts/render_mesh_gallery_offline.py --subjects sub4 sub10 sub13 sub16 --out g.png
  # gen4 training set (14 reals + 30 prior-sampled synthetics), 11 per row:
  python3 scripts/render_mesh_gallery_offline.py --betas scripts/omomo_betas_neutral_aug_gen4.npz \
      --subjects sub1 sub2 ... sub629 --ncol 11 --out gen4_training_bodies.png
"""
import argparse
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _shaper(models, betas):
    spec = importlib.util.spec_from_file_location(
        "smplx_mesh", os.path.join(os.path.dirname(__file__), "smplx_mesh.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kw = {"betas_path": betas}
    if models:
        kw["models_dir"] = models
    return m.SMPLXShaper(**kw)


def shade(v, f, light=np.array([0.4, -0.7, 0.6])):
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = light / np.linalg.norm(light)
    return 0.30 + 0.65 * np.clip(n @ light, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out", default="mesh_gallery.png")
    ap.add_argument("--models", default=None)
    ap.add_argument("--betas", default="scripts/omomo_betas.npz")
    ap.add_argument("--spacing", type=float, default=0.75, help="metres between bodies")
    ap.add_argument("--ncol", type=int, default=None,
                    help="bodies per row (default: all in one row)")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()

    sh = _shaper(a.models, a.betas)
    subs = a.subjects or sh.subjects()
    print(f"[gallery] {len(subs)} bodies: {' '.join(subs)}", flush=True)
    cmap = matplotlib.colormaps["Blues"]

    # One 3D subplot PER body: matplotlib's 3D aspect handling crushes many bodies
    # in a single axis, but renders one body cleanly. Lay them out in a grid.
    ncol = a.ncol or len(subs)
    nrow = -(-len(subs) // ncol)                # ceil division
    fig = plt.figure(figsize=(1.5 * ncol, 4.2 * nrow), dpi=a.dpi)
    for i, s in enumerate(subs):
        v, f = sh.mesh(s)
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.set_axis_off()
        c = shade(v, f)
        pc = Poly3DCollection(v[f], linewidths=0)
        pc.set_facecolor(cmap(0.35 + 0.5 * (c - c.min()) / (c.ptp() + 1e-9)))
        ax.add_collection3d(pc)
        # Same box for every body so sizes are comparable across panels: wide
        # enough for a T-pose arm span (~1.8 m) and tall enough for 2.1 m.
        ctr = v.mean(0)
        r = 0.95                                # lateral half-extent
        ax.set_xlim(ctr[0]-r, ctr[0]+r)
        ax.set_ylim(ctr[1]-r, ctr[1]+r)
        ax.set_zlim(0, 2.1)
        try:
            ax.set_box_aspect((1, 1, 2.1 / (2 * r)))
        except Exception:
            pass
        ax.view_init(elev=6, azim=-89)          # front view
        ax.set_title(f"{s}\n{sh.gender[s]} {v[:,2].max():.2f}m", fontsize=8, pad=0)
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.95, wspace=0.0, hspace=0.15)
    out = os.path.abspath(a.out)
    fig.savefig(out, dpi=a.dpi)
    print(f"[gallery] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

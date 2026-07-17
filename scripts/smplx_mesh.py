#!/usr/bin/env python3
"""Generate shaped SMPL-X SURFACE meshes from per-subject betas.

The Isaac Gym MJCFs are capsule/sphere approximations. This produces the actual
SMPL-X body surface for each subject, so bodies can be compared as meshes rather
than capsules. Rest pose (zero pose params) => only the SHAPE (betas) varies,
which is exactly the body-comparison we want.

No smplx package needed: the shaped rest mesh is v_template + shapedirs @ betas,
straight from the model .npz. Uses the GENDERED model per subject (SMPLX_MALE/
FEMALE), matching how the per-subject MJCFs were built (betas are gendered).

Output vertices are Z-up (Isaac Gym convention) with feet on z=0.

Standalone: writes one OBJ per subject.
  python3 scripts/smplx_mesh.py --subjects sub4 sub16 --out-dir /tmp/meshes
"""
import argparse
import os

import numpy as np

DEFAULT_MODELS = os.path.expanduser(os.environ.get("SMPLX_MODELS",
                                                   "~/Downloads/models/smplx"))
DEFAULT_BETAS = "scripts/omomo_betas.npz"
N_BETAS = 16


def _gender_map(betas_npz):
    """betas npz stores gender as a '_genders' array of 'subN:gender' strings."""
    if "_genders" not in betas_npz.files:
        raise SystemExit(f"FATAL: no '_genders' in betas file; cannot pick the "
                         f"gendered SMPL-X model per subject.")
    return dict(x.split(":") for x in betas_npz["_genders"])


class SMPLXShaper:
    """Loads gendered SMPL-X models lazily and produces shaped rest-pose meshes."""

    def __init__(self, models_dir=DEFAULT_MODELS, betas_path=DEFAULT_BETAS):
        self.models_dir = os.path.expanduser(models_dir)
        if not os.path.isdir(self.models_dir):
            raise SystemExit(f"FATAL: SMPL-X models dir not found: {self.models_dir}\n"
                             f"  set SMPLX_MODELS or pass --models; needs "
                             f"SMPLX_MALE.npz / SMPLX_FEMALE.npz / SMPLX_NEUTRAL.npz")
        self.betas = np.load(betas_path, allow_pickle=True)
        self.gender = _gender_map(self.betas)
        self._cache = {}

    def _model(self, gender):
        g = gender.upper()
        if g not in self._cache:
            p = os.path.join(self.models_dir, f"SMPLX_{g}.npz")
            if not os.path.isfile(p):
                raise SystemExit(f"FATAL: missing model {p}")
            m = np.load(p, allow_pickle=True)
            self._cache[g] = (m["v_template"].astype(np.float64),
                              m["shapedirs"].astype(np.float64)[:, :, :N_BETAS],
                              m["f"].astype(np.uint32))
        return self._cache[g]

    def subjects(self):
        return sorted((k for k in self.betas.files if k.startswith("sub")),
                      key=lambda s: int(s[3:]))

    def mesh(self, subject):
        """Return (vertices Nx3 float32 Z-up feet-on-ground, faces Mx3 uint32)."""
        if subject not in self.betas.files:
            raise SystemExit(f"FATAL: no betas for {subject}")
        if subject not in self.gender:
            raise SystemExit(f"FATAL: no gender for {subject}")
        vt, sd, f = self._model(self.gender[subject])
        beta = self.betas[subject].astype(np.float64)[:N_BETAS]
        v = vt + sd @ beta                          # shaped, SMPL-X frame (Y-up)

        # SMPL-X is Y-up; Isaac Gym is Z-up. Rotate +90deg about X: (x,y,z)->(x,-z,y).
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
        v[:, 2] -= v[:, 2].min()                    # feet on the ground plane
        v[:, 0] -= v[:, 0].mean()                   # centre laterally
        v[:, 1] -= v[:, 1].mean()
        return v.astype(np.float32), f


def write_obj(path, v, f):
    with open(path, "w") as fh:
        for x, y, z in v:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in f:                           # OBJ is 1-indexed
            fh.write(f"f {a+1} {b+1} {c+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--betas", default=DEFAULT_BETAS)
    ap.add_argument("--out-dir", default="smplx_meshes")
    a = ap.parse_args()

    sh = SMPLXShaper(a.models, a.betas)
    subs = a.subjects or sh.subjects()
    os.makedirs(a.out_dir, exist_ok=True)
    for s in subs:
        v, f = sh.mesh(s)
        out = os.path.join(a.out_dir, f"{s}.obj")
        write_obj(out, v, f)
        print(f"  {s:7s} gender={sh.gender[s]:7s} height={v[:,2].max():.3f}m "
              f"verts={len(v)} -> {out}")


if __name__ == "__main__":
    main()

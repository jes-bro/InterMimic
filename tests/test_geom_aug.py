#!/usr/bin/env python3
"""Offline checks for anisotropic geometry augmentation (no Isaac Gym).

Validates the pieces that can't be exercised without the simulator:
  1. variant-URDF generation: the string swap produces valid XML with the per-axis
     scale on BOTH the visual+collision mesh, and leaves the .obj filename intact,
     for every REAL object URDF in the repo.
  2. aniso table: seeded + reproducible, variant 0 is exactly identity, all values in
     [anisoMin, anisoMax].
  3. per-env combined point scale = uniform objectAug scale * per-env aniso.
  4. mass factor V**(massExp/3 - 1): solid (massExp=3) is a no-op for ANY geometry;
     the uniform special case reduces to the old aug**(massExp-3) formula.
"""
import glob
import os
import sys
import xml.etree.ElementTree as ET

import torch

OBJDIR = os.path.join(os.path.dirname(__file__), os.pardir,
                      "isaacgym/src/intermimic/data/assets/objects")

fails = 0
def check(name, ok):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name)
    fails += (0 if ok else 1)


# --- mirror of InterMimic._write_geom_variant_urdf's core swap ---
def variant_text(base_txt, sx, sy, sz):
    return base_txt.replace('scale="1.0 1.0 1.0"', f'scale="{sx:.6f} {sy:.6f} {sz:.6f}"')


# 1) real URDFs -> valid variant XML, scales swapped, mesh path preserved.
urdfs = sorted(glob.glob(os.path.join(OBJDIR, "*.urdf")))
urdfs = [u for u in urdfs if ".geomv" not in os.path.basename(u)]
check("found object URDFs to test", len(urdfs) > 0)
for u in urdfs:
    base = open(u).read()
    n_scale = base.count('scale="1.0 1.0 1.0"')
    vt = variant_text(base, 0.8, 1.1, 1.25)
    try:
        root = ET.fromstring(vt)
    except ET.ParseError as e:
        check(f"{os.path.basename(u)} variant is valid XML", False); continue
    meshes = root.findall(".//mesh")
    scales = [m.get("scale") for m in meshes]
    ok = (n_scale > 0
          and all(s == "0.800000 1.100000 1.250000" for s in scales if s is not None)
          and 'scale="1.0 1.0 1.0"' not in vt
          and all(".obj" in (m.get("filename") or "") for m in meshes))
    check(f"{os.path.basename(u)}: {len(meshes)} mesh(es) rescaled, .obj path intact", ok)


# --- mirror of the aniso-table construction (seed 6789) ---
def build_aniso(nobj, nvar, amin, amax, seed=6789):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(nobj, nvar, 3, generator=g) * (amax - amin) + amin
    if nvar >= 1:
        a[:, 0, :] = 1.0
    return a

# 2) reproducible + v0 identity + in range.
a1 = build_aniso(19, 8, 0.80, 1.20)
a2 = build_aniso(19, 8, 0.80, 1.20)
check("aniso table reproducible under fixed seed", torch.equal(a1, a2))
check("variant 0 is exactly identity (1,1,1)", torch.allclose(a1[:, 0, :], torch.ones(19, 3)))
non0 = a1[:, 1:, :]
check("variants 1.. within [anisoMin, anisoMax]",
      bool((non0 >= 0.80 - 1e-6).all() and (non0 <= 1.20 + 1e-6).all()))
check("variants 1.. are actually anisotropic (not all equal per axis)",
      bool((non0[..., 0] != non0[..., 1]).any()))

# 3) per-env combined point scale.
ne, nobj, nvar = 32, 19, 8
g = torch.Generator().manual_seed(6789)
_ = torch.rand(nobj, nvar, 3, generator=g)          # consume like the real code path
env_variant = torch.randint(0, nvar, (ne,), generator=g)
aniso = build_aniso(nobj, nvar, 0.80, 1.20)
objidx = torch.arange(ne) % nobj
aniso_pe = aniso[objidx, env_variant]               # (ne, 3)
oa_scale = torch.rand(ne)                            # uniform objectAug scale
obj_pts_scale = oa_scale.view(-1, 1, 1) * aniso_pe.view(-1, 1, 3)   # (ne,1,3)
check("combined scale shape (ne,1,3)", tuple(obj_pts_scale.shape) == (ne, 1, 3))
# a point cloud (ne,1024,3) * (ne,1,3) broadcasts correctly and equals uniform*aniso.
pts = torch.randn(ne, 1024, 3)
scaled = pts * obj_pts_scale
expect0 = pts[0] * (oa_scale[0] * aniso_pe[0])
check("broadcast applies uniform*aniso per axis", torch.allclose(scaled[0], expect0, atol=1e-6))

# 4) mass factor V**(massExp/3 - 1).
def corr(sx, sy, sz, aug, massExp):
    V = (sx * sy * sz) * (aug ** 3)
    return V ** (massExp / 3.0 - 1.0)

check("solid (massExp=3) => no correction for ANY geometry",
      abs(corr(0.8, 1.2, 1.1, 1.3, 3.0) - 1.0) < 1e-9)
# uniform special case (sx=sy=sz=aug... i.e. identity aniso, aug uniform) reduces to
# the old formula aug**(massExp-3).
aug = 1.15
check("uniform case reduces to aug**(massExp-3)",
      abs(corr(1.0, 1.0, 1.0, aug, 2.0) - aug ** (2.0 - 3.0)) < 1e-9)
check("shell (massExp=2) makes a bigger object relatively lighter than solid",
      corr(1.2, 1.2, 1.2, 1.0, 2.0) < 1.0)

print()
if fails:
    print(f"FAILED ({fails})"); sys.exit(1)
print("ALL GREEN -- geometry-augmentation offline logic verified.")

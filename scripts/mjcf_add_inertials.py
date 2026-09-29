#!/usr/bin/env python3
"""Write an explicit <inertial> element into every body of a per-subject MJCF.

WHY. PhysX derives each link's mass frame from its collision shapes. For an
axially symmetric capsule the two transverse principal moments are equal, so
the principal-axis frame it picks is numerically arbitrary, and the joint-angle
extraction for the 3-hinge joints is done relative to that frame. On sub4 the
left-knee twist readout wraps by 2*pi at a written angle of ~-0.083 rad and the
+/-pi joint limit then slams the leg at 40-117 rad/s (replay traces, 2026-09-28);
swapping the left leg with sub2's moves the fault with it, and scaling the shin
radius by 1.02 removes it -- i.e. it is the frame PhysX happens to pick for
those exact numbers, not the person. Writing mass, centre of mass, principal
axes and moments explicitly leaves PhysX nothing arbitrary to choose. The
values are the same ones PhysX would compute from the geoms (capsule =
cylinder + two hemispheres, sphere, box), so the physics is unchanged up to
rounding.

  python3 scripts/mjcf_add_inertials.py --subject sub4 \
      --out isaacgym/src/intermimic/data/assets/smplx/smplx_variant_sub4_inertial.xml
  python3 scripts/mjcf_add_inertials.py --in some.xml --out other.xml   # any MJCF

MJCF <inertial pos quat mass diaginertia>: quat is (w x y z); diaginertia is
listed in the frame given by quat. The principal frame is chosen deterministically
(eigenvectors of the body-frame tensor, ascending moments, right-handed).
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "isaacgym/src/intermimic/data/assets/smplx"
DEFAULT_DENSITY = 1000.0   # MJCF default when a geom carries no density attribute


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def mat_to_quat_wxyz(R):
    """Rotation matrix -> (w,x,y,z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def geom_mass_com_inertia(g):
    """(mass, com (3,), inertia tensor about com in the BODY frame (3,3)) of one geom."""
    typ = g.get("type"); rho = float(g.get("density", DEFAULT_DENSITY))
    size = [float(v) for v in g.get("size").split()]
    if typ == "capsule" and g.get("fromto"):
        ft = np.array([float(v) for v in g.get("fromto").split()]); a, b = ft[:3], ft[3:]
        r = size[0]; L = float(np.linalg.norm(b - a)); u = (b - a) / L if L > 1e-12 else np.array([0, 0, 1.0])
        mc = rho * math.pi * r * r * L; ms = rho * (4.0 / 3.0) * math.pi * r ** 3   # cylinder + both hemispheres
        Ia = 0.5 * mc * r * r + 0.4 * ms * r * r
        d = L / 2 + 3 * r / 8                                                          # hemisphere com offset from capsule com
        It = mc * (r * r / 4 + L * L / 12) + 2 * ((83.0 / 320.0) * (ms / 2) * r * r + (ms / 2) * d * d)
        uu = np.outer(u, u); I = Ia * uu + It * (np.eye(3) - uu)
        return mc + ms, (a + b) / 2, I
    if typ == "sphere":
        r = size[0]; m = rho * (4.0 / 3.0) * math.pi * r ** 3
        return m, np.array([float(v) for v in g.get("pos", "0 0 0").split()]), (0.4 * m * r * r) * np.eye(3)
    if typ == "box":
        sx, sy, sz = size[:3]; m = rho * 8 * sx * sy * sz
        Ib = (m / 3.0) * np.diag([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy])
        R = quat_wxyz_to_mat([float(v) for v in g.get("quat").split()]) if g.get("quat") else np.eye(3)
        return m, np.array([float(v) for v in g.get("pos", "0 0 0").split()]), R @ Ib @ R.T
    raise SystemExit(f"ERROR: unsupported geom {typ!r} (attrs {sorted(g.attrib)})")


def body_inertial(body):
    parts = [geom_mass_com_inertia(g) for g in body.findall("geom")]
    if not parts:
        return None
    M = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / M
    I = np.zeros((3, 3))
    for m, c, Ig in parts:
        d = c - com; I += Ig + m * (float(d @ d) * np.eye(3) - np.outer(d, d))   # parallel axis
    vals, vecs = np.linalg.eigh(I)                          # ascending, orthonormal columns
    # Choose THE principal frame nearest the body frame. Eigenvectors are only
    # defined up to sign (and, for a capsule's equal transverse moments, up to
    # rotation within that pair), and a frame ~180 deg from the body frame has a
    # sign-ambiguous quaternion -- the very thing that wraps the joint angles.
    # Greedy: assign to body axis x, y, z the eigenvector most aligned with it,
    # sign-flipped to point along it; fix handedness on the least-determined axis.
    cols, moms, used = [], [], set()
    for ax in range(3):
        best = max((j for j in range(3) if j not in used), key=lambda j: abs(vecs[ax, j]))
        v = vecs[:, best] * (1.0 if vecs[ax, best] >= 0 else -1.0)
        cols.append(v); moms.append(vals[best]); used.add(best)
    R = np.stack(cols, axis=1)
    if np.linalg.det(R) < 0:
        # flipping a column that belongs to a degenerate pair costs nothing physically;
        # otherwise flip the one least aligned with its body axis
        pair = [k for k in range(3) if any(abs(moms[k] - moms[m]) < 1e-9 * max(1.0, moms[k]) for m in range(3) if m != k)]
        k = pair[0] if pair else int(np.argmin([abs(R[ax, ax]) for ax in range(3)]))
        R[:, k] *= -1
    return M, com, np.array(moms), mat_to_quat_wxyz(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="e.g. sub4 -> reads smplx_omomo_sub4.xml from the assets dir")
    ap.add_argument("--in", dest="inp", type=Path, help="any MJCF path (alternative to --subject)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = a.inp or (ASSETS / f"smplx_omomo_{a.subject}.xml")
    tree = ET.parse(src); root = tree.getroot()
    n, total = 0, 0.0
    for body in root.iter("body"):
        for old in body.findall("inertial"):
            body.remove(old)
        res = body_inertial(body)
        if res is None:
            continue
        M, com, vals, q = res
        el = ET.Element("inertial")
        el.set("pos", " ".join(f"{v:.6f}" for v in com))
        el.set("quat", " ".join(f"{v:.8f}" for v in q))
        el.set("mass", f"{M:.6f}")
        el.set("diaginertia", " ".join(f"{max(v, 1e-9):.9f}" for v in vals))
        body.insert(0, el)                                  # first child, before geoms/joints
        n += 1; total += M
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(a.out, encoding="unicode", xml_declaration=False)
    print(f"wrote {a.out}: explicit inertials on {n} bodies, total mass {total:.2f} kg (from {src.name})")


if __name__ == "__main__":
    main()

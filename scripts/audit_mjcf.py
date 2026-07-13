#!/usr/bin/env python3
"""Audit the per-subject SMPL-X MJCFs and flag physically anomalous bodies.

Motivation: in a cross-body eval (subjectBodies=[body], dataSub=[source]) the
subject's own motion is never used -- only its MJCF. So when one body fails and
its neighbours don't, the MJCF is the prime suspect. sub4 is a known crasher;
sub16 is a known "hard" body whose cause is still unexplained and for which the
betas-distance hypothesis is refuted (sub16 is the 14th-most-typical of 17).

What it measures, per subject:
  total mass        sum(density * volume) over geoms -- MJCF has no explicit mass
  height            global z-extent of the geom hull (root at origin)
  limb lengths      parent->child body offsets, and the arm/leg chains
  topology          body / geom / joint / motor counts (must be identical: the
                    policy's action vector is shared across bodies)
  degeneracies      NaN, non-positive radii, zero-length capsules, absurd density

Outliers are flagged with a ROBUST z-score (median / MAD), so one broken body
can't inflate the mean and hide itself.

  python3 scripts/audit_mjcf.py --glob 'isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml'
  python3 scripts/audit_mjcf.py --highlight sub16 sub4
"""
import argparse
import glob as globmod
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

SPHERE, CAPSULE, BOX = "sphere", "capsule", "box"


def _f(s):
    return [float(x) for x in s.replace(",", " ").split()]


def geom_volume(g):
    """MJCF geoms carry density + shape, not mass -- volume must be derived.

    `g` is a plain attrib dict. Returns (volume, radius, capsule_length).
    """
    t = g.get("type", SPHERE)
    size = _f(g.get("size", "0"))
    if t == SPHERE:
        r = size[0]
        return (4.0 / 3.0) * math.pi * r ** 3, r, 0.0
    if t == CAPSULE:
        r = size[0]
        if "fromto" in g:
            p = _f(g["fromto"])
            L = float(np.linalg.norm(np.array(p[3:6]) - np.array(p[:3])))
        else:
            L = 2.0 * size[1] if len(size) > 1 else 0.0
        # cylinder + two hemispherical caps
        return math.pi * r ** 2 * L + (4.0 / 3.0) * math.pi * r ** 3, r, L
    if t == BOX:
        return 8.0 * size[0] * size[1] * size[2], min(size), 0.0
    return 0.0, 0.0, 0.0


def parse_mjcf(path):
    root = ET.parse(path).getroot()
    wb = root.find("worldbody")

    bodies, geoms, problems = OrderedDict(), [], []
    pts = []           # global geom endpoints, for the height/extent hull

    def walk(elem, parent_pos, parent_name):
        for b in elem.findall("body"):
            name = b.get("name", "?")
            pos = np.array(_f(b.get("pos", "0 0 0")))
            gpos = parent_pos + pos          # 'coordinate=local' -> offsets accumulate
            bodies[name] = {"global": gpos, "parent": parent_name,
                            "offset": float(np.linalg.norm(pos))}
            for g in b.findall("geom"):
                vol, r, L = geom_volume(g.attrib)
                dens = float(g.get("density", 0.0))
                m = vol * dens
                geoms.append({"body": name, "type": g.get("type"), "mass": m,
                              "density": dens, "r": r, "L": L})
                # Degeneracy checks -- these are what a bad betas->MJCF fit produces.
                if not np.isfinite([vol, dens, m]).all():
                    problems.append(f"{name}: NaN/Inf (vol={vol} dens={dens})")
                if r <= 0:
                    problems.append(f"{name}: non-positive radius {r}")
                if g.get("type") == CAPSULE and L <= 1e-6:
                    problems.append(f"{name}: zero-length capsule")
                if dens <= 0 or dens > 20000:
                    problems.append(f"{name}: implausible density {dens:.0f}")
                # geom hull points, in global frame
                if "fromto" in g.attrib:
                    p = _f(g.get("fromto"))
                    pts.append(gpos + np.array(p[:3]))
                    pts.append(gpos + np.array(p[3:6]))
                else:
                    off = np.array(_f(g.get("pos", "0 0 0")))
                    pts.append(gpos + off + np.array([0, 0, r]))
                    pts.append(gpos + off - np.array([0, 0, r]))
            walk(b, gpos, name)

    walk(wb, np.zeros(3), None)
    P = np.array(pts)

    def chain(names):
        """Summed offsets along a named body chain (arm/leg length)."""
        return float(sum(bodies[n]["offset"] for n in names if n in bodies))

    return {
        "path": path,
        "n_body": len(bodies),
        "n_geom": len(geoms),
        "n_joint": len(root.findall(".//joint")),
        "n_motor": len(root.findall(".//motor")),
        "mass": sum(g["mass"] for g in geoms),
        "height": float(P[:, 2].max() - P[:, 2].min()) if len(P) else float("nan"),
        "width": float(P[:, 1].max() - P[:, 1].min()) if len(P) else float("nan"),
        "leg": chain(["L_Knee", "L_Ankle", "L_Toe"]),
        "arm": chain(["L_Elbow", "L_Wrist"]),
        "torso": chain(["Torso", "Spine", "Chest", "Neck", "Head"]),
        "problems": problems,
        "bodies": bodies,
        "geoms": geoms,
    }


def robust_z(v):
    """Median/MAD z-score: one broken body can't inflate the mean and hide itself."""
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        return np.zeros_like(v)
    return (v - med) / scale


def subject_of(path):
    m = re.search(r"(sub\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml")
    ap.add_argument("--highlight", nargs="*", default=["sub4", "sub16"],
                    help="subjects to call out explicitly in the report")
    ap.add_argument("--z", type=float, default=3.0, help="robust-z threshold to flag")
    a = ap.parse_args()

    paths = sorted(globmod.glob(a.glob), key=lambda p: int(re.search(r"sub(\d+)", p).group(1))
                   if re.search(r"sub(\d+)", p) else 0)
    if not paths:
        raise SystemExit(f"FATAL: no MJCFs matched {a.glob!r}. On the cluster these live "
                         f"under isaacgym/src/intermimic/data/assets/smplx/ -- generate them first.")

    R = [parse_mjcf(p) for p in paths]
    subs = [subject_of(p) for p in paths]
    METRICS = ["mass", "height", "width", "leg", "arm", "torso"]
    Z = {m: robust_z([r[m] for r in R]) for m in METRICS}

    # --- topology must be IDENTICAL across bodies: the shared policy emits one
    # action vector for all of them (humanoid.py asserts this at load).
    print("=" * 96)
    print("TOPOLOGY  (must be identical across subjects -- the action vector is shared)")
    print("=" * 96)
    topo = {}
    for s, r in zip(subs, R):
        topo.setdefault((r["n_body"], r["n_geom"], r["n_joint"], r["n_motor"]), []).append(s)
    for k, v in topo.items():
        tag = "OK" if len(topo) == 1 else "<<< MISMATCH"
        print(f"  bodies={k[0]:3d} geoms={k[1]:3d} joints={k[2]:3d} motors={k[3]:3d}  "
              f"[{len(v):2d}] {' '.join(v)}  {tag}")
    if len(topo) > 1:
        print("\n  !! Subjects do not share topology -- a policy trained on one CANNOT drive another.")

    print()
    print("=" * 96)
    print(f"PER-SUBJECT METRICS   (z = robust median/MAD z-score; |z| > {a.z} flagged)")
    print("=" * 96)
    hdr = f"{'body':7s} {'mass_kg':>8s} {'z':>6s} {'height_m':>9s} {'z':>6s} {'leg_m':>6s} {'z':>6s} {'arm_m':>6s} {'z':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(subs):
        flags = [m for m in METRICS if abs(Z[m][i]) > a.z]
        mark = ""
        if flags:
            mark = "  <<< OUTLIER: " + ",".join(flags)
        elif s in a.highlight:
            mark = "  <-- watch"
        print(f"{s:7s} {R[i]['mass']:8.1f} {Z['mass'][i]:6.1f} {R[i]['height']:9.3f} "
              f"{Z['height'][i]:6.1f} {R[i]['leg']:6.3f} {Z['leg'][i]:6.1f} "
              f"{R[i]['arm']:6.3f} {Z['arm'][i]:6.1f}{mark}")

    print()
    print("=" * 96)
    print("DEGENERACIES  (NaN, non-positive radius, zero-length capsule, absurd density)")
    print("=" * 96)
    any_bad = False
    for s, r in zip(subs, R):
        if r["problems"]:
            any_bad = True
            print(f"  {s}: {len(r['problems'])} problem(s)")
            for p in r["problems"][:8]:
                print(f"      - {p}")
    if not any_bad:
        print("  none -- every MJCF is geometrically well-formed.")

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    for s in a.highlight:
        if s not in subs:
            print(f"  {s}: NOT FOUND in {a.glob}")
            continue
        i = subs.index(s)
        flags = [f"{m} (z={Z[m][i]:+.1f})" for m in METRICS if abs(Z[m][i]) > a.z]
        if R[i]["problems"]:
            print(f"  {s}: MALFORMED -- {len(R[i]['problems'])} degeneracy(ies). This is the bug.")
        elif flags:
            print(f"  {s}: physically anomalous -- {', '.join(flags)}")
        else:
            print(f"  {s}: MJCF is clean and unremarkable (no degeneracies, no |z|>{a.z} outliers).")
            print(f"       -> the body geometry does NOT explain its failure. Look at the")
            print(f"          termination reason instead (TERM_REASON=1).")


if __name__ == "__main__":
    main()

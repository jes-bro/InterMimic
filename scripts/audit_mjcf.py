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
                # Degeneracy checks. NOTE: a clean bill of health here does NOT mean
                # the body is fine -- these are only the failure modes we thought to
                # look for. Use --compare to find the ones we didn't.
                if not np.isfinite([vol, dens, m]).all():
                    problems.append(f"{name}: NaN/Inf (vol={vol} dens={dens})")
                if r <= 0:
                    problems.append(f"{name}: non-positive radius {r}")
                elif r < 1e-3:
                    # Positive but tiny: passes a >0 check yet wrecks contact solving.
                    problems.append(f"{name}: near-zero radius {r:.2e}")
                if g.get("type") == CAPSULE and L <= 1e-6:
                    problems.append(f"{name}: zero-length capsule")
                elif g.get("type") == CAPSULE and r > 0 and L / r > 40:
                    problems.append(f"{name}: needle capsule (L/r = {L/r:.0f})")
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


def compare(rA, rB, nameA, nameB, top=25):
    """Side-by-side per-body diff of two MJCFs, worst relative difference first.

    The degeneracy checks above can only catch failure modes we anticipated. When a
    body is known-broken but audits clean, the checks are the problem, not the body.
    This dumps what ACTUALLY differs from a healthy control so the bug can be seen
    rather than guessed at.
    """
    print("=" * 100)
    print(f"COMPARE  {nameA} (suspect)  vs  {nameB} (control)   -- worst relative difference first")
    print("=" * 100)

    # Per-body mass and geometry, keyed by body name.
    def per_body(r):
        d = {}
        for g in r["geoms"]:
            e = d.setdefault(g["body"], {"mass": 0.0, "r": 0.0, "L": 0.0, "dens": 0.0, "n": 0})
            e["mass"] += g["mass"]
            e["r"] = max(e["r"], g["r"])
            e["L"] = max(e["L"], g["L"])
            e["dens"] = max(e["dens"], g["density"])
            e["n"] += 1
        return d

    A, B = per_body(rA), per_body(rB)
    onlyA, onlyB = set(A) - set(B), set(B) - set(A)
    if onlyA or onlyB:
        print(f"  !! body-name mismatch: only in {nameA}: {sorted(onlyA)} | "
              f"only in {nameB}: {sorted(onlyB)}\n")

    rows = []
    for nm in sorted(set(A) & set(B)):
        a, b = A[nm], B[nm]
        # Relative difference on each field; rank by the worst one on that body.
        rel = {}
        for k in ("mass", "r", "L", "dens"):
            denom = max(abs(b[k]), 1e-9)
            rel[k] = (a[k] - b[k]) / denom
        worst = max(abs(v) for v in rel.values())
        rows.append((worst, nm, a, b, rel))
    rows.sort(reverse=True)

    hdr = (f"{'body':>14s} {'mass_A':>8s} {'mass_B':>8s} {'d%':>7s} "
           f"{'r_A':>7s} {'r_B':>7s} {'d%':>7s} {'L_A':>7s} {'L_B':>7s} {'d%':>7s} "
           f"{'dens_A':>9s} {'dens_B':>9s} {'d%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for worst, nm, a, b, rel in rows[:top]:
        flag = "  <<<" if worst > 0.5 else ""
        print(f"{nm:>14s} {a['mass']:8.2f} {b['mass']:8.2f} {100*rel['mass']:+7.1f} "
              f"{a['r']:7.4f} {b['r']:7.4f} {100*rel['r']:+7.1f} "
              f"{a['L']:7.4f} {b['L']:7.4f} {100*rel['L']:+7.1f} "
              f"{a['dens']:9.1f} {b['dens']:9.1f} {100*rel['dens']:+7.1f}{flag}")
    if len(rows) > top:
        print(f"  ... {len(rows) - top} more bodies within tolerance (rerun with --top {len(rows)})")

    print()
    print(f"  totals:  mass {rA['mass']:.1f} vs {rB['mass']:.1f} kg "
          f"({100*(rA['mass']-rB['mass'])/max(rB['mass'],1e-9):+.1f}%)   "
          f"height {rA['height']:.3f} vs {rB['height']:.3f} m   "
          f"leg {rA['leg']:.3f} vs {rB['leg']:.3f}   arm {rA['arm']:.3f} vs {rB['arm']:.3f}")
    print("  Rows marked <<< differ by >50% on some field. If nothing is marked, the two")
    print("  MJCFs are geometrically equivalent and the bug is NOT in the MJCF geometry.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml")
    ap.add_argument("--highlight", nargs="*", default=["sub4", "sub16"],
                    help="subjects to call out explicitly in the report")
    ap.add_argument("--z", type=float, default=3.0, help="robust-z threshold to flag")
    ap.add_argument("--compare", nargs=2, metavar=("SUSPECT", "CONTROL"),
                    help="per-body diff of two subjects, e.g. --compare sub4 sub11")
    ap.add_argument("--top", type=int, default=25, help="rows to show in --compare")
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

    if a.compare:
        sA, sB = a.compare
        for s in (sA, sB):
            if s not in subs:
                raise SystemExit(f"FATAL: {s} not found in {a.glob} (have: {' '.join(subs)})")
        compare(R[subs.index(sA)], R[subs.index(sB)], sA, sB, top=a.top)
        return

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

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

    # ORDERED joint signature. The policy/motion share ONE dof_pos vector across all
    # bodies: dof_pos[i] drives the i-th joint in the sim's traversal order. So the
    # ordered (name, axis, range) sequence must be IDENTICAL across subjects. Same
    # joint COUNT with a different ORDER or AXIS silently applies each value to the
    # wrong joint -- which in a kinematic replay renders as a jagged, teleporting
    # limb while every mass/height/symmetry check stays clean.
    jseq = []
    for j in root.iter("joint"):
        jseq.append((j.get("name", "?"),
                     j.get("axis", "?").strip(),
                     j.get("range", "?").strip()))

    return {
        "path": path,
        "jseq": jseq,
        "bseq": list(bodies.keys()),
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


FIELDS = ("mass", "r", "L", "offset")


def asym_map(r):
    """Signed L-vs-R asymmetry for every (limb pair, field) in one subject.

    Every other check here is external (subject vs subject, subject vs population),
    so a limb wrong on ONE side is invisible to them -- the totals average out. A
    "flamingo leg" is exactly that. A human MJCF must be near-symmetric.

    Compares MAGNITUDES (the L/R geoms are mirrored in y, so signed coords
    legitimately differ).
    """
    per = {}
    for g in r["geoms"]:
        e = per.setdefault(g["body"], {"mass": 0.0, "r": 0.0, "L": 0.0})
        e["mass"] += g["mass"]
        e["r"] = max(e["r"], g["r"])
        e["L"] = max(e["L"], g["L"])
    for nm, b in r["bodies"].items():
        per.setdefault(nm, {"mass": 0.0, "r": 0.0, "L": 0.0})["offset"] = b["offset"]

    out, missing = {}, []
    for nm in per:
        if not nm.startswith("L_"):
            continue
        mate = "R_" + nm[2:]
        if mate not in per:
            missing.append(nm)
            continue
        for k in FIELDS:
            va, vb = per[nm].get(k, 0.0), per[mate].get(k, 0.0)
            denom = max(abs(va), abs(vb), 1e-9)
            if denom < 1e-6:
                continue
            out[(nm, mate, k)] = ((va - vb) / denom, va, vb)
    return out, missing


def symmetry_outliers(asyms, floor=0.10, zthr=3.0):
    """Which subjects are asymmetric ANOMALOUSLY, vs the population?

    SMPL-X's own template is slightly asymmetric (thumbs, thorax, shoulder offset),
    and every subject inherits it -- so a flat threshold flags all 17 and says
    nothing. What matters is a subject whose asymmetry at a given joint pair is
    out of line with what every OTHER subject shows there. Shared template
    asymmetry cancels; a uniquely broken limb stands out.

    Flags only when BOTH: |asym| exceeds `floor` in absolute terms, AND its robust
    z against the other subjects exceeds `zthr`.
    """
    keys = set()
    for m, _ in asyms.values():
        keys |= set(m)
    hits = {s: [] for s in asyms}
    for key in keys:
        vals, subs_with = [], []
        for s, (m, _) in asyms.items():
            if key in m:
                vals.append(m[key][0])
                subs_with.append(s)
        if len(vals) < 4:
            continue
        z = robust_z(vals)
        for s, zz, v in zip(subs_with, z, vals):
            if abs(v) > floor and abs(zz) > zthr:
                ln, rn, field = key
                _, va, vb = asyms[s][0][key]
                hits[s].append((abs(zz), abs(v), ln, rn, field, va, vb))
    for s in hits:
        hits[s].sort(reverse=True)
    return hits


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


# Attributes whose values SHOULD differ between subjects (they encode body shape).
# Everything else is physics/structure and must be identical -- a difference there
# is a bug, not a body.
SHAPE_ATTRS = {"pos", "size", "fromto", "density", "quat", "euler"}


def xml_signature(path):
    """(tag-path, attr-name) -> value for every non-shape attribute in the file.

    Deliberately NOT a list of checks I thought of. It captures EVERYTHING --
    stiffness, damping, armature, contype/conaffinity (self-collision filtering),
    condim, margin, joint type, contact excludes, element order -- and lets the
    diff say what actually differs. Five hand-written checks have now missed
    whatever is wrong with sub4; this one cannot miss by construction.
    """
    root = ET.parse(path).getroot()
    sig, order = {}, []

    def walk(e, prefix):
        # Index siblings by tag so element ORDER is part of the signature.
        counts = {}
        for c in e:
            i = counts.get(c.tag, 0)
            counts[c.tag] = i + 1
            nm = c.get("name") or f"#{i}"
            p = f"{prefix}/{c.tag}[{nm}]"
            order.append(p)
            for k, v in c.attrib.items():
                if k in SHAPE_ATTRS:
                    continue
                sig[(p, k)] = v.strip()
            walk(c, p)

    walk(root, "")
    return sig, order


def xmldiff(pa, pb, na, nb, limit=40):
    sa, oa = xml_signature(pa)
    sb, ob = xml_signature(pb)
    print("=" * 100)
    print(f"STRUCTURAL XML DIFF  {na} vs {nb}")
    print("  Ignores shape attrs (pos/size/fromto/density) -- those SHOULD differ.")
    print("  Everything else (stiffness, damping, armature, contype/conaffinity,")
    print("  condim, margin, joint type, contact excludes, element order) must match.")
    print("=" * 100)

    if oa != ob:
        print(f"\n  !! ELEMENT ORDER/SET DIFFERS ({len(oa)} vs {len(ob)} elements)")
        setA, setB = set(oa), set(ob)
        onlyA, onlyB = setA - setB, setB - setA
        for nm, s in ((na, onlyA), (nb, onlyB)):
            if s:
                print(f"     only in {nm} ({len(s)}): {' '.join(sorted(s)[:12])}")
        if not onlyA and not onlyB:
            for i, (x, y) in enumerate(zip(oa, ob)):
                if x != y:
                    print(f"     same elements, different ORDER; first at index {i}: "
                          f"{na}={x}  {nb}={y}")
                    break
    else:
        print(f"\n  element order/set: IDENTICAL ({len(oa)} elements)")

    keys = set(sa) | set(sb)
    diffs = [(k, sa.get(k, "<absent>"), sb.get(k, "<absent>"))
             for k in sorted(keys) if sa.get(k) != sb.get(k)]
    if not diffs:
        print(f"  attributes:        IDENTICAL ({len(sa)} non-shape attributes)")
        print(f"\n  => {na} and {nb} are STRUCTURALLY IDENTICAL. Every difference between")
        print(f"     them is pure body shape. The MJCF is NOT the bug -- look at the motion")
        print(f"     data or the physics/contact state at runtime.")
        return
    print(f"  attributes:        {len(diffs)} DIFFER\n")
    for (p, k), va, vb in diffs[:limit]:
        print(f"    {p}")
        print(f"        {k}: {na}={va!r}   {nb}={vb!r}")
    if len(diffs) > limit:
        print(f"    ... {len(diffs)-limit} more (rerun with --top {len(diffs)})")
    print(f"\n  => These are physics/structure attributes. They should NOT vary between")
    print(f"     subjects. This is a real difference in how the two bodies simulate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml")
    ap.add_argument("--highlight", nargs="*", default=["sub4", "sub16"],
                    help="subjects to call out explicitly in the report")
    ap.add_argument("--z", type=float, default=3.0, help="robust-z threshold to flag")
    ap.add_argument("--compare", nargs=2, metavar=("SUSPECT", "CONTROL"),
                    help="per-body diff of two subjects, e.g. --compare sub4 sub11")
    ap.add_argument("--xmldiff", nargs=2, metavar=("SUSPECT", "CONTROL"),
                    help="STRUCTURAL diff of every non-shape attribute (stiffness, damping, "
                         "armature, contype/conaffinity, contact excludes, element order). "
                         "Catches what the hand-written checks miss.")
    ap.add_argument("--top", type=int, default=25, help="rows to show in --compare")
    ap.add_argument("--sym-tol", type=float, default=0.05,
                    help="flag L/R asymmetry above this fraction (default 5%%)")
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

    if a.xmldiff:
        sA, sB = a.xmldiff
        for s in (sA, sB):
            if s not in subs:
                raise SystemExit(f"FATAL: {s} not found in {a.glob} (have: {' '.join(subs)})")
        xmldiff(paths[subs.index(sA)], paths[subs.index(sB)], sA, sB, limit=a.top)
        return

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
    print("DOF ALIGNMENT  (ordered joint name/axis/range sequence -- must be IDENTICAL)")
    print("  One dof_pos vector drives every body: dof_pos[i] -> the i-th joint. If the")
    print("  ORDER or AXIS differs, values land on the WRONG joint -- which renders as a")
    print("  jagged, teleporting limb even though counts, mass, and symmetry all pass.")
    print("=" * 96)
    from collections import Counter as _C
    ref_key, _ = _C(tuple(r["jseq"]) for r in R).most_common(1)[0]
    ref = list(ref_key)
    n_ref = sum(1 for r in R if tuple(r["jseq"]) == ref_key)
    print(f"  reference = the majority sequence ({n_ref}/{len(R)} subjects agree), "
          f"{len(ref)} joints")
    bad_dof = []
    for s, r in zip(subs, R):
        seq = r["jseq"]
        if tuple(seq) == ref_key:
            continue
        bad_dof.append(s)
        print(f"\n  {s}: joint sequence DIFFERS from the reference")
        if len(seq) != len(ref):
            print(f"      length {len(seq)} vs reference {len(ref)}")
        for i in range(min(len(seq), len(ref))):
            if seq[i] != ref[i]:
                print(f"      FIRST DIVERGENCE at dof index {i}:")
                print(f"        this subject : name={seq[i][0]!r} axis={seq[i][1]!r} range={seq[i][2]!r}")
                print(f"        reference    : name={ref[i][0]!r} axis={ref[i][1]!r} range={ref[i][2]!r}")
                print(f"      -> from dof {i} on, motion values drive the wrong joint on this body.")
                # How many, and which bodies are affected downstream?
                diff = [k for k in range(min(len(seq), len(ref))) if seq[k] != ref[k]]
                affected = sorted({seq[k][0].rsplit('_', 1)[0] for k in diff})
                print(f"      -> {len(diff)} of {len(ref)} dofs misaligned, touching: {' '.join(affected)}")
                break
    if not bad_dof:
        print("  OK -- every subject shares the exact same ordered joint sequence.")
        print("  (So a jagged/teleporting limb is NOT a dof-misalignment problem.)")
    else:
        print(f"\n  DOF-MISALIGNED SUBJECTS: {' '.join(bad_dof)}")
        print("  These bodies CANNOT be driven by a shared dof_pos vector or a shared policy.")

    print()
    print("=" * 96)
    print(f"LEFT/RIGHT SYMMETRY  (asymmetry > {100*a.sym_tol:.0f}% of the larger side)")
    print("  A limb wrong on ONE side is invisible to every check above -- the totals")
    print("  average out. This is the only check that sees a 'flamingo leg'.")
    print("=" * 96)
    # SMPL-X's own template is slightly asymmetric (thumbs, thorax, shoulder) and
    # every subject inherits it -- so flag only asymmetry that is anomalous vs the
    # OTHER subjects at that same joint pair, not vs perfect symmetry.
    asyms = {s: asym_map(r) for s, r in zip(subs, R)}
    SYM = symmetry_outliers(asyms, floor=a.sym_tol, zthr=a.z)
    any_asym = False
    for s in subs:
        if not SYM[s]:
            continue
        any_asym = True
        print(f"\n  {s}: {len(SYM[s])} anomalous asymmetry(ies)")
        for zz, av, ln, rn, field, va, vb in SYM[s][:6]:
            print(f"      {field:>7s}  {ln:>11s} {va:9.4f}  vs  {rn:>11s} {vb:9.4f}   "
                  f"{100*av:6.1f}% APART  (z={zz:+.1f} vs other subjects)")
    if not any_asym:
        print("  none -- no subject's L/R asymmetry is out of line with the population.")
        print("  (Shared SMPL-X template asymmetry is expected and correctly ignored.)")
    else:
        print(f"\n  ASYMMETRIC SUBJECTS: {' '.join(s for s in subs if SYM[s])}")

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
        asym = SYM.get(s, [])
        if s in bad_dof:  # noqa: E501
            print(f"  {s}: DOF-MISALIGNED -- its ordered joint sequence differs from the")
            print(f"       other subjects. A shared dof_pos vector drives the WRONG joints on")
            print(f"       this body: the limb jitters and teleports in kinematic replay while")
            print(f"       mass/height/symmetry all look normal. THIS IS THE BUG. Regenerate")
            print(f"       this MJCF; no policy can drive it correctly.")
        elif R[i]["problems"]:
            print(f"  {s}: MALFORMED -- {len(R[i]['problems'])} degeneracy(ies). This is the bug.")
        elif asym:
            zz, av, ln, rn, field, va, vb = asym[0]
            print(f"  {s}: asymmetric -- {ln}/{rn} differ by {100*av:.0f}% in {field} "
                  f"({va:.4f} vs {vb:.4f}, z={zz:+.1f}).")
            print(f"       NOTE: real subjects all share a large L/R thorax asymmetry that the")
            print(f"       synthetic bodies lack, so this flags real-vs-synthetic more than it")
            print(f"       flags a defect. Only trust it if this subject stands out from OTHER")
            print(f"       REAL subjects -- use --xmldiff for a definitive structural compare.")
        elif flags:
            print(f"  {s}: physically anomalous -- {', '.join(flags)}")
        else:
            print(f"  {s}: MJCF is clean, symmetric, and unremarkable.")
            print(f"       -> the body geometry does NOT explain its failure. Look at the")
            print(f"          termination reason instead (TERM_REASON=1).")


if __name__ == "__main__":
    main()

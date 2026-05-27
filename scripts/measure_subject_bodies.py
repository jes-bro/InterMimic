#!/usr/bin/env python3
"""Measure body proportions from generated per-subject MJCFs.

Walks each smplx_omomo_sub*.xml in the assets dir and computes a few
pose-invariant body proportions, then prints a sorted table. Useful for
sanity-checking that a proposed train/test split actually covers body-size
variation, not just |β| variation.

Usage:
    python scripts/measure_subject_bodies.py
"""

import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path


def get_pos(body):
    """Local-frame pos as (x, y, z), defaulting to zero if absent."""
    p = body.get("pos", "0 0 0").split()
    return tuple(float(v) for v in p)


def find_body(root, name):
    """Find a <body> with given name anywhere in the subtree."""
    return next((b for b in root.iter("body") if b.get("name") == name), None)


def chain_length(root, names):
    """Sum the Euclidean magnitudes of pos vectors along a named body chain.

    `names` lists bodies in order (each is a child of the previous in the MJCF
    tree). Returns total link-length along the chain.
    """
    total = 0.0
    for name in names:
        b = find_body(root, name)
        if b is None:
            return None
        x, y, z = get_pos(b)
        total += (x * x + y * y + z * z) ** 0.5
    return total


def chain_z_drop(root, names):
    """Sum signed z-offsets along a chain (negative for downward chains)."""
    z = 0.0
    for name in names:
        b = find_body(root, name)
        if b is None:
            return None
        z += get_pos(b)[2]
    return z


def total_height(root):
    """Approximate body height: head Z drop above pelvis + foot Z drop below."""
    head_up = chain_z_drop(root, ["Torso", "Spine", "Chest", "Neck", "Head"])
    foot_down = chain_z_drop(root, ["L_Hip", "L_Knee", "L_Ankle"])
    if head_up is None or foot_down is None:
        return None
    return head_up - foot_down  # foot_down is negative


def measure(xml_path):
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    pelvis = next((b for b in worldbody.findall("body") if b.get("name") == "Pelvis"), None)
    if pelvis is None:
        return None
    return {
        "height": total_height(pelvis),
        "leg": chain_length(pelvis, ["L_Hip", "L_Knee", "L_Ankle"]),
        "arm": chain_length(pelvis, ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist"]),
        "torso": chain_length(pelvis, ["Torso", "Spine", "Chest"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--assets-dir",
        default="isaacgym/src/intermimic/data/assets/smplx",
        help="Directory containing smplx_omomo_sub*.xml files",
    )
    p.add_argument("--pattern", default="smplx_omomo_sub*.xml")
    p.add_argument("--sort-by", default="height",
                   choices=["height", "leg", "arm", "torso", "name"])
    args = p.parse_args()

    files = sorted(glob.glob(str(Path(args.assets_dir) / args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.assets_dir}")

    rows = []
    for f in files:
        name = Path(f).stem.replace("smplx_omomo_", "")
        m = measure(f)
        if m is None:
            print(f"  skipped {name} (parse failed)")
            continue
        rows.append((name, m))

    if args.sort_by == "name":
        rows.sort(key=lambda r: r[0])
    else:
        rows.sort(key=lambda r: r[1][args.sort_by])

    print(f"{'subject':10s} {'height(m)':>10s} {'leg(m)':>8s} {'arm(m)':>8s} {'torso(m)':>9s}")
    print("-" * 50)
    for name, m in rows:
        print(f"{name:10s} {m['height']:>10.4f} {m['leg']:>8.4f} "
              f"{m['arm']:>8.4f} {m['torso']:>9.4f}")


if __name__ == "__main__":
    main()

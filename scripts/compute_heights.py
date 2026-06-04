#!/usr/bin/env python3
"""Estimate height of each subject by walking the MJCF skeleton from
Pelvis -> Spine -> Chest -> Neck -> Head and summing absolute z offsets.

Adds the head sphere radius at the end as a rough approximation of
total standing height.

Run from repo root:
    python scripts/compute_heights.py
"""
import re
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "isaacgym/src/intermimic/data/assets/smplx"

SPINE_CHAIN = ["Pelvis", "Spine", "Chest", "Neck", "Head"]

# Match a <body name="X" pos="x y z"> tag.
BODY_RE = re.compile(r'<body\s+name="([^"]+)"\s+pos="([^"]+)"')
# Match a head <geom type="sphere" size="r" .../>
HEAD_GEOM_RE = re.compile(r'<geom[^/]*type="sphere"[^/]*size="([0-9.]+)"[^/]*name="Head"')


def estimate_height(xml_path):
    text = Path(xml_path).read_text()
    bodies = {}
    for m in BODY_RE.finditer(text):
        name, pos = m.group(1), m.group(2)
        x, y, z = (float(v) for v in pos.split())
        bodies[name] = (x, y, z)
    # The chain accumulates z offsets along the spine.
    total_z = 0.0
    for name in SPINE_CHAIN[1:]:
        if name in bodies:
            total_z += abs(bodies[name][2])
    # Head geom sphere radius (best-guess search)
    head_size = 0.10  # default
    head_geom_match = re.search(
        r'<body\s+name="Head"[^>]*>\s*<geom[^/]*size="([0-9.]+)"',
        text,
    )
    if head_geom_match:
        head_size = float(head_geom_match.group(1))
    # Plus pelvis sphere radius below (estimate half-pelvis-to-floor)
    pelvis_geom_match = re.search(
        r'<body\s+name="Pelvis"[^>]*>\s*<geom[^/]*size="([0-9.]+)"',
        text,
    )
    pelvis_size = float(pelvis_geom_match.group(1)) if pelvis_geom_match else 0.10
    # And the L_Hip -> L_Knee -> L_Ankle -> L_Toe leg chain (absolute z).
    leg_z = 0.0
    for j in ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]:
        if j in bodies:
            leg_z += abs(bodies[j][2])
    return total_z + head_size + pelvis_size + leg_z


if __name__ == "__main__":
    subjects = ["sub1", "sub2", "sub3", "sub5", "sub9", "sub10", "sub17"]
    print(f"{'subject':<8s}  height_est (m)")
    for s in subjects:
        path = ASSETS / f"smplx_omomo_{s}.xml"
        if path.exists():
            h = estimate_height(path)
            print(f"  {s:<8s}  {h:.3f}")
        else:
            print(f"  {s:<8s}  (missing MJCF at {path})")

#!/usr/bin/env python3
"""Compare the inertia my formulas produce (scripts/mjcf_add_inertials.py) with
what PhysX reported for the same bodies (scripts/probe_asset_in_gym.py output).

This is the exact computation behind the "matches PhysX to the printed
precision" claim (2026-09-29). The PhysX numbers below were transcribed BY HAND
from the probe's printed table (5 decimals, a subset of bodies). For a
full-precision, all-bodies comparison run the probe with --compare-formulas
instead; this script only documents the hand check.

  python3 scripts/compare_inertia_to_physx.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mjcf_add_inertials as mi  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "isaacgym/src/intermimic/data/assets/smplx"

# From the probe run Jess pasted (ENGINE VIEW, 2026-09-28): body -> (mass, Ixx, Iyy, Izz)
# in the BODY frame, as printed (5 decimals).
PHYSX = {
    "sub4": {"Pelvis": (4.6982, 0.02029, 0.02029, 0.02029), "L_Hip": (6.4510, 0.05442, 0.05321, 0.01150),
             "L_Knee": (3.7102, 0.03397, 0.03376, 0.00604), "L_Ankle": (0.9577, 0.00112, 0.00337, 0.00419),
             "Torso": (2.8412, 0.00792, 0.00792, 0.00714), "Head": (5.9482, 0.03006, 0.03006, 0.03006),
             "L_Elbow": (1.3442, 0.00637, 0.00127, 0.00632)},
    "sub2": {"L_Knee": (4.0825, 0.04123, 0.04111, 0.00696), "L_Hip": (7.6529, 0.07476, 0.07313, 0.01506)},
}


def body_tensor(body):
    """mass, com and inertia tensor about the com, in the BODY frame, summed over
    the body's geoms with the parallel-axis shift -- same math mjcf_add_inertials uses."""
    parts = [mi.geom_mass_com_inertia(g) for g in body.findall("geom")]
    M = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / M
    I = np.zeros((3, 3))
    for m, c, Ig in parts:
        d = c - com
        I += Ig + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, I


def main():
    for s, bodies in PHYSX.items():
        root = ET.parse(ASSETS / f"smplx_omomo_{s}.xml").getroot()
        print(f"{s}: body      mass mine/PhysX      body-frame diagonal mine vs PhysX (Ixx Iyy Izz)      max rel diff")
        for b in root.iter("body"):
            n = b.get("name")
            if n not in bodies:
                continue
            M, com, I = body_tensor(b)
            pm, px, py, pz = bodies[n]
            mine = np.diag(I)
            rel = max(abs(mine[i] - p) / p for i, p in enumerate((px, py, pz)))
            print(f"   {n:8s} {M:7.4f}/{pm:7.4f}   {mine[0]:.5f} {mine[1]:.5f} {mine[2]:.5f}  vs  "
                  f"{px:.5f} {py:.5f} {pz:.5f}   {100 * rel:4.1f}%")
            print(f"            full precision, mine: mass {M:.6f}  diag {mine[0]:.7f} {mine[1]:.7f} {mine[2]:.7f}  "
                  f"offdiag {I[0,1]:+.7f} {I[0,2]:+.7f} {I[1,2]:+.7f}  com {com[0]:+.4f} {com[1]:+.4f} {com[2]:+.4f}")


if __name__ == "__main__":
    main()

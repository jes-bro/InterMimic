#!/usr/bin/env python3
"""Emit a basketball hoop URDF from primitives (no meshes, no VHACD): a rim
ring approximated by tangential cylinder segments, a backboard box behind it
along +x, and a support pole down to the floor.

The asset ORIGIN is the rim ring's center, so the actor pose in staticScene is
simply the measured rattle-segment centroid, with a z-yaw turning +x (the
backboard side) to face along the shot's approach direction.

  python3 scripts/generate_hoop_urdf.py \
      --out isaacgym/src/intermimic/data/assets/objects/hoop_bball.urdf \
      --rim-height 3.034
"""
import argparse
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rim-radius", type=float, default=0.229, help="regulation ring radius (m)")
    ap.add_argument("--tube-radius", type=float, default=0.021)
    ap.add_argument("--segments", type=int, default=12)
    ap.add_argument("--board-offset", type=float, default=0.45,
                    help="ring center to backboard face along +x")
    ap.add_argument("--rim-height", type=float, default=3.034,
                    help="pole extends this far DOWN from the ring to reach the floor")
    args = ap.parse_args()

    seg_len = 2 * math.pi * args.rim_radius / args.segments * 1.05  # slight overlap
    geoms = []
    for k in range(args.segments):
        th = 2 * math.pi * k / args.segments
        x, y = args.rim_radius * math.cos(th), args.rim_radius * math.sin(th)
        # cylinder axis z -> tangent: Rx(90deg) then Rz(th)  (URDF rpy = fixed XYZ)
        geoms.append((f"{x:.4f} {y:.4f} 0", f"{math.pi/2:.6f} 0 {th:.6f}",
                      f'<cylinder radius="{args.tube_radius}" length="{seg_len:.4f}"/>'))
    # backboard: 1.1 x 0.8 plate, bottom edge near ring level
    geoms.append((f"{args.board_offset:.3f} 0 0.35", "0 0 0",
                  '<box size="0.04 1.10 0.80"/>'))
    # pole: from behind the board down to the floor
    geoms.append((f"{args.board_offset + 0.06:.3f} 0 {-args.rim_height/2:.4f}", "0 0 0",
                  f'<box size="0.10 0.10 {args.rim_height:.4f}"/>'))

    parts = []
    for pos, rpy, geo in geoms:
        for tag in ("visual", "collision"):
            parts.append(f'    <{tag}>\n      <origin rpy="{rpy}" xyz="{pos}"/>\n'
                         f'      <geometry>\n        {geo}\n      </geometry>\n    </{tag}>')
    urdf = ('<?xml version="1.0" ?>\n<robot name="hoop_bball">\n  <link name="hoop">\n'
            + "\n".join(parts) + "\n  </link>\n</robot>\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(urdf)
    print(f"wrote {args.out}: {args.segments}-segment rim r={args.rim_radius}, "
          f"backboard at +x {args.board_offset}, pole drop {args.rim_height}")


if __name__ == "__main__":
    main()

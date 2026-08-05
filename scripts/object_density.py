#!/usr/bin/env python3
"""Compute the objectDensity that gives a mesh its real-world mass.

Isaac Gym derives an object's mass from its mesh volume times objectDensity, so
the config value is not a property of the object -- it depends on how big the
reconstructed mesh came out. Copying a density between objects therefore gets
the mass wrong even when both numbers look reasonable: the OMOMO furniture value
of 200 gives a basketball about 1.4 kg against a real 0.62.

    python scripts/object_density.py --mesh <path to .obj> --mass 0.624

Mass is the thing worth specifying because it is the thing you can look up. A
size-7 basketball is 0.567-0.650 kg by rule; a gas canister is whatever the
label says.

Bouncing needs restitution as well, which lives in the env YAML and the object
URDF rather than here -- density alone changes how hard it hits, not how much it
comes back.
"""

import argparse
import sys
from pathlib import Path


# Reference masses in kg, for objects likely to come out of this pipeline.
# Sources are the governing bodies' equipment rules where one exists.
KNOWN_MASSES = {
    "basketball-7": 0.624,     # FIBA size 7, men's: 0.567-0.650
    "basketball-6": 0.538,     # size 6, women's: 0.510-0.567
    "basketball-5": 0.481,     # size 5, youth: 0.470-0.500
    "soccer-5": 0.430,         # FIFA size 5: 0.410-0.450
    "volleyball": 0.270,
    "football-nfl": 0.425,
}


def mesh_volume(path: Path) -> float:
    """Return a mesh's enclosed volume in cubic metres.

    Raises:
        SystemExit: if the mesh is missing, empty, or not watertight. A
            non-watertight mesh has no well-defined volume, and trimesh will
            happily return a plausible-looking negative or near-zero number
            rather than failing, which would silently produce a wrong mass.
    """
    try:
        import trimesh
    except ImportError:
        raise SystemExit("trimesh is required; it ships in the intermimic env")

    if not path.is_file():
        raise SystemExit(f"no mesh at {path}")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if mesh.is_empty:
        raise SystemExit(f"{path.name} has no geometry")

    if not mesh.is_watertight:
        filled = mesh.copy()
        filled.fill_holes()
        if not filled.is_watertight:
            raise SystemExit(
                f"{path.name} is not watertight even after hole filling, so its "
                f"volume is undefined. Isaac Gym will still load it, but any "
                f"density derived from it would be meaningless.")
        print(f"note: {path.name} needed hole filling to close; volume is "
              f"approximate")
        mesh = filled

    volume = float(mesh.volume)
    if volume <= 0:
        raise SystemExit(f"{path.name} has non-positive volume ({volume:.6g}); "
                         f"its faces are probably inward-facing")
    extents = mesh.extents
    print(f"{path.name}: volume {volume:.6f} m^3, "
          f"extents {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m")
    return volume


def main() -> int:
    """Print the density, and the mass the current config would give instead."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, required=True,
                        help="the .obj Isaac Gym loads, i.e. the metric-scale one")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mass", type=float, help="target mass in kg")
    group.add_argument("--known", choices=sorted(KNOWN_MASSES),
                       help="use a standard mass instead of stating one")
    parser.add_argument("--current-density", type=float, default=200.0,
                        help="density in the config today, for comparison "
                             "(default: 200, the OMOMO value)")
    args = parser.parse_args()

    mass = args.mass if args.mass is not None else KNOWN_MASSES[args.known]
    if args.known:
        print(f"using {args.known} = {mass} kg")
    volume = mesh_volume(args.mesh.expanduser().resolve())

    density = mass / volume
    current_mass = args.current_density * volume
    print()
    print(f"target mass      {mass:.3f} kg")
    print(f"current density  {args.current_density:.0f} -> {current_mass:.3f} kg "
          f"({current_mass / mass:.1f}x too heavy)" if current_mass > mass else
          f"current density  {args.current_density:.0f} -> {current_mass:.3f} kg "
          f"({mass / current_mass:.1f}x too light)")
    print()
    print(f"  objectDensity: {density:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

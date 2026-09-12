#!/usr/bin/env python3
"""Per-object density from a target mass.

Isaac Gym derives an object's mass from mesh volume x density, and the env cfg
carries ONE `objectDensity` for every object in the run. That is fine for a
dataset with one mesh per object type. It is wrong for a dataset where every
clip brings its own reconstructed mesh of the SAME real object: the CARI4D
basketball meshes span 0.21-0.26 m, so one density scatters their masses by
about +-35% around the 0.62 kg every one of them actually weighs.

`objectMass: <kg>` in the env cfg replaces `objectDensity` with the thing you
can look up, and this computes the density each mesh needs to hit it.

WHICH VOLUME. PhysX does not see the raw mesh; it sees the VHACD convex
decomposition the asset loader builds (intermimic.py _load_target_asset), and
for a near-convex ball that is the mesh's convex hull to within ~2%. The raw
Hunyuan3D meshes are triangle SOUPS (every face its own component, no shared
vertices, 100k+ boundary edges), so the divergence-theorem `mesh.volume` lands
anywhere from 68% to 99% of the true volume depending on the soup -- measured
2026-09-12 on three real balls. The hull volume is what the mass will actually
be computed from, so that is the volume used here. The task then reads the
mass PhysX assigned back from the actor and refuses to start if it is off.
"""
import trimesh


def hull_volume(obj_path):
    """Convex-hull volume in m^3, plus the raw (unreliable on soups) volume for
    the log line."""
    mesh = trimesh.load(str(obj_path), force="mesh")
    hull = mesh.convex_hull
    vol = float(hull.volume)
    if vol <= 0:
        raise ValueError(f"{obj_path}: non-positive hull volume {vol}")
    try:
        raw = abs(float(mesh.volume))
    except Exception:  # noqa: BLE001 -- purely informational
        raw = float("nan")
    return vol, raw


def density_for_mass(obj_path, mass_kg):
    """Return (density kg/m^3, hull volume m^3, raw mesh volume m^3)."""
    vol, raw = hull_volume(obj_path)
    return mass_kg / vol, vol, raw

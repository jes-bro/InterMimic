#!/usr/bin/env python3
"""Re-export a mesh through trimesh to produce an Isaac-Gym-friendly .obj.

Hunyuan3D meshes sometimes confuse Isaac Gym's mesh loader: non-triangle
faces, dangling mtllib references, vertex normals/texcoords that don't
match face indices, and so on. This script loads the mesh with trimesh
(which is forgiving), forces triangulation, drops materials/normals/texcoords,
and writes vertices + faces only.

Usage:
    python scripts/clean_mesh_for_isaacgym.py path/to/gas.obj
    # overwrites the file in place after backing up to path/to/gas.obj.bak
"""

import argparse
import shutil
import sys
from pathlib import Path

import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj_path", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Default: overwrite input (backup at <input>.bak).")
    args = parser.parse_args()

    src = args.obj_path.expanduser().resolve()
    if not src.is_file():
        print(f"not a file: {src}", file=sys.stderr)
        return 2

    dst = args.out.expanduser().resolve() if args.out else src

    mesh = trimesh.load(str(src), force="mesh", process=False)
    print(f"loaded {src.name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces "
          f"(min/max vertex per face = {mesh.faces.shape[1]})")

    # trimesh with force='mesh' should already triangulate, but be explicit
    if hasattr(mesh, "faces") and mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate() if hasattr(mesh, "triangulate") else mesh

    if dst == src:
        backup = src.with_suffix(src.suffix + ".bak")
        if not backup.exists():
            shutil.copy(str(src), str(backup))
            print(f"backed up original to {backup.name}")

    # Plain export — vertices + faces only, no materials/normals/texcoords
    with dst.open("w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in mesh.faces:
            # OBJ face indices are 1-based
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"wrote {dst}: {len(mesh.vertices)} verts, {len(mesh.faces)} tri faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Finalize a behave_cari4d build for InterMimic replay.

interact2mimic.py (invoked via scripts/run_interact2mimic.py, which forces
mesh=True so the MJCF references per-bone subject-shape convex-hull STLs
rather than generic capsules) writes outputs under cwd as:

    intermimic/InterAct/<dataset_tag>/<seq>.pt              (motion tensor)
    intermimic/data/assets/smplx/smplh_behave_<sub>.xml     (per-subject MJCF)
    intermimic/data/assets/objects/<branch>/<obj>.urdf      (BROKEN: 'hoi/...' mesh ref)
    /tmp/smpl/<uuid>/geom/*.stl                             (per-bone subject hulls)

InterMimic expects them at:

    <intermimic_root>/InterAct/<dataset_tag>/<seq>.pt
    <intermimic_root>/isaacgym/src/intermimic/data/assets/smplx/smplh_behave_<sub>.xml
    <intermimic_root>/isaacgym/src/intermimic/data/assets/objects/<obj>.urdf
    <intermimic_root>/isaacgym/src/intermimic/data/assets/objects/objects/<obj>/<obj>.obj

This script:
  1. Moves the .pt + MJCF files into place.
  2. Copies the source mesh from InterAct into InterMimic's nested object layout.
  3. Writes a working URDF (mirrors the existing OMOMO URDFs — mesh ref is
     'objects/<name>/<name>.obj' relative to assets/objects/).

Run on the cluster (the machine that ran interact2mimic.py), with cwd at the
directory you ran interact2mimic.py from (so 'intermimic/...' resolves).

Usage:
    python scripts/cari4d_finalize.py \\
        --interact-root /path/to/InterAct \\
        --intermimic-root /path/to/InterMimic \\
        --dataset-tag behave_cari4d
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


URDF_TEMPLATE = """<?xml version="1.0" ?>
<robot name="{object_name}.urdf">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="objects/{object_name}/{object_name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="mat">
        <color rgba="0.7 0.8 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="objects/{object_name}/{object_name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _safe_move(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    print(f"  moved  {src} -> {dst}")


def _safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src), str(dst))
    print(f"  copied {src} -> {dst}")


def _extract_stl_uuids(mjcf_path):
    """Read a PHC-generated MJCF and return the unique UUID directories its
    `<mesh file="<uuid>/geom/...">` entries reference."""
    tree = ET.parse(str(mjcf_path))
    uuids = []
    seen = set()
    for mesh in tree.getroot().findall(".//asset/mesh"):
        fname = mesh.attrib.get("file", "")
        if "/geom/" not in fname:
            continue
        uuid = fname.split("/geom/", 1)[0]
        if uuid and uuid not in seen:
            seen.add(uuid)
            uuids.append(uuid)
    return uuids


def _relocate_stl_dir(uuid: str, dst_dir: Path) -> None:
    """Move /tmp/smpl/<uuid>/ to <dst_dir>/ so MJCF's relative
    '<uuid>/geom/<bone>.stl' resolves correctly."""
    src_dir = Path("/tmp/smpl") / uuid
    if not src_dir.is_dir():
        print(f"  WARNING: missing STL source {src_dir} — MJCF references "
              f"'{uuid}/geom/...' but PHC didn't write it. mesh=True may have failed.",
              file=sys.stderr)
        return
    if dst_dir.exists():
        shutil.rmtree(str(dst_dir))
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir), str(dst_dir))
    print(f"  moved  {src_dir} -> {dst_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interact-root", type=Path, required=True,
                        help="InterAct clone (contains data/<dataset-tag>/objects/...).")
    parser.add_argument("--intermimic-root", type=Path, required=True,
                        help="InterMimic clone (target layout for replay).")
    parser.add_argument("--dataset-tag", default="behave_cari4d",
                        help="Dataset tag used with interact2mimic.py (default: behave_cari4d).")
    parser.add_argument("--simulation-cwd", type=Path, default=None,
                        help="Directory you ran interact2mimic.py from (defaults to "
                             "<interact-root>/simulation).")
    args = parser.parse_args()

    interact_root = args.interact_root.expanduser().resolve()
    intermimic_root = args.intermimic_root.expanduser().resolve()
    sim_cwd = (args.simulation_cwd or (interact_root / "simulation")).expanduser().resolve()

    if not (intermimic_root / "isaacgym" / "src" / "intermimic").is_dir():
        print(f"--intermimic-root does not look like InterMimic: {intermimic_root}",
              file=sys.stderr)
        return 2

    sim_outputs = sim_cwd / "intermimic"
    pt_src_dir = sim_outputs / "InterAct" / args.dataset_tag
    mjcf_src_dir = sim_outputs / "data" / "assets" / "smplx"
    interact_objs_dir = interact_root / "data" / args.dataset_tag / "objects"

    pt_dst_dir = intermimic_root / "InterAct" / args.dataset_tag
    asset_root = intermimic_root / "isaacgym" / "src" / "intermimic" / "data" / "assets"
    mjcf_dst_dir = asset_root / "smplx"
    urdf_dst_dir = asset_root / "objects"
    obj_dst_dir = asset_root / "objects" / "objects"

    if not pt_src_dir.is_dir():
        print(f"no .pt outputs at {pt_src_dir}; did interact2mimic.py run yet?",
              file=sys.stderr)
        return 2

    pt_files = sorted(pt_src_dir.glob("*.pt"))
    if not pt_files:
        print(f"no .pt files under {pt_src_dir}", file=sys.stderr)
        return 2

    print(f"[cari4d-finalize] dataset_tag={args.dataset_tag}")
    print(f"[cari4d-finalize] {len(pt_files)} motion file(s) to install\n")

    print("motion tensors:")
    for pt in pt_files:
        _safe_move(pt, pt_dst_dir / pt.name)

    subjects = sorted({pt.stem.split("_")[0] for pt in pt_files})
    print("\nper-subject MJCFs + STL hulls:")
    for sub in subjects:
        candidates = sorted(mjcf_src_dir.glob(f"smplh_*_{sub}.xml"))
        if not candidates:
            print(f"  WARNING: no MJCF found for subject '{sub}' in {mjcf_src_dir}",
                  file=sys.stderr)
            continue
        for src in candidates:
            dst_mjcf = mjcf_dst_dir / src.name
            uuids = _extract_stl_uuids(src)
            _safe_move(src, dst_mjcf)
            for uuid_dir in uuids:
                _relocate_stl_dir(uuid_dir, dst_mjcf.parent / uuid_dir)

    objects = sorted({pt.stem.split("_")[-2] for pt in pt_files})
    print("\nobject URDFs + meshes:")
    for obj in objects:
        mesh_src = interact_objs_dir / obj / f"{obj}.obj"
        if not mesh_src.is_file():
            print(f"  WARNING: mesh missing at {mesh_src}; skipping {obj}",
                  file=sys.stderr)
            continue
        _safe_copy(mesh_src, obj_dst_dir / obj / f"{obj}.obj")
        urdf_path = urdf_dst_dir / f"{obj}.urdf"
        urdf_path.parent.mkdir(parents=True, exist_ok=True)
        urdf_path.write_text(URDF_TEMPLATE.format(object_name=obj))
        print(f"  wrote  {urdf_path}")

    print("\n[cari4d-finalize] done. Replay with:")
    print(f"  cd {intermimic_root}")
    print(f"  sh isaacgym/scripts/data_replay_cari4d.sh")
    print(f"\nGenerated MJCFs (use as robotType in env YAML):")
    for sub in subjects:
        for f in sorted(mjcf_dst_dir.glob(f"smplh_*_{sub}.xml")):
            print(f"  smplx/{f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

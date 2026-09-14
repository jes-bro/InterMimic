#!/usr/bin/env python3
"""Turn a CARI4D static-prop sidecar into a staticScene entry: mesh, URDF, pose.

CARI4D's prep/static_prop.py reconstructs a thing that does not move -- the
chair under a guitarist, a CPR manikin, a piano bench -- as one mesh and ONE
pose, held for the whole clip, in the same camera frame as the bundle's
per-frame object pose. The simulator already knows how to hold fixed geometry:
cfg['env']['staticScene'] (built for the basketball hoop) creates a
fix_base_link actor per env, after the humanoid and the object, outside every
observation and reward slice. This script is the adapter between the two.

    python scripts/cari4d_prop_to_scene.py \\
        --sidecar work/<seq>/props/chair/<seq>_chair.json \\
        --bundle  output/opt/<...>/<seq>.pth \\
        --pt      InterAct/behave_cari4d_guitar/sub400_guitar_000.pt \\
        --rotate-axis x --rotate-degrees 180        # what step 3.5 did to the .pt

THE FRAME PROBLEM, AND WHY THIS DOES NOT GUESS. The hoop was placed by hand
and came out in the wrong orientation once, because "the camera frame turned
into the simulator frame" is several stages (SMPL y-up -> z-up, retargeting,
floor seating, the gravity flip of rotate_pt.py, drop-to-floor) and it is easy
to apply one of them twice or not at all. Here the rotation is COMPOSED from
what those stages state they did -- interact2mimic.py's fixed +90 deg about X,
then rotate_pt.py's rotation (--rotate-axis/--rotate-degrees, or --from-calib,
whichever the conversion used) -- and the translation is FITTED from the data:
the tracked object's per-frame position is in both the bundle (pose_abs, camera
frame) and the installed tensor (channels 318:321, simulator frame), so the
offset between the two frames is measured, not derived. Every floor shift and
retarget offset lands in that measured translation. The residual of the fit is
printed; a few centimetres is the same motion in two frames, tens of
centimetres means the .pt is not this bundle's clip or was rotated differently.
As a second check the rotation is also fitted (Kabsch) from the same points and
compared with the composed one -- meaningful only when the object moved enough
to pin a rotation down, which the script measures and says.

MESH CONVENTION. The sidecar's pose applies to the mesh's vertices exactly as
stored (FoundationPose's register output); interact2mimic.py applies the
tracked object's pose to its mesh exactly as stored too, so the two are
consistent and the mesh is written UNcentred, vertices and faces only, the way
cari4d_finalize.py cleans the tracked object for Isaac Gym's loader.

FLOOR CHECK. With the pose applied, the prop's lowest vertex is reported
against z = 0. A chair or bench should sit within a few centimetres of the
floor; a manikin lies on it. --snap-to-floor shifts the prop vertically to
touch it, which corrects a small depth error in the registration but would
hide a large one, so it is off by default and the shift is printed.

Writes  <assets>/objects/objects/<name>/<name>.obj, <assets>/objects/<name>.urdf
and <assets>/objects/<name>.scene.json (pose, checks, provenance), and prints
the YAML block to paste under env: in the task cfg. `vhacd: true` on the entry
is read by intermimic.py's _load_static_scene_assets and gives the prop a
convex decomposition instead of one convex hull -- a chair's seat and legs, not
a blob you cannot sit on.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

HERE = Path(__file__).resolve().parent


def _load_sibling(name: str):
    """Import a sibling script by file name, without the scripts/ dir on sys.path."""
    spec = importlib.util.spec_from_file_location(f"_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bundle_object(bundle_path: Path, key: str = "pr"):
    """(T,3) object translations and (T,3) smpl_t from a CARI4D bundle, camera frame."""
    mod = _load_sibling("cari4d_to_interact")
    if not bundle_path.is_file():
        raise SystemExit(f"no bundle at {bundle_path}")
    bundle = mod._load_bundle(bundle_path)
    if key not in bundle:
        raise SystemExit(f"bundle has no '{key}'; got {list(bundle)}")
    src = bundle[key]
    pose_abs = np.asarray(src["pose_abs"].detach().cpu().numpy(), dtype=np.float64)
    smpl_t = np.asarray(src["smpl_t"].detach().cpu().numpy(), dtype=np.float64)
    return pose_abs[:, :3, 3], smpl_t


def load_pt(pt_path: Path):
    """(T,3) obj_pos, (T,3) root_pos and (T,52,3) body_pos from a 591-channel tensor."""
    if not pt_path.is_file():
        raise SystemExit(f"no motion tensor at {pt_path}")
    data = torch.load(str(pt_path), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{pt_path.name}: {data.shape[-1]} channels, want 591")
    d = data.double().numpy()
    return d[:, 318:321], d[:, 0:3], d[:, 162:318].reshape(len(d), -1, 3)


def composed_rotation(rotate_axis, rotate_degrees, from_calib):
    """Camera frame -> simulator frame rotation, composed from what the stages did.

    interact2mimic.py:559 applies a fixed +90 deg about X to every position and
    rotation (SMPL y-up to simulator z-up). rotate_pt.py then applies either an
    axis/angle (the driver's default is x/180) or the calibration's camera-to-
    world rotation. Both are stated in the conversion job's log; pass the same.
    """
    r_x90 = sRot.from_euler("x", np.pi / 2)
    if from_calib:
        r_after = _load_sibling("rotate_pt").rotation_from_calibration(from_calib)
    elif rotate_axis:
        r_after = sRot.from_euler(rotate_axis, float(rotate_degrees), degrees=True)
    else:
        r_after = sRot.identity()
    return (r_after * r_x90).as_matrix()


def fit_translation(R, src, dst):
    """Translation t with dst ~ R @ src + t, and the RMS residual in metres."""
    t = (dst - src @ R.T).mean(axis=0)
    rms = float(np.sqrt((((src @ R.T + t) - dst) ** 2).sum(axis=1).mean()))
    return t, rms


def spread_m(points):
    """Smallest principal extent of a point cloud: how well it pins a rotation."""
    c = points - points.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False) / np.sqrt(max(len(points), 1))
    return float(s[-1]) if len(s) else 0.0


def rotation_gap_deg(R1, R2):
    """Angle between two rotations, in degrees."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def clean_mesh(src: Path, dst: Path):
    """Write vertices and triangles only, uncentred, as Isaac Gym's loader wants."""
    import trimesh
    mesh = trimesh.load(str(src), force="mesh", process=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return np.asarray(mesh.vertices, dtype=np.float64), len(mesh.faces)


def main() -> int:
    """Read the sidecar, fit the frame, write the asset, print the YAML."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sidecar", type=Path, required=True, help="<seq>_<prop>.json from CARI4D")
    parser.add_argument("--bundle", type=Path, required=True, help="the CARI4D .pth the clip was converted from")
    parser.add_argument("--pt", type=Path, required=True, help="the installed motion tensor, after rotate_pt")
    parser.add_argument("--bundle-key", default="pr", choices=["pr", "gt", "in"])
    parser.add_argument("--name", default=None,
                        help="asset name (default <prop>_<seq>, lowercased); becomes objects/<name>.urdf")
    parser.add_argument("--rotate-axis", choices=["x", "y", "z"], default="x",
                        help="rotate_pt.py's --axis used on this clip (driver default x)")
    parser.add_argument("--rotate-degrees", type=float, default=180.0)
    parser.add_argument("--from-calib", default=None, metavar="CSV:CAM_UID",
                        help="rotate_pt.py's --from-calib, if that is what the clip used instead")
    parser.add_argument("--no-rotate", action="store_true", help="the .pt was never rotated by rotate_pt")
    parser.add_argument("--assets", type=Path, default=None,
                        help="isaacgym/src/intermimic/data/assets (default: relative to this script)")
    parser.add_argument("--snap-to-floor", action="store_true",
                        help="shift the prop vertically so its lowest vertex touches z=0")
    parser.add_argument("--no-vhacd", action="store_true", help="omit vhacd: true from the entry")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    args = parser.parse_args()

    with args.sidecar.expanduser().open() as f:
        side = json.load(f)
    pose_cam = np.asarray(side["pose_cam"], dtype=np.float64)
    mesh_src = Path(side["mesh"])
    if not mesh_src.is_file():
        raise SystemExit(f"sidecar names a mesh that is not here: {mesh_src}")
    name = (args.name or f"{side['prop']}_{side['seq']}").lower()
    assets = args.assets or (HERE.parent / "isaacgym" / "src" / "intermimic" / "data" / "assets")

    # ---- frame: composed rotation, fitted translation, both checked ----------
    obj_cam, smpl_t = load_bundle_object(args.bundle.expanduser().resolve(), args.bundle_key)
    obj_sim, root_sim, body_sim = load_pt(args.pt.expanduser().resolve())
    if len(obj_cam) != len(obj_sim):
        raise SystemExit(f"frame counts differ: bundle {len(obj_cam)}, tensor {len(obj_sim)}. "
                         "These are not the same clip.")
    R = composed_rotation(None if args.no_rotate else args.rotate_axis, args.rotate_degrees,
                          None if args.no_rotate else args.from_calib)
    t, rms_obj = fit_translation(R, obj_cam, obj_sim)
    _, rms_root = fit_translation(R, smpl_t, root_sim)
    print(f"# frame fit over {len(obj_cam)} frames: object residual {rms_obj * 100:.1f} cm, "
          f"root residual {rms_root * 100:.1f} cm (the root differs from smpl_t by the pelvis "
          f"offset, so it runs higher)")
    if rms_obj > 0.15:
        print(f"# WARNING: {rms_obj * 100:.0f} cm is too large for the same clip in two frames. "
              "Wrong --rotate-* for this .pt, or not this bundle's clip. Do not use this pose.")
    # Kabsch as a second opinion, when the motion can support one.
    cam_from_bundle = _load_sibling("cam_from_bundle")
    R_fit, _, rms_fit = cam_from_bundle.rigid_fit(obj_cam, obj_sim)
    sp = spread_m(obj_cam)
    gap = rotation_gap_deg(R, R_fit)
    if sp > 0.05:
        print(f"# rotation fitted from the object path agrees with the composed one to "
              f"{gap:.1f} deg (path spread {sp * 100:.0f} cm, fit residual {rms_fit * 100:.1f} cm)")
        if gap > 5.0:
            print("# WARNING: the fitted rotation disagrees with the composed one; the .pt was "
                  "probably rotated differently from what --rotate-* says.")
    else:
        print(f"# object path too flat ({sp * 100:.0f} cm) to fit a rotation independently; "
              "trusting the composed one")

    # ---- the prop in simulator frame ------------------------------------------
    R_w = R @ pose_cam[:3, :3]
    p_w = R @ pose_cam[:3, 3] + t
    quat = sRot.from_matrix(R_w).as_quat()          # xyzw, what gymapi.Quat takes

    # ---- mesh + URDF ------------------------------------------------------------
    finalize = _load_sibling("cari4d_finalize")
    mesh_dst = assets / "objects" / "objects" / name / f"{name}.obj"
    urdf_dst = assets / "objects" / f"{name}.urdf"
    if args.dry_run:
        import trimesh
        verts = np.asarray(trimesh.load(str(mesh_src), force="mesh", process=False).vertices, np.float64)
        n_faces = -1
    else:
        verts, n_faces = clean_mesh(mesh_src, mesh_dst)
        urdf_dst.write_text(finalize.URDF_TEMPLATE.format(object_name=name))
    verts_w = verts @ R_w.T + p_w
    lowest = float(verts_w[:, 2].min())
    feet = body_sim[:, [7, 8, 10, 11], 2].min()
    print(f"# prop lowest vertex at z={lowest:+.3f} m (the clip's lowest foot is at "
          f"z={feet:+.3f}); prop extent {np.round(verts_w.max(0) - verts_w.min(0), 2).tolist()} m")
    shift = 0.0
    if args.snap_to_floor:
        shift = -lowest
        p_w = p_w + np.array([0.0, 0.0, shift])
        print(f"# snapped to the floor: shifted {shift:+.3f} m in z")
    elif abs(lowest) > 0.10:
        print(f"# NOTE: {abs(lowest) * 100:.0f} cm {'above' if lowest > 0 else 'below'} the floor. "
              "A chair or bench should touch it; --snap-to-floor if the registration depth is the "
              "only thing off, otherwise look at the sidecar's checks first.")
    pelvis = root_sim.mean(axis=0)
    print(f"# prop centre {np.round(p_w, 3).tolist()}; mean pelvis {np.round(pelvis, 3).tolist()}; "
          f"prop is {np.linalg.norm(p_w - pelvis):.2f} m from it")

    entry = {"asset": f"{name}.urdf", "pos": [round(float(v), 4) for v in p_w],
             "quat": [round(float(v), 4) for v in quat]}
    if not args.no_vhacd:
        entry["vhacd"] = True
    record = {
        "entry": entry, "name": name, "sidecar": str(args.sidecar.expanduser().resolve()),
        "bundle": str(args.bundle.expanduser().resolve()), "pt": str(args.pt.expanduser().resolve()),
        "rotation": ("none" if args.no_rotate else (args.from_calib or f"{args.rotate_axis}/{args.rotate_degrees}")),
        "fit": {"object_rms_m": rms_obj, "root_rms_m": rms_root, "kabsch_gap_deg": gap,
                "object_spread_m": sp, "lowest_vertex_z_m": lowest, "floor_shift_m": shift},
        "sidecar_checks": side.get("checks", {}), "mesh_faces": n_faces,
    }
    if not args.dry_run:
        (assets / "objects" / f"{name}.scene.json").write_text(json.dumps(record, indent=1))
        print(f"# wrote {mesh_dst}, {urdf_dst}, {name}.scene.json")

    print()
    print("  staticScene:")
    print(f"    # {side['prop']} from {side['seq']}: CARI4D static_prop sidecar, "
          f"object-fit residual {rms_obj * 100:.1f} cm")
    print(f"    - asset: {entry['asset']}")
    print(f"      pos: {entry['pos']}")
    print(f"      quat: {entry['quat']}   # xyzw")
    if "vhacd" in entry:
        print("      vhacd: true")
    return 0


if __name__ == "__main__":
    sys.exit(main())

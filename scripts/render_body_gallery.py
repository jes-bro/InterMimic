#!/usr/bin/env python3
"""Spawn every per-subject body in ONE Isaac Gym env, side by side, render one image.

Two modes:
  (default) CAPSULE -- the MJCF bodies actually simulated (spheres + capsules).
  --mesh    SURFACE -- the real shaped SMPL-X surface mesh per subject, loaded
            into the sim as static triangle meshes.

Diagnostic: proves subjectBodies swaps the body, and makes proportional
differences (height/limb/girth) visible that a moving replay hides. Rest pose =>
only SHAPE varies. Gravity off; it's a static portrait, not a sim.

Runs on the cluster (Isaac Gym + GPU). CAPSULE mode needs the per-subject MJCFs;
--mesh mode needs the SMPL-X models (SMPLX_MODELS or ~/Downloads/models/smplx)
and scripts/omomo_betas.npz. From the repo root, intermimic-gym env:

  python3 scripts/render_body_gallery.py                       # capsules, all subjects
  python3 scripts/render_body_gallery.py --mesh                # SMPL-X surfaces
  python3 scripts/render_body_gallery.py --mesh --subjects sub4 sub10 sub13 sub16 --out g.png
"""
import argparse
import glob
import importlib.util
import os
import re

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym import gymapi
from isaacgym.torch_utils import get_axis_params
import numpy as np


ASSET_DIR = "isaacgym/src/intermimic/data/assets/smplx"
CHAR_H = 0.89


def discover_mjcf(subjects):
    if subjects:
        paths = [os.path.join(ASSET_DIR, f"smplx_omomo_{s}.xml") for s in subjects]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise SystemExit("FATAL: missing MJCF(s):\n  " + "\n  ".join(missing))
        return list(zip(subjects, paths))
    paths = sorted(glob.glob(os.path.join(ASSET_DIR, "smplx_omomo_sub*.xml")),
                   key=lambda p: int(re.search(r"sub(\d+)", p).group(1)))
    if not paths:
        raise SystemExit(f"FATAL: no smplx_omomo_sub*.xml under {ASSET_DIR}/")
    subs = [re.search(r"(sub\d+)", os.path.basename(p)).group(1) for p in paths]
    return list(zip(subs, paths))


def _load_shaper():
    """Import scripts/smplx_mesh by path (a ROS 'scripts' pkg can shadow it)."""
    spec = importlib.util.spec_from_file_location(
        "smplx_mesh", os.path.join(os.path.dirname(__file__), "smplx_mesh.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SMPLXShaper


def make_sim(gym):
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)     # freeze the rest pose
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0
    sp.physx.num_threads = 4
    sp.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise SystemExit("FATAL: create_sim failed (no GPU / graphics device?)")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)
    return sim


def build_capsules(gym, sim, bodies, spacing):
    """MJCF actors in one env, row along +Y; camera down +X. Returns (env, cam_pos)."""
    ao = gymapi.AssetOptions()
    ao.angular_damping = 0.01
    ao.max_angular_velocity = 100.0
    ao.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    n = len(bodies)
    span = (n - 1) * spacing
    env = gym.create_env(sim, gymapi.Vec3(-2, -span/2-2, 0),
                         gymapi.Vec3(2, span/2+2, 3), 1)
    for i, (s, p) in enumerate(bodies):
        asset = gym.load_asset(sim, os.path.dirname(p), os.path.basename(p), ao)
        if asset is None:
            raise SystemExit(f"FATAL: failed to load {p}")
        pose = gymapi.Transform()
        base = get_axis_params(CHAR_H, 2)
        pose.p = gymapi.Vec3(base[0], base[1] - span/2 + i*spacing, base[2])
        pose.r = gymapi.Quat(0, 0, 0, 1)
        h = gym.create_actor(env, asset, pose, s, 0, 1, 0)
        col = (gymapi.Vec3(0.80, 0.55, 0.32) if i % 2 == 0
               else gymapi.Vec3(0.36, 0.56, 0.78))
        for j in range(gym.get_actor_rigid_body_count(env, h)):
            gym.set_rigid_body_color(env, h, j, gymapi.MESH_VISUAL, col)
    return env, gymapi.Vec3(max(4.0, span*0.65), 0.0, 1.2), gymapi.Vec3(0.0, 0.0, 0.9)


def _mesh_source(mesh_npz, models, betas):
    """Return (mesh_fn(subject)->(v,f), gender_dict, subject_list).

    Prefer a prebaked npz (no SMPL-X models needed on this box) if given; else
    compute from the models. The npz is produced on a machine that HAS the models
    (scripts/smplx_mesh.py bakes it) and copied over -- see --mesh-npz."""
    if mesh_npz:
        if not os.path.isfile(mesh_npz):
            raise SystemExit(f"FATAL: --mesh-npz {mesh_npz} not found")
        d = np.load(mesh_npz, allow_pickle=True)
        faces = d["_faces"].astype(np.uint32)
        gender = dict(x.split(":") for x in d["_genders"])
        subs = [str(s) for s in d["_subjects"]]

        def fn(s):
            if s not in d.files:
                raise SystemExit(f"FATAL: {s} not in {mesh_npz} (have: {' '.join(subs)})")
            return d[s].astype(np.float32), faces
        return fn, gender, subs

    Shaper = _load_shaper()
    sh = Shaper(models, betas)
    return (lambda s: sh.mesh(s)), sh.gender, sh.subjects()


def build_meshes(gym, sim, subjects, spacing, mesh_fn, gender):
    """SMPL-X surface meshes as static triangle meshes, row along +X (bodies face
    -Y). Camera in front on -Y. Returns (env, cam_pos, cam_target)."""
    n = len(subjects)
    span = (n - 1) * spacing
    for i, s in enumerate(subjects):
        v, f = mesh_fn(s)                            # Z-up, feet on ground, centred
        tm = gymapi.TriangleMeshParams()
        tm.nb_vertices = v.shape[0]
        tm.nb_triangles = f.shape[0]
        tm.transform.p = gymapi.Vec3(-span/2 + i*spacing, 0.0, 0.0)  # spread on X
        gym.add_triangle_mesh(sim,
                              v.flatten(order="C"),
                              f.flatten(order="C").astype(np.uint32),
                              tm)
        print(f"  {s:7s} gender={gender.get(s,'?'):7s} h={v[:,2].max():.3f}m", flush=True)
    # A camera still needs an env; keep it empty.
    env = gym.create_env(sim, gymapi.Vec3(-span/2-2, -4, 0),
                         gymapi.Vec3(span/2+2, 4, 3), 1)
    dist = max(3.5, span * 0.75)
    return env, gymapi.Vec3(0.0, -dist, 1.15), gymapi.Vec3(0.0, 0.0, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*",
                    default=(os.environ.get("SUBJECTS", "").split() or None))
    ap.add_argument("--mesh", action="store_true",
                    help="render SMPL-X SURFACE meshes instead of the capsule MJCFs")
    ap.add_argument("--out", default=os.environ.get("OUT", "body_gallery.png"))
    ap.add_argument("--spacing", type=float, default=None,
                    help="metres between bodies (default: 0.7 capsule, 1.1 mesh)")
    ap.add_argument("--models", default=os.environ.get(
        "SMPLX_MODELS", "~/Downloads/models/smplx"))
    ap.add_argument("--betas", default="scripts/omomo_betas.npz")
    ap.add_argument("--mesh-npz", default=os.environ.get("MESH_NPZ") or None,
                    help="prebaked meshes (from smplx_mesh bake); avoids needing "
                         "SMPL-X models on this machine")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()

    spacing = a.spacing if a.spacing is not None else (1.1 if a.mesh else 0.7)
    gym = gymapi.acquire_gym()
    sim = make_sim(gym)

    if a.mesh:
        mesh_fn, gender, all_subs = _mesh_source(a.mesh_npz, a.models, a.betas)
        subs = a.subjects or all_subs
        src = a.mesh_npz if a.mesh_npz else f"models @ {a.models}"
        print(f"[gallery] MESH ({src}): {len(subs)} bodies: {' '.join(subs)}", flush=True)
        env, cam_p, cam_t = build_meshes(gym, sim, subs, spacing, mesh_fn, gender)
    else:
        bodies = discover_mjcf(a.subjects)
        print(f"[gallery] CAPSULE: {len(bodies)} bodies: "
              f"{' '.join(s for s, _ in bodies)}", flush=True)
        env, cam_p, cam_t = build_capsules(gym, sim, bodies, spacing)

    cp = gymapi.CameraProperties()
    cp.width, cp.height = a.width, a.height
    cam = gym.create_camera_sensor(env, cp)
    gym.set_camera_location(cam, env, cam_p, cam_t)

    gym.prepare_sim(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
    img = img.reshape(a.height, a.width, 4)[..., :3]

    import imageio
    out = os.path.abspath(a.out)
    imageio.imwrite(out, img)
    print(f"[gallery] wrote {out}", flush=True)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

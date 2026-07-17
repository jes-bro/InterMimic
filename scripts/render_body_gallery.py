#!/usr/bin/env python3
"""Spawn every per-subject SMPL-X body in ONE Isaac Gym env, side by side, and
render a single image. Diagnostic: proves subjectBodies actually swaps the MJCF,
and makes the proportional differences (height/limb length/girth) visible that a
moving replay hides.

All actors live in a SINGLE env, spaced along +Y, standing in rest pose (dof=0).
Gravity is off so nothing falls -- it's a static portrait, not a sim. One wide
camera frames the whole row and writes one PNG.

Runs on the cluster (needs Isaac Gym + a GPU + the per-subject MJCFs, which only
exist there). From the repo root, in the intermimic-gym env:

  python3 scripts/render_body_gallery.py
  python3 scripts/render_body_gallery.py --subjects sub4 sub10 sub13 sub16 --out gallery_suspects.png
  SUBJECTS="sub2 sub16" python3 scripts/render_body_gallery.py     # env-var form
"""
import argparse
import glob
import os
import re

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym import gymapi
from isaacgym.torch_utils import get_axis_params
import numpy as np


ASSET_DIR = "isaacgym/src/intermimic/data/assets/smplx"
CHAR_H = 0.89          # root height for a standing SMPL-X (matches humanoid.py:340)


def discover(subjects):
    if subjects:
        paths = [os.path.join(ASSET_DIR, f"smplx_omomo_{s}.xml") for s in subjects]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise SystemExit("FATAL: missing MJCF(s):\n  " + "\n  ".join(missing))
        return list(zip(subjects, paths))
    paths = sorted(glob.glob(os.path.join(ASSET_DIR, "smplx_omomo_sub*.xml")),
                   key=lambda p: int(re.search(r"sub(\d+)", p).group(1)))
    if not paths:
        raise SystemExit(f"FATAL: no smplx_omomo_sub*.xml under {ASSET_DIR}/ -- "
                         f"these are generated on the cluster; none found here.")
    subs = [re.search(r"(sub\d+)", os.path.basename(p)).group(1) for p in paths]
    return list(zip(subs, paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*",
                    default=(os.environ.get("SUBJECTS", "").split() or None))
    ap.add_argument("--out", default=os.environ.get("OUT", "body_gallery.png"))
    ap.add_argument("--spacing", type=float, default=0.7,
                    help="metres between adjacent bodies along +Y")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()

    bodies = discover(a.subjects)
    n = len(bodies)
    print(f"[gallery] {n} bodies: {' '.join(s for s, _ in bodies)}", flush=True)

    gym = gymapi.acquire_gym()

    # --- sim: Z-up, gravity OFF so the rest pose holds without falling ---
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0
    sp.physx.num_threads = 4
    sp.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise SystemExit("FATAL: create_sim failed (no GPU / no graphics device?)")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    # --- load every MJCF (same options as humanoid.py so bodies match the sim) ---
    ao = gymapi.AssetOptions()
    ao.angular_damping = 0.01
    ao.max_angular_velocity = 100.0
    ao.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    assets = []
    for s, p in bodies:
        asset = gym.load_asset(sim, os.path.dirname(p), os.path.basename(p), ao)
        if asset is None:
            raise SystemExit(f"FATAL: failed to load {p}")
        assets.append(asset)

    # --- ONE env; all bodies spawned in a row along +Y, centred on 0 ---
    span = (n - 1) * a.spacing
    lower = gymapi.Vec3(-2.0, -span / 2 - 2.0, 0.0)
    upper = gymapi.Vec3(2.0, span / 2 + 2.0, 3.0)
    env = gym.create_env(sim, lower, upper, 1)

    up_idx = 2  # Z
    for i, ((s, _), asset) in enumerate(zip(bodies, assets)):
        pose = gymapi.Transform()
        y = -span / 2 + i * a.spacing
        base = get_axis_params(CHAR_H, up_idx)          # (x, y, z) with z=CHAR_H
        pose.p = gymapi.Vec3(base[0], base[1] + y, base[2])
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        h = gym.create_actor(env, asset, pose, s, 0, 1, 0)
        # tint every other body so adjacent ones are easy to tell apart
        col = (gymapi.Vec3(0.80, 0.55, 0.32) if i % 2 == 0
               else gymapi.Vec3(0.36, 0.56, 0.78))
        for j in range(gym.get_actor_rigid_body_count(env, h)):
            gym.set_rigid_body_color(env, h, j, gymapi.MESH_VISUAL, col)

    # --- wide camera looking down the row from +X ---
    cp = gymapi.CameraProperties()
    cp.width, cp.height = a.width, a.height
    cam = gym.create_camera_sensor(env, cp)
    # Far enough back on X to see the whole span; centred on the row.
    dist = max(4.0, span * 0.65)
    gym.set_camera_location(cam, env,
                            gymapi.Vec3(dist, 0.0, 1.2),
                            gymapi.Vec3(0.0, 0.0, 0.9))

    gym.prepare_sim(sim)
    # No gravity + no actions => one step just settles the render transforms;
    # the bodies do not move from their rest pose.
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
    img = img.reshape(a.height, a.width, 4)[..., :3]

    import imageio
    out = os.path.abspath(a.out)
    imageio.imwrite(out, img)
    print(f"[gallery] wrote {out}  ({n} bodies, +Y spacing {a.spacing} m)", flush=True)

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

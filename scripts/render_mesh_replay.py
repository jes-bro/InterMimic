#!/usr/bin/env python3
"""Render the SMPL-X SURFACE doing what the humanoid did, as an mp4 (offline).

Stage 2 of the mesh-replay pipeline. Drive it from either:
  --dump traj.npz   a policy rollout dumped by DUMP_TRAJ (pose_from_bodies; most
                    faithful -- the sim's own per-frame global body state).
  --clip x.pt       a raw OMOMO clip's ground-truth motion (pose_from_dof); needs
                    --subject to say which body to shape.

Shaded matplotlib render (no pyrender/GPU needed), fixed camera, floor, object
drawn as a marker. So a viewer can watch the mesh perform the motion and tell
subjects apart.

  python3 scripts/render_mesh_replay.py --dump rollout.npz --out replay.mp4
  python3 scripts/render_mesh_replay.py --clip InterAct/OMOMO_new/sub2_largetable_000.pt \
      --subject sub2 --out ref.mp4
"""
import argparse
import importlib.util
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio


def _poser(models, betas):
    spec = importlib.util.spec_from_file_location(
        "smplx_pose", os.path.join(os.path.dirname(__file__), "smplx_pose.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SMPLXPoser(models_dir=models, betas_path=betas)


def _ik_frames(P, subject, target_joints, obj_pos, obj_rot, warm):
    """Retarget: fit a native SMPL-X pose to the target joints per frame, then
    forward the clean surface. Fixes the arm twist that MJCF-frame skinning has.

    obj_rot may be None, in which case the object is drawn unrotated -- fine for
    a sphere, wrong for anything with a recognisable orientation.
    """
    print(f"[mesh-replay] IK retargeting {len(target_joints)} frames "
          f"(subject={subject})...", flush=True)
    poses, trans = P.fit_sequence(subject, target_joints, iters_warm=warm, verbose=False)
    for t in range(len(target_joints)):
        v, f = P.verts_from_pose(subject, poses[t], trans[t])
        yield v, f, obj_pos[t], (None if obj_rot is None else obj_rot[t])


def subject_key(raw, available):
    """Return the betas key a dump's subject string refers to.

    The player records whatever names the body, and that is only sometimes a
    bare subject id: with subjectBodies set it is 'sub4', but a config using
    robotType records the MJCF path instead -- 'smplx/smplh_behave_sub100.xml'.
    Both mean the same body, and neither the sim nor the archive is wrong, so
    the sub<N> is pulled out of whichever form arrived.

    Raises:
        SystemExit: if no key matches, listing what is available -- a wrong
            match would render a real person who is not this one.
    """
    if raw in available:
        return raw
    found = re.findall(r"sub\d+", raw)
    for candidate in found:
        if candidate in available:
            print(f"[mesh-replay] dump names the body {raw!r}; using betas "
                  f"entry {candidate!r}")
            return candidate
    raise SystemExit(
        f"FATAL: dump subject {raw!r} has no betas entry. Available: "
        f"{sorted(available)}. Pass --subject to say which one, or add it with "
        f"scripts/add_subject_betas.py")


def frames_from_dump(P, path, warm, stride, subject=None):
    """Frames from a simulator rollout: the sim's own global body state."""
    d = np.load(path, allow_pickle=True)
    subject = subject_key(subject or str(d["subject"]), set(P.betas.files))
    rot = d["obj_rot"][::stride] if "obj_rot" in d.files else None
    return _ik_frames(P, subject, d["body_pos"][::stride], d["obj_pos"][::stride],
                      rot, warm)


def frames_from_clip(P, path, subject, warm, stride):
    """Frames from a reference clip, i.e. what the policy is asked to imitate."""
    import torch
    x = torch.load(path, map_location="cpu", weights_only=False).detach().numpy()
    tgt = x[::stride, 162:318].reshape(-1, 52, 3)
    return _ik_frames(P, subject, tgt, x[::stride, 318:321],
                      x[::stride, 321:325], warm)


def load_object_mesh(path, max_faces):
    """Return (verts, faces) for the object, decimated enough to animate.

    A reconstructed mesh runs to tens of thousands of faces -- the basketball is
    81,836 -- and matplotlib draws every one as a separate polygon, so a full
    mesh turns a 100-frame render into an overnight job. Decimating once at load
    costs nothing visually at this size on screen.

    Returns:
        (verts, faces), or (None, None) if the mesh cannot be read, since a
        marker is better than no render.
    """
    try:
        import trimesh
    except ImportError:
        print("[mesh-replay] trimesh not available; drawing the object as a marker")
        return None, None
    if not os.path.isfile(path):
        print(f"[mesh-replay] no object mesh at {path}; drawing a marker")
        return None, None
    mesh = trimesh.load(path, force="mesh", process=False)
    n = len(mesh.faces)
    if n > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
            print(f"[mesh-replay] object mesh {n} -> {len(mesh.faces)} faces")
        except Exception as exc:
            print(f"[mesh-replay] could not decimate ({exc}); rendering all "
                  f"{n} faces, which will be slow")
    else:
        print(f"[mesh-replay] object mesh {n} faces")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    # Centre it: the simulator's pose refers to the body origin, and a mesh
    # whose vertices are offset from that would follow the trajectory displaced.
    verts = verts - verts.mean(axis=0)
    return verts, np.asarray(mesh.faces, dtype=np.int64)


def quat_to_mat(q):
    """Convert an (x, y, z, w) quaternion to a rotation matrix.

    Isaac Gym stores object rotation in this order, matching the layout the
    591-channel clips use.
    """
    x, y, z, w = np.asarray(q, dtype=np.float64) / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_object(verts, pos, rot):
    """Place the object's vertices at a frame's pose."""
    if rot is None:
        return verts + np.asarray(pos, dtype=np.float64)
    return verts @ quat_to_mat(rot).T + np.asarray(pos, dtype=np.float64)


def view_angles_from_camera(cam_pos, target):
    """Return matplotlib (elev, azim) for a camera at cam_pos looking at target.

    matplotlib's 3D axes are aimed with two spherical angles about the data,
    not with a camera pose, so a viewpoint worked out in world coordinates --
    a real camera's, say -- has to be converted. azim is measured in the x-y
    plane from +x; elev is the angle above that plane.

    The result matches the DIRECTION the camera looks from, not its projection:
    matplotlib does not model focal length, so the framing still differs even
    when the angle agrees.
    """
    d = np.asarray(cam_pos, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    r = np.linalg.norm(d)
    if r < 1e-9:
        raise SystemExit("camera position and target coincide; no direction to view from")
    elev = float(np.degrees(np.arcsin(np.clip(d[2] / r, -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(d[1], d[0])))
    return elev, azim


def parse_vec3(text):
    """Parse an 'x,y,z' string into a float array.

    Raises:
        SystemExit: on anything that is not three numbers, rather than viewing
            from a silently defaulted direction.
    """
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(f"expected 'x,y,z', got {text!r}")
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        raise SystemExit(f"expected three numbers, got {text!r}")


def shade(v, f, light=np.array([0.3, -0.6, 0.8])):
    """Per-face lambert shading from vertex normals of the tri mesh."""
    tri = v[f]                                        # (F,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = light / np.linalg.norm(light)
    b = np.clip(n @ light, 0, 1)
    return 0.25 + 0.7 * b                             # ambient + diffuse in [0.25,0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="rollout npz from DUMP_TRAJ")
    ap.add_argument("--clip", help="raw OMOMO .pt clip (ground-truth motion)")
    ap.add_argument("--subject", help="body to shape. Required with --clip; with "
                                      "--dump it overrides whatever the sim "
                                      "recorded, which may be an MJCF path "
                                      "rather than a subject id")
    ap.add_argument("--out", default="mesh_replay.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--models", default=None)
    ap.add_argument("--betas", default="scripts/omomo_betas.npz")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--ik-iters", type=int, default=100,
                    help="IK warm-start iterations per frame (higher = tighter fit)")
    ap.add_argument("--object", default=None,
                    help="path to the object's .obj, to draw it as real "
                         "geometry instead of a marker. e.g. isaacgym/src/"
                         "intermimic/data/assets/objects/objects/bball/bball.obj")
    ap.add_argument("--obj-faces", type=int, default=800,
                    help="decimate the object to this many faces (default: 800); "
                         "matplotlib draws each face separately, so a full "
                         "reconstruction is unusably slow")
    ap.add_argument("--bg", default="black",
                    help="background colour, any matplotlib colour "
                         "(default: black). The mesh shading is unchanged, so a "
                         "light ground needs a light value here.")
    ap.add_argument("--cam-pos", default=None, metavar="X,Y,Z",
                    help="view from this world position instead of --elev/--azim. "
                         "Pass what scripts/cam_from_bundle.py prints, to look "
                         "from where the take was filmed.")
    ap.add_argument("--cam-target", default=None, metavar="X,Y,Z",
                    help="what --cam-pos looks at (default: the motion's centre)")
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--azim", type=float, default=55.0)
    a = ap.parse_args()

    if bool(a.dump) == bool(a.clip):
        raise SystemExit("FATAL: pass exactly one of --dump / --clip")
    if a.clip and not a.subject:
        raise SystemExit("FATAL: --clip needs --subject (which body to shape)")

    P = _poser(a.models, a.betas)
    gen = (frames_from_dump(P, a.dump, a.ik_iters, a.stride, a.subject) if a.dump
           else frames_from_clip(P, a.clip, a.subject, a.ik_iters, a.stride))

    # Stride is applied BEFORE the IK fit (in the generators), so don't re-subsample.
    frames = list(gen)
    if not frames:
        raise SystemExit("FATAL: no frames produced")
    elev, azim = a.elev, a.azim
    obj_v, obj_f = (None, None)
    if a.object:
        obj_v, obj_f = load_object_mesh(a.object, a.obj_faces)

    allv = np.concatenate([v for v, _, _, _ in frames], 0)
    ctr = allv.mean(0)
    if a.cam_pos:
        target = parse_vec3(a.cam_target) if a.cam_target else ctr
        elev, azim = view_angles_from_camera(parse_vec3(a.cam_pos), target)
        print(f"[mesh-replay] viewing from {a.cam_pos} -> elev={elev:.1f} "
              f"azim={azim:.1f}")
    rng = float(np.abs(allv - ctr).max()) * 1.05
    faces = frames[0][1]

    # imageio's FFMPEG plugin may be absent even when the ffmpeg BINARY exists, so
    # write PNG frames and assemble with ffmpeg directly (robust, dep-light).
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="meshreplay_")
    for i, (v, f, obj, orot) in enumerate(frames):
        fig = plt.figure(figsize=(6, 8), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        # Both the figure and the axes: set_axis_off hides the panes but leaves
        # the canvas behind them, so colouring only one shows a border of the
        # other around the render.
        fig.patch.set_facecolor(a.bg)
        ax.set_facecolor(a.bg)
        c = shade(v, f)
        pc = Poly3DCollection(v[f], linewidths=0)
        pc.set_facecolor(plt.cm.get_cmap("Blues")(0.35 + 0.5 * (c - c.min()) / (c.ptp() + 1e-9)))
        ax.add_collection3d(pc)
        # The object: real geometry when we have it, a marker otherwise.
        if obj_v is not None:
            ov = pose_object(obj_v, obj, orot)
            oc = shade(ov, obj_f)
            opc = Poly3DCollection(ov[obj_f], linewidths=0)
            opc.set_facecolor(plt.cm.get_cmap("Oranges")(
                0.35 + 0.5 * (oc - oc.min()) / (oc.ptp() + 1e-9)))
            ax.add_collection3d(opc)
        else:
            ax.scatter(obj[0], obj[1], obj[2], s=60, c="#e08a2b", depthshade=False)
        ax.set_xlim(ctr[0]-rng, ctr[0]+rng); ax.set_ylim(ctr[1]-rng, ctr[1]+rng)
        ax.set_zlim(0, 2*rng)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), img)
        plt.close(fig)
        if i % 20 == 0:
            print(f"  frame {i+1}/{len(frames)}", flush=True)

    out = os.path.abspath(a.out)
    # Pick a software encoder that this ffmpeg build actually has (libx264 is
    # often absent; mpeg4 is a universal fallback).
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    codec = "libx264" if "libx264" in enc else "mpeg4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
           "-i", os.path.join(tmp, "f%05d.png"), "-c:v", codec]
    if codec == "libx264":
        cmd += ["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    else:
        cmd += ["-q:v", "4"]                          # mpeg4 quality (1 best .. 31)
    cmd.append(out)
    subprocess.run(cmd, check=True)
    print(f"[mesh-replay] encoder={codec}", flush=True)
    for p in os.listdir(tmp):
        os.remove(os.path.join(tmp, p))
    os.rmdir(tmp)
    print(f"[mesh-replay] wrote {len(frames)} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()

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


def frames_from_dump(P, path):
    d = np.load(path, allow_pickle=True)
    subject = str(d["subject"])
    if subject not in P.betas.files:
        raise SystemExit(f"FATAL: dump subject {subject!r} has no betas entry")
    n = len(d["body_rot"])
    obj = d["obj_pos"]
    for t in range(n):
        v, f = P.pose_from_bodies(subject, d["body_rot"][t], d["body_pos"][t])
        yield v, f, obj[t]


def frames_from_clip(P, path, subject):
    import torch
    x = torch.load(path, map_location="cpu", weights_only=False).detach().numpy()
    for t in range(len(x)):
        v, f = P.pose_from_dof(subject, x[t, 9:162], x[t, 0:3], x[t, 3:7])
        yield v, f, x[t, 318:321]                    # obj_pos in the raw layout


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
    ap.add_argument("--subject", help="body to shape (required with --clip)")
    ap.add_argument("--out", default="mesh_replay.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--models", default=None)
    ap.add_argument("--betas", default="scripts/omomo_betas.npz")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--azim", type=float, default=55.0)
    a = ap.parse_args()

    if bool(a.dump) == bool(a.clip):
        raise SystemExit("FATAL: pass exactly one of --dump / --clip")
    if a.clip and not a.subject:
        raise SystemExit("FATAL: --clip needs --subject (which body to shape)")

    P = _poser(a.models, a.betas)
    gen = (frames_from_dump(P, a.dump) if a.dump
           else frames_from_clip(P, a.clip, a.subject))

    # Prepass: collect frames (subsampled) + a stable bounding box so the camera
    # doesn't jitter frame to frame.
    frames = [fr for i, fr in enumerate(gen) if i % a.stride == 0]
    if not frames:
        raise SystemExit("FATAL: no frames produced")
    allv = np.concatenate([v for v, _, _ in frames], 0)
    ctr = allv.mean(0)
    rng = float(np.abs(allv - ctr).max()) * 1.05
    faces = frames[0][1]

    # imageio's FFMPEG plugin may be absent even when the ffmpeg BINARY exists, so
    # write PNG frames and assemble with ffmpeg directly (robust, dep-light).
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="meshreplay_")
    for i, (v, f, obj) in enumerate(frames):
        fig = plt.figure(figsize=(6, 8), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        c = shade(v, f)
        pc = Poly3DCollection(v[f], linewidths=0)
        pc.set_facecolor(plt.cm.get_cmap("Blues")(0.35 + 0.5 * (c - c.min()) / (c.ptp() + 1e-9)))
        ax.add_collection3d(pc)
        # object marker + a faint floor at z=0
        ax.scatter(obj[0], obj[1], obj[2], s=60, c="#e08a2b", depthshade=False)
        ax.set_xlim(ctr[0]-rng, ctr[0]+rng); ax.set_ylim(ctr[1]-rng, ctr[1]+rng)
        ax.set_zlim(0, 2*rng)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=a.elev, azim=a.azim)
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

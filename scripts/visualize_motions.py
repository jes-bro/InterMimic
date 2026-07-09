#!/usr/bin/env python3
"""Fast, no-sim visual index of OMOMO source motions -- for MANUAL activity labeling.

Each clip is a [T, 591] tensor. We only need three slices (indices verified
against intermimic.py _load_motion):

    body_pos    = [:, 162:162+52*3]  -> 52 SMPL-X joints, world xyz
    obj_pos     = [:, 318:321]       -> object centroid xyz
    contact_obj = round([:, 330])    -> 1 while the hand(s) hold the object

Two output modes (both show the body skeleton (blue) + object square: green=no
contact, red=in contact; no Isaac Gym, no GPU, no display -- pure matplotlib):
  --mode video (default) -- one animated clip each (gif, the most foolproof way
       to read the action). ~0.6 MB and a few seconds per clip.
  --mode filmstrip       -- a static strip of N keyframes; fast to scan but a
       fast/ambiguous motion can be hard to read.
Output is grouped into per-object subfolders, plus a labels_template.csv (one
row per clip, an empty `activity` column to fill in) and an index.html that
embeds every clip (video players for mp4, inline for gif/png).

Usage (from repo root):
    # smoke test on the bundled clips (video/gif)
    python scripts/visualize_motions.py --motion-dir InterAct/OMOMO --out-dir viz_motions

    # the full 4421-clip dataset -- START WITH A SUBSET (video of all is hours):
    python scripts/visualize_motions.py --motion-dir ~/new_one/OMOMO_new \
        --objects woodchair --out-dir viz_woodchair          # one object at a time

    # halve render time/size with a coarser frame rate, or use static filmstrips
    python scripts/visualize_motions.py --motion-dir ~/new_one/OMOMO_new --video-stride 2
    python scripts/visualize_motions.py --motion-dir ~/new_one/OMOMO_new --mode filmstrip
"""
import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless -- write pngs, never open a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# --- 591-channel layout (see module docstring / intermimic.py _load_motion) ---
BODY_POS_SLICE = slice(162, 162 + 52 * 3)
OBJ_POS_SLICE = slice(318, 321)
CONTACT_OBJ_IDX = 330
N_BODY_JOINTS = 52

# SMPL-X body kinematic tree for the first 22 (torso+limb) joints; parent[j] is
# the index j connects to. The remaining 30 joints are the hands -- we scatter
# those rather than draw finger bones (too cluttered at this zoom). This is the
# canonical SMPL-X body ordering; we verify it renders as a human on a test clip.
SMPLX_BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
                      7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def parse_name(path):
    """sub<N>_<object>_<seq>.pt -> (subject, object, seq).  Returns Nones on
    an unrecognized name rather than guessing."""
    m = re.match(r"(sub\d+)_([a-z]+)_(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def load_clip(path):
    """Return (body[T,52,3], obj[T,3], contact[T]) for one clip, or None if the
    tensor isn't the expected [T,591] shape (fail visibly, don't fabricate)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(data) or data.ndim != 2 or data.shape[1] < 331:
        return None
    data = data.detach()  # clips are saved with requires_grad -> detach before numpy()
    T = data.shape[0]
    body = data[:, BODY_POS_SLICE].reshape(T, N_BODY_JOINTS, 3).numpy()
    obj = data[:, OBJ_POS_SLICE].numpy()
    contact = data[:, CONTACT_OBJ_IDX].round().numpy()
    return body, obj, contact


# Reorder axes so `up_axis` maps to matplotlib's vertical (z). z-up is verified
# correct for this Isaac-Gym-converted data.
_AXIS_ORDER = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}


def _fixed_box(body, obj):
    """Shared axis center+radius across the WHOLE clip so translation is visible
    (per-frame autoscale would hide the person/object moving through space)."""
    allpts = np.concatenate([body.reshape(-1, 3), obj], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    return (lo + hi) / 2.0, float((hi - lo).max()) / 2.0 + 0.1


def _draw_pose(ax, j, o, in_contact, order, center, radius, title):
    """Draw ONE frame into ax: skeleton bones + body/hand joints + object square.
    Shared by the filmstrip and the video so both render identically."""
    def xyz(p):
        return p[..., order[0]], p[..., order[1]], p[..., order[2]]
    for child, parent in enumerate(SMPLX_BODY_PARENTS):
        if parent >= 0:
            seg = np.stack([j[child], j[parent]], axis=0)
            ax.plot(*xyz(seg), color="steelblue", lw=1.4)
    ax.scatter(*xyz(j[:22]), s=7, color="navy")
    ax.scatter(*xyz(j[22:]), s=1.5, color="skyblue")           # hands
    ax.scatter(*[[v] for v in xyz(o)], s=80, marker="s",       # object centroid
               color=("red" if in_contact else "seagreen"),
               edgecolors="k", linewidths=0.4)
    c = center[list(order)]
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-72)
    if title:
        ax.set_title(title, fontsize=8)


def render_filmstrip(body, obj, contact, title, out_png, n_frames, up_axis):
    """Write an n_frames-wide keyframe filmstrip png (fast, scannable)."""
    order = _AXIS_ORDER[up_axis]
    center, radius = _fixed_box(body, obj)
    frame_idx = np.linspace(0, body.shape[0] - 1, n_frames).astype(int)
    fig = plt.figure(figsize=(2.6 * n_frames, 3.0))
    for i, fi in enumerate(frame_idx):
        ax = fig.add_subplot(1, n_frames, i + 1, projection="3d")
        _draw_pose(ax, body[fi], obj[fi], contact[fi] > 0, order, center, radius, f"t={fi}")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=80, bbox_inches="tight")
    plt.close(fig)


def render_video(body, obj, contact, title, out_path, up_axis, fps, frame_stride, fmt):
    """Write an animated clip (mp4 or gif) -- the most foolproof way to read the
    action. frame_stride subsamples frames to trade smoothness for render time."""
    import matplotlib.animation as animation
    order = _AXIS_ORDER[up_axis]
    center, radius = _fixed_box(body, obj)
    frames = list(range(0, body.shape[0], max(1, frame_stride)))
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(fi):
        ax.cla()
        _draw_pose(ax, body[fi], obj[fi], contact[fi] > 0, order, center, radius,
                   f"{title}  t={fi}")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps)
    try:
        if fmt == "mp4":
            # yuv420p + even dims so the mp4 plays in browsers / QuickTime.
            # NB: needs a SOFTWARE h264 encoder (libx264); some ffmpeg builds
            # ship only hardware encoders and will fail here -> use --format gif.
            writer = animation.FFMpegWriter(fps=fps, bitrate=1500,
                                            extra_args=["-pix_fmt", "yuv420p"])
        else:
            writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=90)
    finally:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion-dir", required=True,
                    help="dir of *.pt clips (e.g. InterAct/OMOMO or ~/new_one/OMOMO_new)")
    ap.add_argument("--out-dir", default="viz_motions",
                    help="where filmstrips / csv / index.html are written")
    ap.add_argument("--mode", choices=["video", "filmstrip"], default="video",
                    help="video = one animated clip each (most foolproof); "
                         "filmstrip = a static keyframe strip (fast to scan)")
    ap.add_argument("--format", choices=["mp4", "gif"], default="gif",
                    help="video container. gif (default) is foolproof -- plays in "
                         "any browser/viewer, only needs Pillow. mp4 is smaller but "
                         "needs a SOFTWARE h264 encoder (libx264); not on every box.")
    ap.add_argument("--fps", type=int, default=20, help="playback fps for video mode")
    ap.add_argument("--video-stride", type=int, default=1,
                    help="render every Nth frame in video mode (2-3 = faster/smaller)")
    ap.add_argument("--frames", type=int, default=6, help="keyframes per filmstrip")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="only these object names (default: all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth clip (subsample for a quick pass)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0=all)")
    ap.add_argument("--up-axis", choices=["x", "y", "z"], default="z",
                    help="which coord is vertical (z-up: verified correct for this "
                         "Isaac-Gym-converted OMOMO data)")
    args = ap.parse_args()

    motion_dir = os.path.expanduser(args.motion_dir)
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pt = sorted(glob.glob(os.path.join(motion_dir, "*.pt")))
    if not all_pt:
        raise SystemExit(f"[viz] no .pt clips under {motion_dir}")

    # Filter + subsample, fail loudly on unparseable names rather than skip silently.
    clips = []
    for p in all_pt:
        sub, obj, seq = parse_name(p)
        if sub is None:
            print(f"[viz] WARNING: unrecognized name, skipping: {os.path.basename(p)}")
            continue
        if args.objects and obj not in args.objects:
            continue
        clips.append((p, sub, obj, seq))
    clips = clips[:: args.stride]
    if args.limit:
        clips = clips[: args.limit]
    print(f"[viz] {len(clips)} clips to render from {motion_dir} "
          f"(objects={args.objects or 'all'}, stride={args.stride})")

    rows, by_object = [], {}
    for n, (p, sub, obj, seq) in enumerate(clips, 1):
        loaded = load_clip(p)
        if loaded is None:
            print(f"[viz] WARNING: unexpected tensor shape, skipping {os.path.basename(p)}")
            continue
        body, objp, contact = loaded
        obj_dir = out_dir / obj
        obj_dir.mkdir(exist_ok=True)
        if args.mode == "video":
            out_file = obj_dir / (Path(p).stem + "." + args.format)
            render_video(body, objp, contact, Path(p).stem, str(out_file),
                         args.up_axis, args.fps, args.video_stride, args.format)
        else:
            out_file = obj_dir / (Path(p).stem + ".png")
            render_filmstrip(body, objp, contact, Path(p).stem, str(out_file),
                             args.frames, args.up_axis)
        contact_frac = float((contact > 0).mean())
        rows.append({"clip": Path(p).stem, "subject": sub, "object": obj,
                     "seq": seq, "n_frames": body.shape[0],
                     "contact_frac": round(contact_frac, 3), "activity": ""})
        by_object.setdefault(obj, []).append(out_file.relative_to(out_dir))
        if n % 50 == 0 or n == len(clips):
            print(f"[viz]   {n}/{len(clips)} rendered")

    # labels template: one row per clip with an empty `activity` to fill in.
    csv_path = out_dir / "labels_template.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "subject", "object", "seq",
                                          "n_frames", "contact_frac", "activity"])
        w.writeheader()
        w.writerows(rows)

    # index.html: filmstrips grouped by object so you scan one activity-ish
    # bucket at a time and jot the label next to each clip name.
    html = ["<html><head><meta charset='utf-8'><title>OMOMO motions</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee}"
            "img{width:100%;max-width:1400px;display:block;margin:2px 0}"
            "h2{position:sticky;top:0;background:#222;padding:6px}"
            ".clip{margin:10px 0;border-bottom:1px solid #333}</style></head><body>",
            f"<h1>{len(rows)} clips — {len(by_object)} objects</h1>"]
    for obj in sorted(by_object):
        html.append(f"<h2>{obj} ({len(by_object[obj])})</h2>")
        for rel in sorted(by_object[obj]):
            if rel.suffix == ".mp4":
                media = (f"<video src='{rel}' controls loop muted preload='none' "
                         f"style='max-width:520px'></video>")
            else:  # gif or png -> plain img
                media = f"<img src='{rel}' loading='lazy'>"
            html.append(f"<div class='clip'><b>{rel.stem}</b><br>{media}</div>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))

    print(f"[viz] done: {len(rows)} filmstrips under {out_dir}/")
    print(f"[viz]   labels template -> {csv_path}")
    print(f"[viz]   contact sheet   -> {out_dir/'index.html'}  (open in a browser)")


if __name__ == "__main__":
    main()

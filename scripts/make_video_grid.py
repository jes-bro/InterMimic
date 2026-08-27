#!/usr/bin/env python3
"""Grid figure comparing videos frame-for-frame: one row per video, one column per frame.

Built for the camera-matched bball comparison -- source footage against the 4D
reconstruction replayed in Isaac Gym from the same viewpoint. Frame k of the
replay corresponds to frame k of the footage by construction (the replay is
recorded with SKIP_VIDEO_FRAMES=1, which drops the first captured image to
offset play_dataset_step's one-step write lag).

THE CROP IS SHARED, DELIBERATELY. The two videos are the same scene through the
same camera, so cropping them independently would destroy exactly the
correspondence the figure exists to show. One fractional window is measured on
the reference row -- the sim replay, where the neutral grey background makes the
subject easy to isolate -- and then applied to every row. Real footage has no
such clean key, which is why it is not measured there.

Frames are pulled with ffmpeg; imageio often has no video backend installed.

    python3 scripts/make_video_grid.py --dir ~/Downloads/bball_rendersaug27
    python3 scripts/make_video_grid.py --dir DIR --frames 5 30 50 70 --zoom 0.6
    python3 scripts/make_video_grid.py --dir DIR --with-policy
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from PIL import Image                     # noqa: E402


def extract_frame(video, index, out_png):
    """Pull one frame by index. Raises SystemExit rather than yielding a blank panel."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"select=eq(n\\,{index})", "-vsync", "0",
           "-frames:v", "1", "-y", str(out_png)]
    subprocess.run(cmd, check=True, capture_output=True)
    if not out_png.is_file() or out_png.stat().st_size == 0:
        raise SystemExit(f"ffmpeg produced no frame {index} from {video.name}")
    return Image.open(out_png).convert("RGB")


def subject_box(frames, sat_thr):
    """Union bounding box (fractional) of coloured pixels across frames.

    The sim background is neutral grey, so saturation isolates the humanoid.
    Union, not centroid: a window placed on the centroid clips the tallest pose.
    """
    x0 = y0 = 1.0
    x1 = y1 = 0.0
    found = False
    for img in frames:
        a = np.asarray(img, np.float32)
        m = (a.max(-1) - a.min(-1)) > sat_thr
        if not m.any():
            continue
        found = True
        yy, xx = np.nonzero(m)
        h, w = m.shape
        x0 = min(x0, xx.min() / w); x1 = max(x1, xx.max() / w)
        y0 = min(y0, yy.min() / h); y1 = max(y1, yy.max() / h)
    return (x0, y0, x1, y1), found


def crop_frac(img, box):
    """Crop to a fractional (left, top, right, bottom) box."""
    w, h = img.size
    return img.crop((int(box[0] * w), int(box[1] * h),
                     int(box[2] * w), int(box[3] * h)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="directory holding the mp4s")
    p.add_argument("--source", default="Date03_Sub01_bball_dribble.0.color.mp4")
    p.add_argument("--recon", default="replay_optj3d_cf_cam04.mp4")
    p.add_argument("--policy", default=None,
                   help="optional third row; --with-policy finds it automatically")
    p.add_argument("--with-policy", action="store_true",
                   help="add the newest policy_*.mp4 in --dir as a third row")
    p.add_argument("--frames", type=int, nargs="+", default=[8, 30, 50, 72],
                   help="frame indices; the clip is 101 frames (catch ~0-10, "
                        "carry ~27-49, takeoff ~49, flight ~50-58)")
    p.add_argument("--zoom", type=float, default=None,
                   help="0..1 fraction of the frame to keep around the subject; "
                        "omit for full frames. Applied identically to every row.")
    p.add_argument("--margin", type=float, default=0.35,
                   help="padding around the measured subject box when --zoom is set")
    p.add_argument("--crop-mode", choices=["shared", "per-row", "none"], default="shared",
                   help="shared: one box per column across all rows (same camera). "
                        "per-row: each row measured on itself (different cameras).")
    p.add_argument("--policy-offset", type=int, default=0,
                   help="frames to shift the policy row; it is recorded without "
                        "SKIP_VIDEO_FRAMES=1 so it can sit one frame off")
    p.add_argument("--aspect", type=float, default=4/3,
                   help="output width/height of each panel")
    p.add_argument("--sat-thr", type=float, default=20.0)
    p.add_argument("--title", default="4D reconstruction vs. source footage, same camera")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; it is how frames are read")

    rows = [("Source video", args.dir / args.source),
            ("4D reconstruction\n(Isaac Gym)", args.dir / args.recon)]
    policy = args.policy
    if args.with_policy and policy is None:
        found = sorted(args.dir.glob("policy_*.mp4"))
        if not found:
            raise SystemExit(f"--with-policy but no policy_*.mp4 in {args.dir}")
        policy = found[-1].name
    if policy:
        rows.append(("Policy rollout", args.dir / policy))

    for _, v in rows:
        if not v.is_file():
            raise SystemExit(f"no video at {v}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # The policy rollout is recorded WITHOUT SKIP_VIDEO_FRAMES=1, unlike the
        # replay, so it can sit one frame off the other two. --policy-offset
        # corrects it without disturbing rows that are already aligned.
        offsets = [0] * len(rows)
        if policy:
            offsets[-1] = args.policy_offset
        grid = []
        for i, (label, video) in enumerate(rows):
            imgs = [extract_frame(video, max(0, f + offsets[i]), tmp / f"r{i}_f{f}.png")
                    for f in args.frames]
            grid.append(imgs)

        # Crop layout depends on whether the rows share a camera.
        #
        #   shared  : one box per column, used by EVERY row. Correct when the
        #             rows are the same scene through the same camera -- cropping
        #             them independently would destroy the correspondence the
        #             figure exists to show.
        #   per-row : each row measured on itself. Correct when the rows come
        #             from DIFFERENT cameras (e.g. a policy rollout rendered from
        #             the default 3/4 view against a cam04 replay), where a shared
        #             box would put the subject off-panel in one of them.
        #   none    : full frames.
        boxes = [(0.0, 0.0, 1.0, 1.0)] * len(args.frames)
        per_row_boxes = None
        if args.zoom is not None and args.crop_mode == "per-row":
            per_row_boxes = []
            for i in range(len(rows)):
                col = []
                for j in range(len(args.frames)):
                    sub, found = subject_box([grid[i][j]], args.sat_thr)
                    if not found or (sub[2] - sub[0]) > 0.9:
                        # Real footage keys as saturated almost everywhere, so
                        # there is no subject to isolate -- fall back to the full
                        # frame rather than crop to a meaningless box.
                        col.append((0.0, 0.0, 1.0, 1.0))
                        continue
                    cx, cy = (sub[0] + sub[2]) / 2, (sub[1] + sub[3]) / 2
                    hw = max((sub[2] - sub[0]) * (1 + args.margin) / 2, args.zoom / 2)
                    hh = max((sub[3] - sub[1]) * (1 + args.margin) / 2, args.zoom / 2)
                    W, H = grid[i][j].size
                    hw = max(hw, hh * args.aspect * H / W)
                    hh = max(hh, hw * W / (args.aspect * H))
                    hw, hh = min(hw, 0.5), min(hh, 0.5)
                    cx = min(max(cx, hw), 1 - hw)
                    cy = min(max(cy, hh), 1 - hh)
                    col.append((cx - hw, cy - hh, cx + hw, cy + hh))
                per_row_boxes.append(col)
            print(f"per-row cropping ({len(rows)} rows measured independently "
                  f"-- correct for mixed camera views)")
        elif args.zoom is not None and args.crop_mode == "shared":
            per_col = []
            for j in range(len(args.frames)):
                # Measure on the RECON row: its neutral grey background keys
                # cleanly, where real footage does not.
                sub, found = subject_box([grid[1][j]], args.sat_thr)
                if not found:
                    per_col.append((0.0, 0.0, 1.0, 1.0))
                    continue
                cx, cy = (sub[0] + sub[2]) / 2, (sub[1] + sub[3]) / 2
                half_w = max((sub[2] - sub[0]) * (1 + args.margin) / 2, args.zoom / 2)
                half_h = max((sub[3] - sub[1]) * (1 + args.margin) / 2, args.zoom / 2)
                # Keep a constant output aspect so the panels tile evenly, then
                # slide the window back inside the frame rather than shrinking
                # it -- shrinking would silently re-crop the subject out.
                half_w = max(half_w, half_h * args.aspect * grid[1][j].size[1] / grid[1][j].size[0])
                half_h = max(half_h, half_w * grid[1][j].size[0] / (args.aspect * grid[1][j].size[1]))
                half_w, half_h = min(half_w, 0.5), min(half_h, 0.5)
                cx = min(max(cx, half_w), 1 - half_w)
                cy = min(max(cy, half_h), 1 - half_h)
                per_col.append((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
            boxes = per_col
            for j, (f, b) in enumerate(zip(args.frames, boxes)):
                print(f"  frame {f:3d}: crop {b[0]:.3f},{b[1]:.3f} -> {b[2]:.3f},{b[3]:.3f}")

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
        })
        fig, axes = plt.subplots(
            nrows=len(rows), ncols=len(args.frames),
            figsize=(len(args.frames) * 2.9, len(rows) * 2.0),
        )
        axes = np.atleast_2d(axes)

        for i, (label, _) in enumerate(rows):
            for j, fid in enumerate(args.frames):
                ax = axes[i][j]
                ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
                box = per_row_boxes[i][j] if per_row_boxes else boxes[j]
                ax.imshow(crop_frac(grid[i][j], box))
                if i == 0:
                    ax.set_title(f"t={fid/30:.2f}s   (frame {fid})", fontsize=10)
                if j == 0:
                    ax.set_ylabel(label, fontsize=11, rotation=0,
                                  ha="center", va="center", labelpad=52)

        plt.subplots_adjust(wspace=0.02, hspace=0.03, left=0.12, right=0.99,
                            top=0.88, bottom=0.01)
        plt.suptitle(args.title, fontsize=12, y=0.97)

        out = args.out
        if out is None:
            tag = "zoom" if args.zoom is not None else "full"
            out = args.dir / f"grid_recon_vs_source_{len(rows)}row_{tag}.png"
        plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Qualitative comparison grid, v3 -- one test subject, bigger figures, faded sim background.

Descends from ~/Downloads/make_grid_heldout{,_v2}.py. Three changes over v2:

  1. ONE test subject (default sub5). v2 stacked sub1 and sub5, five rows deep,
     which is what made each panel small.
  2. Tighter crop, so the humanoid fills the panel instead of the court.
  3. The grey sim floor/sky is faded toward a flat colour.

WHY A FADE AND NOT A CUTOUT. Measured on the v2 frames: 95% of pixels sit at
saturation <= 8 (the neutral floor and sky) and the coloured subject is the ~1.2%
above 20 -- a clean split. But the CHAIR is near-white, so it is low-saturation
too, and plate-differencing does not rescue it either (only 0.5% of pixels move
across a 20-frame window, i.e. the chair is mostly static). Any mask cheap enough
to be worth writing will misclassify the chair. A fade degrades gracefully there
-- a misclassified chair goes light grey but stays visible -- where a hard cutout
would delete it. `--bg-mode solid` is available if you want the hard version.

The mask is the UNION of "coloured" and "moved", so the chair is kept in the
frames where it does move.

DON'T TRUST MY EYE ON THIS -- run --sweep and pick. It writes one image per
background setting so you can compare them yourself; every filename records its
settings, so nothing overwrites anything.

    python3 scripts/make_grid_heldout_v3.py --sweep
    python3 scripts/make_grid_heldout_v3.py --bg-fade 0.8 --crop 0.30 0.40
    python3 scripts/make_grid_heldout_v3.py --subject sub1     # the other one
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OBJECT = "woodchair"
SOURCE = "sub6"
# v2's frames root: raw and uncropped, kept separate from the older ~/Downloads/fun
# renders so an rglob cannot pick up a stale run.
FRAMES_ROOT = Path("/home/jess/Downloads/fun/v2_ood/frames_woodchair")


def crop_center(img, w_frac, h_frac, x_shift_frac=0.10, y_shift_frac=0.08):
    """Crop a centred window, nudged right/down to re-centre a subject that
    sits right and low in these renders (the shifts are v2's, kept as-is)."""
    w, h = img.size
    new_w, new_h = int(w * w_frac), int(h * h_frac)
    left = (w - new_w) // 2 + int(w * x_shift_frac)
    top = (h - new_h) // 2 + int(h * y_shift_frac)
    left = max(0, min(left, w - new_w))          # clamp so we never run off the edge
    top = max(0, min(top, h - new_h))
    return img.crop((left, top, left + new_w, top + new_h))


def background_plate(frame_dir, n=20):
    """Per-pixel median over the first n frames == the static content.

    Returns None when there are too few frames to be meaningful, in which case
    the caller falls back to the saturation term alone.
    """
    frames = sorted(frame_dir.glob("frame_*.png"))[:n]
    if len(frames) < 5:
        return None
    stack = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) for f in frames])
    return np.median(stack, axis=0)


def fade_background(img, plate, sat_thr, move_thr, mode, fade, colour):
    """Fade or replace everything that is neither coloured nor moving.

    Args:
        img: PIL RGB frame, uncropped (so it lines up with `plate`).
        plate: the median background, or None to skip the motion term.
        sat_thr: max(RGB)-min(RGB) above which a pixel counts as subject.
        move_thr: per-channel difference from the plate that counts as motion.
        mode: 'fade' blends the background toward `colour`; 'solid' replaces it
            outright; 'none' returns the frame untouched.
        fade: 0..1, how far toward `colour` the background moves in 'fade' mode.
        colour: RGB triple the background is pushed toward.

    Returns:
        A PIL RGB frame.
    """
    if mode == "none":
        return img
    a = np.asarray(img.convert("RGB"), np.float32)

    keep = (a.max(-1) - a.min(-1)) > sat_thr                 # coloured => subject
    if plate is not None:
        keep |= np.abs(a - plate).max(-1) > move_thr         # moved => subject

    # Feather the mask so the fade does not leave a hard jagged outline around
    # the humanoid. A 1px box blur on the 0/1 mask is enough at this scale.
    m = keep.astype(np.float32)
    m = np.stack([np.roll(np.roll(m, dy, 0), dx, 1)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)]).max(0)
    m = np.clip(m, 0, 1)[..., None]

    bg = np.asarray(colour, np.float32)
    strength = 1.0 if mode == "solid" else float(fade)
    faded = a * (1 - strength) + bg * strength
    return Image.fromarray(np.clip(a * m + faded * (1 - m), 0, 255).astype(np.uint8))


def build(args, out_path):
    """Render one grid image with the given settings."""
    sub = args.subject
    n = sub[3:]
    # The source row is the SAME kinematic playback whatever target body the env
    # used, so v1/v2's sub1-named source dir is reused verbatim -- it is not a
    # leftover from the dropped sub1 rows.
    rows = [
        ("Source Motion",        f"render_sub1_{OBJECT}_source_{SOURCE}"),
        (f"Sub. {n} Ours",       f"render_{sub}_{OBJECT}_full_method"),
        (f"Sub. {n} InterMimic", f"render_{sub}_{OBJECT}_vanilla"),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    })

    fig, axes = plt.subplots(
        nrows=len(rows), ncols=len(args.frames),
        figsize=(len(args.frames) * 2.6, len(rows) * 2.0),
    )
    axes = np.atleast_2d(axes)

    for i, (label, dirname) in enumerate(rows):
        d = FRAMES_ROOT / dirname
        plate = background_plate(d) if d.is_dir() else None
        for j, fid in enumerate(args.frames):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

            img_path = d / f"frame_{fid:03d}.png"
            if not img_path.exists():
                # Loud, in the figure itself: a silently missing panel would read
                # as a policy that produced nothing.
                ax.text(0.5, 0.5, f"MISSING\n{dirname}\nframe {fid}",
                        ha="center", va="center", fontsize=6, color="crimson")
                continue

            img = fade_background(Image.open(img_path), plate, args.sat_thr,
                                  args.move_thr, args.bg_mode, args.bg_fade,
                                  args.bg_colour)
            ax.imshow(crop_center(img, *args.crop))
            if i == 0:
                ax.set_title(f"t={fid/30:.1f}s", fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=11, rotation=0,
                              ha="center", va="center", labelpad=62)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.13, right=0.99,
                        top=0.88, bottom=0.01)
    plt.suptitle(f"Performance on Object=Wood Chair, Source=Subject {SOURCE[3:]} "
                 f"with Test Subject {n}", fontsize=12, y=0.98)
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="sub5", help="the single test subject to show")
    p.add_argument("--frames", type=int, nargs="+", default=[15, 30, 45, 60],
                   help="frame indices; columns are labelled frame/30 seconds")
    p.add_argument("--crop", type=float, nargs=2, default=[0.30, 0.40],
                   metavar=("W_FRAC", "H_FRAC"),
                   help="fraction of width/height kept; smaller = bigger humanoid "
                        "(v2 used 0.42 0.52)")
    p.add_argument("--bg-mode", choices=["fade", "solid", "none"], default="fade")
    p.add_argument("--bg-fade", type=float, default=0.75,
                   help="0..1, how far the background moves toward --bg-colour")
    p.add_argument("--bg-colour", type=int, nargs=3, default=[255, 255, 255],
                   metavar=("R", "G", "B"))
    p.add_argument("--sat-thr", type=float, default=20.0,
                   help="max(RGB)-min(RGB) above which a pixel is subject; the "
                        "measured floor/sky sit at <=8")
    p.add_argument("--move-thr", type=float, default=25.0,
                   help="difference from the median plate that counts as motion")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--sweep", action="store_true",
                   help="write one image per background setting to compare by eye")
    args = p.parse_args()

    if not FRAMES_ROOT.is_dir():
        raise SystemExit(f"no frames at {FRAMES_ROOT}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"grid_v3_{OBJECT}_src{SOURCE[3:]}_{args.subject}"
    crop_tag = f"crop{args.crop[0]:.2f}x{args.crop[1]:.2f}"

    if not args.sweep:
        tag = f"{args.bg_mode}{args.bg_fade:.2f}" if args.bg_mode == "fade" else args.bg_mode
        build(args, args.out_dir / f"{stem}_{crop_tag}_bg-{tag}.png")
        return

    # Filenames carry their settings, so a sweep never overwrites a previous one
    # and you can tell the variants apart without opening them.
    for mode, fade in (("none", 0.0), ("fade", 0.5), ("fade", 0.75),
                       ("fade", 0.9), ("solid", 1.0)):
        args.bg_mode, args.bg_fade = mode, fade
        tag = f"{mode}{fade:.2f}" if mode == "fade" else mode
        build(args, args.out_dir / f"{stem}_{crop_tag}_bg-{tag}.png")


if __name__ == "__main__":
    main()

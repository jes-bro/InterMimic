#!/usr/bin/env python3
"""Crop rendered rollout frames to individual images -- bigger subject, no grid.

The grid script (make_grid_heldout_v3.py) crops with a FIXED offset that was
hand-tuned for one set of renders. Reused on a different set, that offset can
push the subject half out of frame -- and I cannot check that by looking.

So the crop window is centred on the subject by MEASUREMENT instead: the sim
background is neutral grey (measured: 95% of pixels at saturation <= 8) while
the humanoid is coloured, so the centroid of the saturated pixels locates the
subject. One window is computed per directory from every frame in it, which
keeps the framing steady across the sequence instead of jittering per frame.
--shift overrides with a manual offset if the automatic one is wrong.

Backgrounds are left ALONE by default; --bg-mode fade turns on the grid
script's fade if you want it.

    python3 scripts/crop_frames.py ~/Downloads/fun/frames_woodchair/render_sub1_woodchair_source_sub2
    python3 scripts/crop_frames.py DIR1 DIR2 --crop 0.26 0.36
    python3 scripts/crop_frames.py DIR --shift 0.10 0.08     # v2's manual offset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_grid_heldout_v3 import background_plate, fade_background  # noqa: E402


def subject_bbox(frame_dir, sat_thr):
    """Union bounding box of the coloured subject over a whole directory.

    The UNION across frames, not a per-frame box and not the centroid. A
    centroid is where the mass is, which for a crouching-then-standing sequence
    sits below the head -- so a window centred there clips the standing poses.
    The union is the smallest window that can hold every frame.

    Returns:
        (x0, y0, x1, y1) in 0..1, plus a flag that is False when nothing was
        coloured enough to locate (caller reports it rather than silently
        pretending the frame centre was measured).
    """
    x0 = y0 = 1.0
    x1 = y1 = 0.0
    found = False
    for f in sorted(frame_dir.glob("frame_*.png")):
        a = np.asarray(Image.open(f).convert("RGB"), np.float32)
        mask = (a.max(-1) - a.min(-1)) > sat_thr
        if not mask.any():
            continue
        found = True
        yy, xx = np.nonzero(mask)
        h, w = mask.shape
        x0 = min(x0, xx.min() / w); x1 = max(x1, xx.max() / w)
        y0 = min(y0, yy.min() / h); y1 = max(y1, yy.max() / h)
    if not found:
        return (0.25, 0.25, 0.75, 0.75), False
    return (x0, y0, x1, y1), True


def fit_window(boxes, img_size, margin, aspect):
    """Smallest window holding every box, padded, at the requested pixel aspect.

    One size is returned for ALL directories so the outputs stay pixel-identical
    and can be laid out together; each directory is still centred on its own box
    by the caller.

    Args:
        boxes: list of (x0, y0, x1, y1) fractional boxes.
        img_size: (W, H) of the source frames.
        margin: fractional padding added around the box, e.g. 0.15.
        aspect: target width/height of the OUTPUT in pixels, e.g. 4/3.

    Returns:
        (w_frac, h_frac), each clamped to 1.0.
    """
    W, H = img_size
    need_w = max(b[2] - b[0] for b in boxes) * (1 + margin)
    need_h = max(b[3] - b[1] for b in boxes) * (1 + margin)
    # Grow whichever dimension is short of the aspect; never shrink, or the
    # padding just measured would be given straight back.
    h_frac = max(need_h, need_w * W / (aspect * H))
    w_frac = max(need_w, aspect * h_frac * H / W)
    return min(w_frac, 1.0), min(h_frac, 1.0)


def crop_about(img, w_frac, h_frac, cx, cy):
    """Crop a w_frac x h_frac window centred on fractional point (cx, cy)."""
    w, h = img.size
    new_w, new_h = int(w * w_frac), int(h * h_frac)
    left = int(cx * w) - new_w // 2
    top = int(cy * h) - new_h // 2
    left = max(0, min(left, w - new_w))          # clamp, never run off the edge
    top = max(0, min(top, h - new_h))
    return img.crop((left, top, left + new_w, top + new_h))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", type=Path, nargs="+", help="frame directories to crop")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "Downloads/cropped_frames")
    p.add_argument("--crop", type=float, nargs=2, default=None,
                   metavar=("W_FRAC", "H_FRAC"),
                   help="fixed window size, overriding the measured fit; "
                        "smaller = bigger subject")
    p.add_argument("--margin", type=float, default=0.15,
                   help="padding around the measured subject box (0.15 = 15%%); "
                        "raise it to back the framing out further")
    p.add_argument("--aspect", type=float, default=4/3,
                   help="output width/height in pixels")
    p.add_argument("--shift", type=float, nargs=2, default=None,
                   metavar=("X_FRAC", "Y_FRAC"),
                   help="manual offset from centre, disabling the automatic "
                        "subject-centring (v2 used 0.10 0.08)")
    p.add_argument("--bg-mode", choices=["none", "fade", "solid"], default="none")
    p.add_argument("--bg-fade", type=float, default=0.75)
    p.add_argument("--bg-colour", type=int, nargs=3, default=[255, 255, 255])
    p.add_argument("--sat-thr", type=float, default=20.0,
                   help="max(RGB)-min(RGB) above which a pixel counts as subject")
    p.add_argument("--move-thr", type=float, default=25.0)
    args = p.parse_args()

    # Measure every directory first: the window size is shared across them, so
    # it cannot be chosen until the largest subject is known.
    plans = []
    for d in args.dirs:
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")
        frames = sorted(d.glob("frame_*.png"))
        if not frames:
            raise SystemExit(f"no frame_*.png in {d}")
        box, found = subject_bbox(d, args.sat_thr)
        if not found:
            print(f"WARNING {d.name}: no pixels above saturation {args.sat_thr} "
                  f"-- framing is a guess, not a measurement")
        plans.append((d, frames, box, found))

    img_size = Image.open(plans[0][1][0]).size
    if args.crop is not None:
        w_frac, h_frac = args.crop
        sizing = "fixed"
    else:
        w_frac, h_frac = fit_window([p[2] for p in plans], img_size,
                                    args.margin, args.aspect)
        sizing = f"fitted (margin {args.margin:.0%})"
    print(f"window {w_frac:.3f}x{h_frac:.3f} of {img_size[0]}x{img_size[1]} "
          f"-> {int(w_frac*img_size[0])}x{int(h_frac*img_size[1])} px, {sizing}")

    for d, frames, box, found in plans:
        if args.shift is not None:
            cx, cy = 0.5 + args.shift[0], 0.5 + args.shift[1]
            how = f"manual shift {args.shift[0]:+.2f},{args.shift[1]:+.2f}"
        else:
            # Centre of the union BOX, not the centroid: the box is what has to
            # fit, and its centre is the only point that keeps equal slack above
            # and below the subject's full range.
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            how = (f"centred on subject box y {box[1]:.3f}..{box[3]:.3f}"
                   if found else "frame centre (nothing measured)")

        plate = background_plate(d) if args.bg_mode != "none" else None
        out = args.out_dir / d.name
        out.mkdir(parents=True, exist_ok=True)

        for f in frames:
            img = Image.open(f)
            if args.bg_mode != "none":
                img = fade_background(img, plate, args.sat_thr, args.move_thr,
                                      args.bg_mode, args.bg_fade, args.bg_colour)
            crop_about(img, w_frac, h_frac, cx, cy).save(out / f.name)

        print(f"  {d.name}: {len(frames)} frames -> {out}  ({how}, bg {args.bg_mode})")


if __name__ == "__main__":
    main()

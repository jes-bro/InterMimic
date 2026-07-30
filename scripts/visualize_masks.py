#!/usr/bin/env python3
"""Render a SAM3 mask HDF5 over its video so you can SEE where tracking holds.

Matches CARI4D's own convention (prep/run_sam3_masks.py:245-261): side-by-side
raw | overlay, person in RED, object in BLUE, alpha 0.5. On top of that it adds
what you need to judge a take for trimming:

  * a per-frame HUD -- frame index, both mask areas, TRACKED / LOST
  * a timeline strip -- the whole take's tracked/lost pattern with a cursor, so
    a glance tells you where the usable runs are
  * --zoom -- a magnified inset around the object. A basketball is ~18x18 px in
    a 796x448 frame; at 1x you cannot see whether the mask is on the ball, on a
    shoe, or on nothing.

Streams frames through ffmpeg rather than loading the take (1922 frames is ~2 GB
as RGB), and needs no cv2 or imageio-ffmpeg -- just numpy, PIL and an ffmpeg
with libx264.

Usage:
    python3 scripts/visualize_masks.py \
        --masks ~/Downloads/sam3expertmasks/masks/cam04_masks_k0.h5 \
        --video ~/Downloads/egoexoexpert/cam04.mp4 \
        --out /tmp/vis.mp4 --zoom

    # just a stretch, to check a candidate trim
    python3 scripts/visualize_masks.py ... --start 354 --end 503
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trim_to_tracked import (  # noqa: E402
    PERSON, OBJECT, read_areas, find_ffmpeg, video_frame_count)

PERSON_RGB = np.array([255, 0, 0], dtype=np.float32)    # CARI4D: red
OBJECT_RGB = np.array([0, 0, 255], dtype=np.float32)    # CARI4D: blue
ALPHA = 0.5
STRIP_H = 18


def composite(frame, person, obj, alpha=ALPHA):
    """RGB frame with person/object masks blended in. Object drawn last so an
    overlap reads as the object -- that is the one whose tracking is in doubt."""
    out = frame.astype(np.float32)
    if person.any():
        out[person] = out[person] * (1 - alpha) + PERSON_RGB * alpha
    if obj.any():
        out[obj] = out[obj] * (1 - alpha) + OBJECT_RGB * alpha
    return out.clip(0, 255).astype(np.uint8)


def object_bbox(mask, pad=12):
    """(x0, y0, x1, y1) around the mask, padded. None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad))


def timeline_strip(good, width, height=STRIP_H, cursor=None):
    """Whole-take tracked/lost bar: green tracked, dark red lost, yellow cursor."""
    n = len(good)
    # Column i covers frames [i*n/width, (i+1)*n/width); a column is green only
    # if EVERY frame in it is tracked, so short dropouts stay visible.
    edges = (np.arange(width + 1) * n / width).astype(int)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        strip[:, i] = (60, 190, 90) if good[lo:hi].all() else (150, 30, 30)
    if cursor is not None and n > 0:
        c = min(width - 1, int(cursor * width / n))
        strip[:, max(0, c - 1):c + 2] = (255, 220, 0)
    return strip


def draw_hud(img, lines, xy=(6, 4), fg=(255, 255, 255), bg=(0, 0, 0)):
    """Text with a dark plate behind it, so it stays readable over any frame."""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for i, text in enumerate(lines):
        y = xy[1] + i * 13
        d.rectangle([xy[0] - 3, y - 2, xy[0] + 7 * len(text) + 3, y + 11], fill=bg)
        d.text((xy[0], y), text, fill=fg)
    return np.array(pil)


def zoom_inset(frame_ov, bbox, size=140):
    """Nearest-neighbour magnification of bbox, letterboxed to size x size."""
    x0, y0, x1, y1 = bbox
    crop = frame_ov[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    k = max(1, int(min(size / max(w, 1), size / max(h, 1))))
    big = np.repeat(np.repeat(crop, k, axis=0), k, axis=1)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    bh, bw = min(size, big.shape[0]), min(size, big.shape[1])
    out[:bh, :bw] = big[:bh, :bw]
    return out


def ffmpeg_reader(ffmpeg, path, W, H, start=0, count=None):
    """Yield RGB frames [start, start+count) by streaming raw video."""
    cmd = [ffmpeg, "-v", "error", "-i", path]
    if start or count:
        end = "" if count is None else f":end_frame={start + count}"
        cmd += ["-vf", f"trim=start_frame={start}{end},setpts=PTS-STARTPTS"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    nbytes = W * H * 3
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--masks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="first frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame (inclusive)")
    ap.add_argument("--zoom", action="store_true",
                    help="magnified inset around the object (use it: the ball is tiny)")
    ap.add_argument("--zoom-size", type=int, default=140)
    ap.add_argument("--min-person-px", type=int, default=1)
    ap.add_argument("--min-object-px", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args(argv)

    cam, frames, per, obj, (H, W) = read_areas(args.masks)
    good = (per >= args.min_person_px) & (obj >= args.min_object_px)

    nv = video_frame_count(args.video)
    if nv != len(frames):
        print(f"ERROR: video has {nv} frames, masks cover {len(frames)}. Refusing "
              f"to render a misaligned overlay.", file=sys.stderr)
        return 2

    lo = args.start
    hi = len(frames) - 1 if args.end is None else args.end
    if not (0 <= lo <= hi < len(frames)):
        print(f"ERROR: --start/--end out of range 0..{len(frames)-1}", file=sys.stderr)
        return 2
    n = hi - lo + 1

    ffmpeg = find_ffmpeg(args.ffmpeg)
    out_w, out_h = W * 2, H + STRIP_H
    print(f"[vis] {cam} frames {lo}-{hi} ({n}) -> {args.out}  {out_w}x{out_h}")
    print(f"[vis] tracked in this range: {int(good[lo:hi+1].sum())}/{n} "
          f"({100*good[lo:hi+1].mean():.1f}%)   ffmpeg={ffmpeg}")

    wcmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{out_w}x{out_h}", "-r", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", args.out]
    writer = subprocess.Popen(wcmd, stdin=subprocess.PIPE)

    import h5py
    written = 0
    with h5py.File(args.masks, "r") as f:
        g = f[cam]
        for k, frame in enumerate(ffmpeg_reader(ffmpeg, args.video, W, H, lo, n)):
            i = frames[lo + k]
            pm = g[f"{i:06d}-k0.{PERSON}"][()]
            om = g[f"{i:06d}-k0.{OBJECT}"][()]
            ov = composite(frame, pm, om)

            if args.zoom:
                bb = object_bbox(om)
                if bb is not None:
                    d = Image.fromarray(ov)
                    ImageDraw.Draw(d).rectangle(list(bb), outline=(0, 255, 255))
                    ov = np.array(d)
                    ins = zoom_inset(ov, bb, args.zoom_size)
                    if ins is not None:
                        ov[0:ins.shape[0], ov.shape[1]-ins.shape[1]:] = ins

            state = "TRACKED" if good[lo + k] else "LOST"
            ov = draw_hud(ov, [f"f{i}  {state}",
                               f"person {int(per[lo+k])}px",
                               f"object {int(obj[lo+k])}px"])
            top = np.concatenate([frame, ov], axis=1)
            strip = timeline_strip(good, out_w, STRIP_H, cursor=lo + k)
            writer.stdin.write(np.concatenate([top, strip], axis=0).tobytes())
            written += 1

    writer.stdin.close()
    writer.wait()
    if written != n:
        print(f"ERROR: wrote {written} frames but expected {n} -- the video and "
              f"the masks did not stay in step.", file=sys.stderr)
        return 1
    print(f"[vis] wrote {written} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Join a source video and its replay into one synchronized side-by-side clip.

Judging a reconstruction against footage by alternating between two windows is
unreliable -- a limb that lags by three frames, or an object that drifts only
during one phase, is invisible that way and obvious when the two play together.

    python scripts/make_side_by_side.py \\
        --left  <source>.mp4 \\
        --right renders/<replay>.mp4 \\
        --out   comparison.mp4

Frame k on the left is shown against frame k on the right, so the two must
already correspond. They do by construction when the replay was recorded with
SKIP_VIDEO_FRAMES=1: play_dataset_step's writes reach the simulator one step
late, so dropping the first captured image lines the replay back up with the
motion, and hence with the source frames the motion came from. --offset shifts
one against the other if that ever stops holding.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_frames(path: Path, limit: int = None) -> list:
    """Return a video's frames as a list of (H, W, 3) uint8 arrays.

    Raises:
        SystemExit: if the file is missing or holds no frames, rather than
            producing a half-width video with one side blank.
    """
    import imageio.v2 as imageio

    if not path.is_file():
        raise SystemExit(f"no video at {path}")
    frames = []
    reader = imageio.get_reader(str(path))
    try:
        for i, frame in enumerate(reader):
            if limit is not None and i >= limit:
                break
            frames.append(np.asarray(frame)[..., :3])
    finally:
        reader.close()
    if not frames:
        raise SystemExit(f"{path.name} contains no frames")
    return frames


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Scale a frame to the given height, preserving aspect ratio.

    Uses PIL when available for a smooth result, and nearest-neighbour indexing
    otherwise -- upscaling a 448p source blockily is worth more than refusing to
    build the comparison at all.
    """
    # Even width, because libx264 with yuv420p rejects odd dimensions and two
    # panels of odd width join into an odd total. Rounding here rather than
    # padding the joined frame keeps the seam exactly at the centre.
    width = int(round(frame.shape[1] * height / frame.shape[0]))
    width += width % 2
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(frame).resize((width, height),
                                                        Image.BILINEAR))
    except ImportError:
        rows = (np.arange(height) * frame.shape[0] // height).clip(0, frame.shape[0] - 1)
        cols = (np.arange(width) * frame.shape[1] // width).clip(0, frame.shape[1] - 1)
        return frame[rows][:, cols]


def label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw text in the frame's top-left corner, on a dark strip for legibility.

    Silently returns the frame unchanged if PIL is missing: a label is a
    convenience, and losing it should not cost the comparison.
    """
    if not text:
        return frame
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> int:
    """Build the joined video and report how the two sides were matched."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True,
                        help="source footage")
    parser.add_argument("--right", type=Path, required=True,
                        help="simulator replay")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0,
                        help="frames to advance the RIGHT video against the "
                             "left. Positive drops leading replay frames "
                             "(default: 0)")
    parser.add_argument("--height", type=int, default=None,
                        help="output height (default: the taller input, so the "
                             "sharper side is not thrown away)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-label", default="source video")
    parser.add_argument("--right-label", default="4D reconstruction (Isaac Gym)")
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    left = read_frames(args.left.expanduser().resolve())
    right = read_frames(args.right.expanduser().resolve())
    print(f"left  {args.left.name}: {len(left)} frames, "
          f"{left[0].shape[1]}x{left[0].shape[0]}")
    print(f"right {args.right.name}: {len(right)} frames, "
          f"{right[0].shape[1]}x{right[0].shape[0]}")

    if args.offset > 0:
        right = right[args.offset:]
        print(f"dropped {args.offset} leading replay frames")
    elif args.offset < 0:
        left = left[-args.offset:]
        print(f"dropped {-args.offset} leading source frames")

    n = min(len(left), len(right))
    if len(left) != len(right):
        # Said out loud: a silent truncation reads as "these clips correspond"
        # when one may simply be a different take.
        print(f"lengths differ by {abs(len(left) - len(right))} frames; "
              f"using the first {n}")
    if n == 0:
        raise SystemExit("no overlapping frames after the offset")

    height = args.height or max(left[0].shape[0], right[0].shape[0])
    height += height % 2          # same even-dimension requirement
    out_frames = []
    for i in range(n):
        l = resize_to_height(left[i], height)
        r = resize_to_height(right[i], height)
        if not args.no_labels:
            l = label(l, args.left_label)
            r = label(r, args.right_label)
        out_frames.append(np.concatenate([l, r], axis=1))

    import imageio.v2 as imageio
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 so the writer does not silently pad the width, which
    # would shift the seam between the two panels off centre.
    imageio.mimwrite(str(args.out), out_frames, fps=args.fps,
                     macro_block_size=1)
    h, w = out_frames[0].shape[:2]
    print(f"wrote {args.out} -- {n} frames at {w}x{h}, {args.fps:g} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())

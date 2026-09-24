#!/usr/bin/env python3
"""Build a filmstrip figure from rendered rollout videos: one row per clip.

WHY A SCRIPT. A qualitative figure gets rebuilt every time a better render turns
up, and the thing that makes it wrong is subtle: sampling frames across the WHOLE
video. A render is 400 frames covering four or five ATTEMPTS at a 70-100 frame
clip, so evenly spacing six frames over the whole file gives six different
attempts -- a strip where the object teleports and the pose jumps, which reads as
the method failing. The default here is the FIRST attempt only.

    python3 scripts/make_qual_figure.py --out fig.png \\
        "Basketball=renders/body-sub10__src-sub402_rev030c__act100k.mp4:0-71" \\
        "Soccer=renders/body-sub16__src-sub496_soccer...mp4:0-90"

Each argument is "<label>=<path>" with an optional ":<start>-<end>" frame range.
With no range the first ATTEMPT_FRAC of the video is used (default 0.25), which
is roughly one attempt for a 400-frame render of a 70-100 frame clip. Pass the
clip's own frame count when you know it -- the render log prints it
("<clip>.pt: 70 frames").

Frames are sampled evenly across the range INCLUSIVE of both ends, so the first
column is the start of the attempt and the last is its end.

    --frames N     columns per row (default 6)
    --height H     row height in px (default 220); width follows the aspect
    --crop W:H:X:Y crop every frame before scaling, e.g. to cut dead floor
    --no-labels    omit the row labels (for a figure that labels in LaTeX)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ATTEMPT_FRAC = 0.25


def probe(path):
    """-> (n_frames, fps, width, height). Raises SystemExit if unreadable."""
    if not os.path.isfile(path):
        raise SystemExit(f"no video at {path}")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"ffprobe failed on {path}: {out.stderr.strip()}")
    w, h, rate, n = out.stdout.strip().split(",")[:4]
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    return int(n), fps, int(w), int(h)


def parse_spec(spec, default_frac=ATTEMPT_FRAC):
    """'Label=path.mp4:0-71' -> (label, path, start, end|None).

    The range is optional and split off the RIGHT, so a path containing '=' or
    ':' (a Windows-ish or timestamped name) still parses.
    """
    if "=" not in spec:
        raise SystemExit(f"spec '{spec}': want <label>=<path>[:<start>-<end>]")
    label, rest = spec.split("=", 1)
    start, end = 0, None
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if "-" in tail:
            a, b = tail.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                rest, start, end = head, int(a), int(b)
    if not label.strip():
        raise SystemExit(f"spec '{spec}': empty label")
    return label.strip(), rest, start, end


def frame_indices(n_total, start, end, count, frac=ATTEMPT_FRAC):
    """Evenly spaced frame numbers across [start, end], both ends included."""
    if end is None:
        end = max(start, int(n_total * frac) - 1)
    end = min(end, n_total - 1)
    if end < start:
        raise SystemExit(f"frame range {start}-{end} is empty (video has {n_total})")
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [int(round(start + i * step)) for i in range(count)]


def extract(path, idxs, fps, outdir, crop=None):
    """Pull the given frame numbers out as PNGs. -> [paths], in order."""
    files = []
    for i, n in enumerate(idxs):
        dst = os.path.join(outdir, f"f{i:03d}.png")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{n / fps:.4f}", "-i", path]
        if crop:
            cmd += ["-vf", f"crop={crop}"]
        cmd += ["-frames:v", "1", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(dst):
            raise SystemExit(f"ffmpeg could not read frame {n} of {path}: {r.stderr.strip()}")
        files.append(dst)
    return files


def compose(rows, out, height, gap=4, label_w=150, labels=True, dpi=200):
    """rows: [(label, [frame paths])] -> one figure, one row per clip.

    matplotlib rather than hand-pasting tiles: the row labels sit outside the
    images (so nothing is drawn over the humanoid), the column spacing is uniform
    without arithmetic, and the extension picks the format -- .pdf for LaTeX,
    .png for a slide or a message.
    """
    import matplotlib
    matplotlib.use("Agg")                      # no display on a login shell
    import matplotlib.pyplot as plt

    if not rows:
        raise SystemExit("nothing to compose")
    n_rows = len(rows)
    n_cols = max(len(f) for _, f in rows)
    im0 = Image.open(rows[0][1][0])
    aspect = im0.width / im0.height

    # Figure size in inches from the requested row height in px, so --height
    # keeps meaning "how tall is one frame" whichever backend draws it.
    fig_w = n_cols * height * aspect / dpi + (label_w / dpi if labels else 0)
    fig_h = n_rows * height / dpi
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi,
                             squeeze=False)
    fig.subplots_adjust(left=(label_w / dpi) / fig_w if labels else 0,
                        right=1, top=1, bottom=0,
                        wspace=gap / height, hspace=gap / height)

    for r, (label, files) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c < len(files):
                ax.imshow(Image.open(files[c]).convert("RGB"))
            if c == 0 and labels:
                # Outside the axes, vertically centred on the row.
                ax.text(-0.04, 0.5, label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=max(7, height / 22),
                        fontweight="bold")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return int(fig_w * dpi), int(fig_h * dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="<label>=<video>[:<start>-<end>]")
    ap.add_argument("--out", default="qual_figure.png",
                    help="the extension picks the format: .pdf for LaTeX, .png otherwise")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--height", type=int, default=220)
    ap.add_argument("--crop", default=None, help="W:H:X:Y, applied before scaling")
    ap.add_argument("--frac", type=float, default=ATTEMPT_FRAC,
                    help="fraction of the video treated as one attempt when a "
                         "spec gives no explicit range (default 0.25)")
    ap.add_argument("--no-labels", action="store_true")
    a = ap.parse_args()
    if a.frames < 1:
        raise SystemExit("--frames must be at least 1")

    tmp = tempfile.mkdtemp(prefix="qualfig_")
    try:
        rows = []
        for i, spec in enumerate(a.specs):
            label, path, start, end = parse_spec(spec)
            n, fps, w, h = probe(path)
            idxs = frame_indices(n, start, end, a.frames, a.frac)
            print(f"  {label}: {os.path.basename(path)}  {n} frames @ {fps:g} fps "
                  f"-> sampling {idxs}")
            d = os.path.join(tmp, f"row{i}")
            os.makedirs(d)
            rows.append((label, extract(path, idxs, fps, d, a.crop)))
        size = compose(rows, a.out, a.height, labels=not a.no_labels)
        print(f"\nwrote {a.out}  ({size[0]}x{size[1]} px, {len(rows)} rows x {a.frames})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

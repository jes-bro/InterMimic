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
    # Multi-line labels: a qualitative figure has to say BOTH what the task is
    # and which held-out body is doing it -- "Soccer" alone does not show
    # zero-shot transfer to an unseen embodiment. '|' or a literal '\n' in the
    # label starts a new line ('|' because a real newline is awkward to type in
    # a shell argument).
    label = label.strip().replace("\\n", "\n").replace("|", "\n")
    return label, rest, start, end


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


def _load(path, bg="white"):
    """Open a frame, flattening any alpha onto bg.

    Photoshop hands back transparent PNGs when the background is cut out;
    .convert('RGB') on those makes the transparent region BLACK, which looks
    deliberate and is not.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        flat = Image.new("RGBA", im.size, bg)
        return Image.alpha_composite(flat, im).convert("RGB")
    return im.convert("RGB")


def motion_bbox(paths, thresh=12, pad=0.10, min_frac=1e-4):
    """One crop box per ROW, covering wherever anything moved across its frames.

    THE CAMERA MUST STAY STILL. A per-frame box that follows the humanoid would
    re-centre him in every panel, which reads as a tracking shot and destroys the
    thing a filmstrip is for -- seeing him move THROUGH the scene. So the box is
    the union over the row's frames, applied identically to each one.

    The camera is static and the backdrop is a fixed gradient, so the pixels that
    change between frames are the humanoid and the object. The per-pixel maximum
    deviation from the median frame is thresholded; the union of that mask is the
    box, padded by `pad` of its size.

    Returns (x, y, w, h), or None when almost nothing moved (a near-static clip
    like a CPR compression), in which case the caller keeps the full frame rather
    than cropping to a speck.
    """
    import numpy as np

    arrs = [np.asarray(_load(p).convert("L"), dtype=np.float32) for p in paths]
    if len(arrs) < 2:
        return None
    stack = np.stack(arrs)
    med = np.median(stack, axis=0)
    diff = np.abs(stack - med).max(axis=0)
    mask = diff > thresh
    if mask.sum() < min_frac * mask.size:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    px, py = int(w * pad), int(h * pad)
    H, W = mask.shape
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(W - 1, x1 + px); y1 = min(H - 1, y1 + py)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def fit_aspect(box, aspect, W, H):
    """Grow a box to the given w/h aspect, staying inside the frame.

    Every row is scaled to the same height in the figure, so rows with different
    aspects would come out different widths and the grid would not line up.
    """
    x, y, w, h = box
    if w / h < aspect:
        new_w = min(W, int(round(h * aspect)))
        x = max(0, min(W - new_w, x - (new_w - w) // 2))
        w = new_w
    else:
        new_h = min(H, int(round(w / aspect)))
        y = max(0, min(H - new_h, y - (new_h - h) // 2))
        h = new_h
    return x, y, w, h


def _slug(label):
    """'Basketball\\nsub10 (held out)' -> 'basketball_sub10_held_out'."""
    keep = [c.lower() if c.isalnum() else "_" for c in label]
    return "".join(keep).strip("_").replace("__", "_") or "row"


def frames_from_dir(path):
    """A row can be a DIRECTORY of stills instead of a video.

    That is the Photoshop round-trip: --frames-dir writes the sampled frames out,
    you cut the background in an editor, and the cleaned PNGs are handed straight
    back as a row. Files are taken in sorted order, which is why the dump names
    them f000, f001, ... -- an editor that appends '_edited' keeps that order.
    """
    files = sorted(os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"no .png/.jpg frames in {path}")
    return files


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


# Times, by preference, falling back through the metric-compatible clones. The
# figure sits next to LaTeX body text, so a serif that matches the paper reads as
# part of it rather than as a screenshot. Nimbus Roman is URW's Times clone and
# is metrically identical; Liberation Serif is the same idea from another foundry.
SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral",
         "DejaVu Serif"]


def compose(rows, out, height, gap=4, label_w=150, labels=True, dpi=200,
            title=None, bg="white"):
    """rows: [(label, [frame paths])] -> one figure, one row per clip.

    matplotlib rather than hand-pasting tiles: the row labels sit outside the
    images (so nothing is drawn over the humanoid), the column spacing is uniform
    without arithmetic, and the extension picks the format -- .pdf for LaTeX,
    .png for a slide or a message.
    """
    import matplotlib
    matplotlib.use("Agg")                      # no display on a login shell
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "text.color": "#111111",
    })

    if not rows:
        raise SystemExit("nothing to compose")
    n_rows = len(rows)
    n_cols = max(len(f) for _, f in rows)
    im0 = _load(rows[0][1][0])
    aspect = im0.width / im0.height

    # Figure size in inches from the requested row height in px, so --height
    # keeps meaning "how tall is one frame" whichever backend draws it.
    title_h = (height * 0.45 / dpi) if title else 0.0
    fig_w = n_cols * height * aspect / dpi + (label_w / dpi if labels else 0)
    fig_h = n_rows * height / dpi + title_h
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi,
                             squeeze=False)
    fig.subplots_adjust(left=(label_w / dpi) / fig_w if labels else 0,
                        right=1, top=1 - (title_h / fig_h), bottom=0,
                        wspace=gap / height, hspace=gap / height)
    if title:
        fig.suptitle(title, y=1.0, va="top", fontsize=max(9, height / 16))

    for r, (label, files) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c < len(files):
                ax.imshow(_load(files[c], bg))
            if c == 0 and labels:
                # Outside the axes, vertically centred on the row.
                # First line is the task, the rest identify the body. The body
                # line is the point of the figure, so it is legible but quieter
                # than the task -- two weights rather than two sizes.
                head, _, tail = label.partition("\n")
                ax.text(-0.04, 0.5, head, transform=ax.transAxes,
                        ha="right", va=("bottom" if tail else "center"),
                        fontsize=max(8, height / 18), fontweight="bold")
                if tail:
                    ax.text(-0.04, 0.5, tail, transform=ax.transAxes,
                            ha="right", va="top", fontsize=max(7, height / 22),
                            color="#444444", linespacing=1.15)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, facecolor="white")
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
    ap.add_argument("--crop", default=None,
                    help="fixed ffmpeg crop W:H:X:Y, applied at extraction. "
                         "Disables the automatic crop.")
    ap.add_argument("--no-auto-crop", dest="auto_crop", action="store_false",
                    help="keep the full 1280x720 frame")
    ap.add_argument("--motion-thresh", dest="motion_thresh", type=float, default=12,
                    help="grey-level change counted as motion (default 12)")
    ap.add_argument("--min-crop", dest="min_crop", type=int, default=420,
                    help="smallest crop height in px (default 420). A clip where "
                         "the subject barely moves would otherwise crop to a "
                         "speck and be upscaled into mush.")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="padding around the motion box, as a fraction (default 0.10)")
    ap.add_argument("--frac", type=float, default=ATTEMPT_FRAC,
                    help="fraction of the video treated as one attempt when a "
                         "spec gives no explicit range (default 0.25)")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--frames-dir", default=None,
                    help="keep the sampled frames here as PNGs instead of "
                         "discarding them -- edit the background out in Photoshop, "
                         "then rebuild passing the directory in place of the video")
    ap.add_argument("--title", default=None,
                    help="caption across the top, e.g. \"Zero-shot generalization "
                         "to unseen embodiments across tasks in CrossMimic4D\"")
    a = ap.parse_args()
    if a.frames < 1:
        raise SystemExit("--frames must be at least 1")

    tmp = tempfile.mkdtemp(prefix="qualfig_")
    try:
        rows = []
        for i, spec in enumerate(a.specs):
            label, path, start, end = parse_spec(spec)
            one_line = label.replace("\n", " / ")
            if os.path.isdir(path):
                files = frames_from_dir(path)
                print(f"  {one_line}: {len(files)} still(s) from {path}")
                rows.append((label, files))
                continue
            n, fps, w, h = probe(path)
            idxs = frame_indices(n, start, end, a.frames, a.frac)
            print(f"  {one_line}: {os.path.basename(path)}  {n} frames @ {fps:g} fps "
                  f"-> sampling {idxs}")
            d = os.path.join(a.frames_dir, f"row{i}_{_slug(label)}") if a.frames_dir \
                else os.path.join(tmp, f"row{i}")
            os.makedirs(d, exist_ok=True)
            rows.append((label, extract(path, idxs, fps, d, a.crop)))
        # Auto-crop: one static box per row, from where things moved. Skipped
        # when --crop was given (ffmpeg already cropped) or --no-auto-crop.
        if a.auto_crop and not a.crop:
            boxes = {}
            for i, (label, files) in enumerate(rows):
                b = motion_bbox(files, thresh=a.motion_thresh, pad=a.pad)
                if b is None:
                    print(f"  {label.splitlines()[0]}: little motion -- keeping the full frame")
                else:
                    boxes[i] = b
            if boxes:
                # ONE crop SIZE for every row, centred on each row's own motion.
                #
                # Not just a common aspect: rows are scaled to the same height in
                # the figure, so a row cropped to 157x177 would be blown up 1.5x
                # while a 585x659 row is shrunk -- the person would appear at
                # wildly different sizes and one row would be visibly soft. Same
                # box size means same magnification and same sharpness, and the
                # bodies stay comparable, which is the whole point when the rows
                # differ only by embodiment.
                im0 = _load(rows[0][1][0])
                W0, H0 = im0.width, im0.height
                aspect = sorted(w / h for _, _, w, h in boxes.values())[len(boxes) // 2]
                fitted = {i: fit_aspect(b, aspect, W0, H0) for i, b in boxes.items()}
                bw = min(W0, max(w for _, _, w, _ in fitted.values()))
                bh = min(H0, max(h for _, _, _, h in fitted.values()))
                bw = max(bw, int(a.min_crop * aspect))
                bh = max(bh, a.min_crop)
                bw, bh = min(bw, W0), min(bh, H0)
                for i, (label, files) in enumerate(rows):
                    if i not in fitted:
                        continue
                    fx, fy, fw, fh = fitted[i]
                    cx, cy = fx + fw / 2, fy + fh / 2
                    x = int(max(0, min(W0 - bw, cx - bw / 2)))
                    y = int(max(0, min(H0 - bh, cy - bh / 2)))
                    w, h = bw, bh
                    print(f"  {label.splitlines()[0]}: crop {w}x{h}+{x}+{y}")
                    out_files = []
                    for f in files:
                        c = _load(f).crop((x, y, x + w, y + h))
                        dst = f[:-4] + "_crop.png"
                        c.save(dst)
                        out_files.append(dst)
                    rows[i] = (label, out_files)
        if a.frames_dir:
            print(f"\nframes kept in {a.frames_dir} -- edit them and rebuild by "
                  f"passing the DIRECTORY in place of the video")
        size = compose(rows, a.out, a.height, labels=not a.no_labels,
                       title=a.title)
        print(f"\nwrote {a.out}  ({size[0]}x{size[1]} px, {len(rows)} rows x {a.frames})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

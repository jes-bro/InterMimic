#!/usr/bin/env python3
"""Trim a video + its SAM3 mask HDF5 to the longest run of frames where BOTH the
person and the object are tracked.

Why: SAM3 loses small fast objects (a basketball) for most of a take, but CARI4D
only needs a stretch where person AND object are visible. Rather than throw the
take away, cut it down to the best contiguous stretch.

A lost track is an EMPTY mask, not a missing dataset -- every frame has both
datasets, and the object's is all-zero when SAM3 dropped it. So "tracked" means
"mask area >= threshold", not "dataset present".

Input HDF5 layout (from prep/run_sam3_masks.py):
    <cam>/<frame:06d>-k0.person_mask.png      bool (H, W)
    <cam>/<frame:06d>-k0.obj_rend_mask.png    bool (H, W)

Outputs into --out-dir:
    <cam>_masks_k0.h5    same layout, frames RENUMBERED from 000000
    <cam>.mp4            frame-exact trim of the source video
    trim_manifest.json   source frame range, so the cut is traceable

Fail-loud by design: if the video's frame count does not match the mask count,
this refuses to run. A silent off-by-N would misalign every mask against its
frame and quietly poison the reconstruction.

Usage:
    # report the runs, write nothing
    python3 scripts/trim_to_tracked.py --masks .../cam04_masks_k0.h5 --list

    # cut to the longest run
    python3 scripts/trim_to_tracked.py \
        --masks ~/Downloads/sam3expertmasks/masks/cam04_masks_k0.h5 \
        --video ~/Downloads/egoexoexpert/cam04.mp4 \
        --out-dir ~/Downloads/expert_trimmed

    # bridge dropouts of up to 3 frames, and take the 2nd-longest run
    python3 scripts/trim_to_tracked.py ... --gap-tolerance 3 --rank 2
"""

import argparse
import json
import os
import re
import subprocess
import sys

import h5py
import numpy as np

PERSON = "person_mask.png"
OBJECT = "obj_rend_mask.png"
FRAME_RE = re.compile(r"(\d+)-k(\d+)\.")


def read_areas(h5_path):
    """Return (cam, frames, person_area, object_area, (H, W))."""
    with h5py.File(h5_path, "r") as f:
        cams = list(f.keys())
        if len(cams) != 1:
            raise ValueError(f"expected exactly one camera group, got {cams}")
        cam = cams[0]
        g = f[cam]
        frames = sorted({int(FRAME_RE.match(k).group(1)) for k in g.keys()
                         if FRAME_RE.match(k)})
        if not frames:
            raise ValueError(f"no '<frame>-k<n>.*' datasets under {cam}/")
        # Frame numbering must be dense: a hole would shift every later mask.
        if frames != list(range(frames[0], frames[-1] + 1)):
            missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
            raise ValueError(f"frame numbering has holes (e.g. {missing[:5]}); "
                             f"refusing to guess how they align to the video")
        shape = g[f"{frames[0]:06d}-k0.{PERSON}"].shape
        per = np.array([g[f"{i:06d}-k0.{PERSON}"][()].sum() for i in frames])
        obj = np.array([g[f"{i:06d}-k0.{OBJECT}"][()].sum() for i in frames])
    return cam, np.array(frames), per, obj, shape


def find_runs(good, gap_tolerance=0):
    """Contiguous runs of True in `good`, as (start_idx, end_idx) inclusive.

    gap_tolerance bridges short False stretches: a dropout of <= that many
    frames does not split a run. The bridged frames stay in the output (their
    masks are empty), which is the caller's problem to accept -- that is why the
    default is 0.
    """
    n = len(good)
    runs, start, gap = [], None, 0
    for i in range(n):
        if good[i]:
            if start is None:
                start, gap = i, 0
            else:
                gap = 0
            last_good = i
        else:
            if start is not None:
                gap += 1
                if gap > gap_tolerance:
                    runs.append((start, last_good))
                    start = None
    if start is not None:
        runs.append((start, last_good))
    return runs


def trim_masks(src_h5, dst_h5, cam, frames, lo, hi):
    """Copy frames[lo..hi] into a new HDF5, renumbered from 0."""
    with h5py.File(src_h5, "r") as fin, h5py.File(dst_h5, "w") as fout:
        gin, gout = fin[cam], fout.create_group(cam)
        for new_i, src_i in enumerate(range(frames[lo], frames[hi] + 1)):
            for kind in (PERSON, OBJECT):
                data = gin[f"{src_i:06d}-k0.{kind}"][()]
                gout.create_dataset(f"{new_i:06d}-k0.{kind}", data=data,
                                    compression="gzip")
    return hi - lo + 1


def find_ffmpeg(explicit=None):
    """First ffmpeg on the system that can actually encode h264.

    Not all builds ship libx264 -- on this machine /usr/local/bin/ffmpeg (which
    wins on PATH) does not, while /usr/bin/ffmpeg does. Picking blind gives a
    bare 'Unrecognized option crf' three steps into a long job, so probe first
    and say which one got used.
    """
    if explicit:
        cands = [explicit]
    else:
        cands = [c for c in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg")]
    for c in cands:
        try:
            out = subprocess.run([c, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if "libx264" in out.stdout:
            return c
    raise RuntimeError(
        f"no ffmpeg with libx264 found (tried: {', '.join(cands)}). "
        f"Pass --ffmpeg /path/to/ffmpeg. Encoding without h264 would produce a "
        f"file the downstream pipeline cannot read.")


def trim_video(src, dst, lo_frame, hi_frame, ffmpeg="ffmpeg"):
    """Frame-exact cut of [lo_frame, hi_frame] inclusive, re-encoded."""
    vf = (f"trim=start_frame={lo_frame}:end_frame={hi_frame + 1},"
          f"setpts=PTS-STARTPTS")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vf", vf,
           "-an", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", dst]
    subprocess.run(cmd, check=True)


def video_frame_count(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path])
    return int(out.decode().strip().split(",")[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--masks", required=True, help="SAM3 masks .h5")
    ap.add_argument("--video", help="source video (required unless --list)")
    ap.add_argument("--out-dir", help="output dir (required unless --list)")
    ap.add_argument("--min-person-px", type=int, default=1,
                    help="min person mask area to count as tracked (default 1)")
    ap.add_argument("--min-object-px", type=int, default=1,
                    help="min object mask area to count as tracked (default 1). "
                         "Raise it to reject a few-pixel spurious blob.")
    ap.add_argument("--gap-tolerance", type=int, default=0,
                    help="bridge dropouts of up to N frames (default 0)")
    ap.add_argument("--rank", type=int, default=1,
                    help="take the Nth longest run (1 = longest)")
    ap.add_argument("--fps", type=float, default=30.0, help="for reporting only")
    ap.add_argument("--ffmpeg", default=None,
                    help="ffmpeg binary to use (default: first one found that "
                         "supports libx264)")
    ap.add_argument("--list", action="store_true",
                    help="report the runs and exit without writing")
    args = ap.parse_args(argv)

    cam, frames, per, obj, shape = read_areas(args.masks)
    good = (per >= args.min_person_px) & (obj >= args.min_object_px)
    runs = find_runs(good, args.gap_tolerance)
    runs.sort(key=lambda r: -(r[1] - r[0] + 1))

    print(f"[trim] {cam}: {len(frames)} frames {shape[1]}x{shape[0]}, "
          f"{int(good.sum())} tracked ({100.0*good.mean():.1f}%)")
    print(f"[trim] {len(runs)} contiguous run(s) "
          f"(min_person={args.min_person_px}px min_object={args.min_object_px}px "
          f"gap_tolerance={args.gap_tolerance})")
    for r, (lo, hi) in enumerate(runs[:10], start=1):
        n = hi - lo + 1
        mark = "  <-- selected" if r == args.rank else ""
        print(f"    #{r:<2} frames {frames[lo]:>6}-{frames[hi]:<6} "
              f"{n:>5} frames  {n/args.fps:>6.1f}s  "
              f"median object {np.median(obj[lo:hi+1]):.0f}px{mark}")

    if args.list:
        return 0
    if not args.video or not args.out_dir:
        print("ERROR: --video and --out-dir are required unless --list",
              file=sys.stderr)
        return 2
    if not runs:
        print("ERROR: no frames have both masks; nothing to trim", file=sys.stderr)
        return 1
    if args.rank > len(runs):
        print(f"ERROR: --rank {args.rank} but only {len(runs)} runs exist",
              file=sys.stderr)
        return 2

    # No silent misalignment: the masks are indexed by video frame number, so a
    # count mismatch means every mask would be paired with the wrong frame.
    nv = video_frame_count(args.video)
    if nv != len(frames):
        print(f"ERROR: video has {nv} frames but masks cover {len(frames)}. "
              f"Refusing to trim -- masks would be misaligned. Check that "
              f"{os.path.basename(args.video)} is the video the masks were "
              f"computed from.", file=sys.stderr)
        return 2

    lo, hi = runs[args.rank - 1]
    os.makedirs(args.out_dir, exist_ok=True)
    dst_h5 = os.path.join(args.out_dir, f"{cam}_masks_k0.h5")
    dst_mp4 = os.path.join(args.out_dir, f"{cam}.mp4")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"[trim] using ffmpeg: {ffmpeg}")
    n = trim_masks(args.masks, dst_h5, cam, frames, lo, hi)
    trim_video(args.video, dst_mp4, int(frames[lo]), int(frames[hi]), ffmpeg=ffmpeg)

    nv_out = video_frame_count(dst_mp4)
    manifest = dict(source_video=os.path.abspath(args.video),
                    source_masks=os.path.abspath(args.masks),
                    camera=cam, source_frame_start=int(frames[lo]),
                    source_frame_end=int(frames[hi]), n_frames=int(n),
                    duration_sec=round(n / args.fps, 3),
                    min_person_px=args.min_person_px,
                    min_object_px=args.min_object_px,
                    gap_tolerance=args.gap_tolerance, rank=args.rank,
                    output_video_frames=int(nv_out))
    with open(os.path.join(args.out_dir, "trim_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[trim] wrote {n} frames ({n/args.fps:.1f}s) from source "
          f"{frames[lo]}-{frames[hi]} -> {args.out_dir}/")
    if nv_out != n:
        print(f"[trim] WARNING: trimmed video has {nv_out} frames but {n} masks "
              f"were written -- do not use this output until that is explained.",
              file=sys.stderr)
        return 1
    print(f"[trim] verified: trimmed video and mask count agree ({n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

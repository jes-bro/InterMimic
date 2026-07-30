"""Overlay training-reward curves from rl_games TensorBoard event files.

Each training run writes events.out.tfevents.* under
<train_dir>/<experiment>/summaries/ (see learning/a2c_common.py). This script
reads one or more such run directories and overlays their reward trajectories
so A/B experiments (e.g. normalize_value on/off) can be compared directly.

Two panels, same y-axis (mean episode reward, raw env reward -- rl_games tag
'rewards0'):
  left  : reward vs environment frames  (sample efficiency -- the fair x-axis
          when runs differ in speed or env count)
  right : reward vs wall-clock hours    (what you actually waited)

Raw per-iteration values are noisy (one point per epoch), so each curve is
drawn twice: the raw trace at low opacity and an exponential moving average
(EMA, same smoothing TensorBoard's slider applies) on top.

Usage:
    python scripts/plot_tb_rewards.py \
        --run "normalize_value=~/Downloads/normval_ab_results/normval" \
        --run "from scratch=~/Downloads/normval_ab_results/no_normval" \
        --out ~/Downloads/normval_ab_results/normval_ab_rewards.png

Each --run is LABEL=DIR where DIR contains the event file (the summaries/ dir
itself, or any dir holding a single run's events). Missing tags or empty event
files are an error, not a silent skip.
"""

import argparse
import glob
import os
import sys

import numpy as np

# Force a non-interactive backend before pyplot import so the script works
# headless (ssh / slurm) exactly like it does locally.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Chart chrome (light surface) -- palette per the repo's plotting convention:
# series hues assigned in fixed order, text/grid in neutral ink, not series color.
# Six hues so a 6-arm comparison never reuses a colour (curves are also direct-
# labelled at their endpoints, but repeated colours still read as "same run").
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#8a5cf0", "#c2325f"]  # blue, orange, aqua, yellow, violet, crimson
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def load_scalar(run_dir, tag):
    """Return (steps, wall_times, values) arrays for one scalar tag.

    Raises (rather than returning empty) if the dir has no events or the tag
    is absent -- a wrong path should fail loudly, not plot an empty line.
    """
    run_dir = os.path.expanduser(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir does not exist: {run_dir}")

    # A requeued run writes ONE event file per job, so a run dir can hold several.
    # They are read separately rather than letting EventAccumulator merge a whole
    # directory, because two things need handling that it does not do:
    #   * a job that died before writing scalars leaves an EMPTY file (seen: 3 of
    #     the 4 files in the 'none' dir). Merging them is harmless; erroring on
    #     them is not, so they are skipped -- but a run with NO usable file still
    #     raises.
    #   * resuming restarts from the last checkpoint, so each segment REPLAYS the
    #     epochs between that checkpoint and the crash (e.g. 1205-1268). Those
    #     steps appear twice with different values; concatenating gives a
    #     non-monotonic x-axis and a curve that doubles back on itself.
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*")))
    if not files:
        raise FileNotFoundError(f"no event files under {run_dir}")

    # MERGE KEY IS EPOCH, NOT FRAMES. On resume the epoch counter is restored from
    # the checkpoint and continues (5->349, 305->1268, 1205->3696), but the frame
    # counter RESTARTS at 131,072 every job. Ordering or deduplicating on frames
    # therefore interleaves the segments into nonsense; frames have to be rebuilt
    # cumulatively instead.
    iter_tag = tag.rsplit("/", 1)[0] + "/iter"

    segments, skipped = [], []
    for f in files:
        # size_guidance scalars=0 keeps EVERY point instead of TB's default
        # reservoir-sampling to 10k (60k+ points here -- we want them all).
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        have = acc.Tags()["scalars"]
        if tag not in have or iter_tag not in have:
            skipped.append(os.path.basename(f))
            continue
        ev = acc.Scalars(tag)
        ep = acc.Scalars(iter_tag)
        n = min(len(ev), len(ep))
        segments.append((np.array([e.step for e in ep[:n]], dtype=np.float64),
                         np.array([e.step for e in ev[:n]], dtype=np.float64),
                         np.array([e.wall_time for e in ev[:n]], dtype=np.float64),
                         np.array([e.value for e in ev[:n]], dtype=np.float64)))
    if not segments:
        raise ValueError(
            f"no event file under {run_dir} has tag {tag!r} "
            f"({len(files)} file(s) checked). Is this a run's summaries/ dir?")
    if skipped:
        print(f"  [{os.path.basename(run_dir)}] skipped {len(skipped)} event "
              f"file(s) with no {tag!r} (empty/aborted jobs)", file=sys.stderr)

    segments.sort(key=lambda s: s[0][0])          # by first EPOCH

    # Frames and wall clock are both rebuilt CUMULATIVELY: each segment counts
    # only what it did itself, laid end to end. Segment-local frames already count
    # from that job's start, so they just get the running total added. For time,
    # using raw timestamps would charge the hours a requeued job sat in the slurm
    # queue to training.
    frames_adj, walls_adj = [], []
    f_off = w_off = 0.0
    for _, frames_s, walls_s, _ in segments:
        frames_adj.append(frames_s + f_off)
        walls_adj.append(walls_s - walls_s[0] + w_off)
        f_off += frames_s[-1]
        w_off += walls_s[-1] - walls_s[0]

    epochs = np.concatenate([s[0] for s in segments])
    frames = np.concatenate(frames_adj)
    walls = np.concatenate(walls_adj)
    vals = np.concatenate([s[3] for s in segments])
    seg_id = np.concatenate([np.full(len(s[0]), i, dtype=np.int64)
                             for i, s in enumerate(segments)])

    # Sort by epoch, then by segment; keeping the LAST of each duplicated epoch
    # means the later (post-resume) segment wins on the replayed overlap. The
    # frames axis still counts the replayed work, because it was really spent.
    order = np.lexsort((seg_id, epochs))
    epochs, frames, walls, vals = (epochs[order], frames[order],
                                   walls[order], vals[order])
    keep = np.ones(epochs.shape, dtype=bool)
    keep[:-1] = epochs[1:] != epochs[:-1]
    n_dup = int((~keep).sum())
    if len(segments) > 1:
        print(f"  [{os.path.basename(run_dir)}] merged {len(segments)} segments "
              f"on epoch (frames restart each resume); dropped {n_dup} replayed "
              f"point(s)", file=sys.stderr)
    return frames[keep], walls[keep], vals[keep]


def ema(values, weight):
    """TensorBoard-style exponential moving average (weight in [0,1))."""
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = weight * acc + (1.0 - weight) * v
        out[i] = acc
    return out


def value_at_frame(steps, smoothed, frame):
    """Smoothed reward at a given frame count (linear interpolation)."""
    return float(np.interp(frame, steps, smoothed))


def style_axes(ax):
    """Recessive chrome: hairline grid, muted ticks, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="run to plot; repeatable, drawn in the order given")
    p.add_argument("--tag", default="rewards0",
                   help="scalar tag family to plot (default rewards0; the "
                        "/frame and /time variants of it are read)")
    p.add_argument("--smoothing", type=float, default=0.97,
                   help="EMA weight, TensorBoard-style (default 0.97)")
    p.add_argument("--max-frames", type=float, default=None, metavar="BILLIONS",
                   help="clip the x-axis to this many BILLION env frames (and the "
                        "time panel to the wall-clock hours that budget took). Use "
                        "when comparing runs of very different lengths, so the "
                        "short ones are not squeezed into a stub.")
    p.add_argument("--out", required=True, help="output PNG path")
    args = p.parse_args(argv)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            p.error(f"--run must be LABEL=DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        runs.append((label, run_dir))

    fig, (ax_frames, ax_time) = plt.subplots(
        1, 2, figsize=(12, 4.8), facecolor=SURFACE, sharey=True)

    summary = []      # (label, steps, smoothed) for the console report
    end_labels = []   # (y, x, label, color) direct labels, placed after the loop
    for i, (label, run_dir) in enumerate(runs):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        # x = env frames, from the '<tag>/frame' variant's step field
        steps, walls, vals = load_scalar(run_dir, f"{args.tag}/frame")
        smoothed = ema(vals, args.smoothing)
        hours = (walls - walls[0]) / 3600.0  # wall clock, relative to run start

        # Clip to a shared frame budget. EMA is causal, so smoothing the full
        # series then clipping gives the same values as clipping first.
        if args.max_frames is not None:
            keep = steps <= args.max_frames * 1e9
            if not keep.any():
                raise ValueError(f"{label}: no data at or below "
                                 f"{args.max_frames}B frames (run starts at "
                                 f"{steps[0]/1e9:.3f}B)")
            steps, vals, smoothed, hours = (steps[keep], vals[keep],
                                            smoothed[keep], hours[keep])

        for ax, x in ((ax_frames, steps / 1e9), (ax_time, hours)):
            ax.plot(x, vals, color=color, linewidth=1.0, alpha=0.18)
            ax.plot(x, smoothed, color=color, linewidth=2.0, label=label)

        # Direct label at each curve's end so identity isn't color-alone.
        # Collected here, placed after the loop so near-equal endpoints can be
        # nudged apart instead of overprinting each other.
        end_labels.append((smoothed[-1], steps[-1] / 1e9, label, color))
        summary.append((label, steps, smoothed))

    # Place end labels, enforcing a minimum vertical gap (in data units) so
    # curves finishing at similar rewards stay readable.
    ylo, yhi = ax_frames.get_ylim()
    min_gap = 0.055 * (yhi - ylo)
    placed = []
    for y, x, label, color in sorted(end_labels):      # bottom-up
        y_text = y if not placed or y - placed[-1] >= min_gap else placed[-1] + min_gap
        placed.append(y_text)
        ax_frames.annotate(
            f" {label}", xy=(x, y), xytext=(x, y_text), color=color,
            fontsize=9, fontweight="bold", va="center", annotation_clip=False)

    # Console read-out: final smoothed values, plus the comparison at a COMMON
    # frame budget (runs stop at different frame counts; comparing last values
    # alone would favor whichever ran longer).
    common = min(s[-1] for _, s, _ in summary)
    print(f"[plot_tb_rewards] smoothed reward (EMA {args.smoothing}):")
    for label, steps, smoothed in summary:
        print(f"  {label:<24} last {smoothed[-1]:7.3f} @ {steps[-1]/1e9:.2f}B frames"
              f"   | at common {common/1e9:.2f}B: {value_at_frame(steps, smoothed, common):.3f}")

    for ax, xlabel in ((ax_frames, "env frames (billions)"),
                       (ax_time, "wall-clock hours")):
        style_axes(ax)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax_frames.set_ylabel("mean episode reward (raw)", color=INK_2, fontsize=10)
    ax_frames.legend(loc="lower right", frameon=False, fontsize=9,
                     labelcolor=INK_2)
    fig.suptitle(f"{args.tag}: " + "  vs  ".join(l for l, _, _ in summary),
                 color=INK, fontsize=12, x=0.02, ha="left")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"[plot_tb_rewards] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

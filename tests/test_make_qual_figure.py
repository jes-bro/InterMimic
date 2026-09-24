"""make_qual_figure.py: the frame sampling, which is what makes a strip wrong.

A render is 400 frames covering four or five ATTEMPTS at a 70-100 frame clip.
Sampling six frames across the whole file gives six different attempts, and the
resulting strip shows the object teleporting between panels -- it reads as the
method failing when it is the figure that is broken. So what is pinned here is
mostly the index maths:

  * an explicit range is honoured exactly, both ends included
  * with no range only the first attempt's worth of frames is used, never the
    whole video
  * the sampled indices never run past the end of the video

Plus the guards: a missing video, an empty range and a malformed spec all fail
loudly rather than producing a figure that looks plausible.

Videos are generated with ffmpeg testsrc, so the tests need no renders.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "make_qual_figure.py")

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not installed")

sys.path.insert(0, os.path.join(REPO, "scripts"))
import make_qual_figure as M  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 400-frame 30fps clip, the shape render_pair.sh writes."""
    p = tmp_path_factory.mktemp("vid") / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=320x180:rate=30:duration=13.34",
         "-frames:v", "400", "-pix_fmt", "yuv420p", str(p)],
        check=True)
    return str(p)


# --- the index maths ---------------------------------------------------------

def test_explicit_range_includes_both_ends():
    idx = M.frame_indices(400, 0, 71, 6)
    assert idx[0] == 0 and idx[-1] == 71 and len(idx) == 6


def test_indices_are_evenly_spaced():
    idx = M.frame_indices(400, 0, 100, 6)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_default_uses_one_attempt_not_the_whole_video():
    """The bug this script exists to avoid."""
    idx = M.frame_indices(400, 0, None, 6)
    assert idx[-1] <= 100, f"sampled out to frame {idx[-1]} -- that is 5 attempts"


def test_frac_controls_the_default_window():
    assert M.frame_indices(400, 0, None, 6, frac=0.5)[-1] > \
           M.frame_indices(400, 0, None, 6, frac=0.25)[-1]


def test_range_is_clamped_to_the_video():
    """A clip length copied from a longer render must not index past the end."""
    assert M.frame_indices(80, 0, 200, 4)[-1] == 79


def test_single_frame_is_the_start():
    assert M.frame_indices(400, 30, 90, 1) == [30]


def test_empty_range_fails_loudly():
    with pytest.raises(SystemExit) as e:
        M.frame_indices(400, 90, 30, 4)
    assert "empty" in str(e.value)


# --- spec parsing ------------------------------------------------------------

def test_spec_with_range():
    assert M.parse_spec("Basketball=/a/b.mp4:0-71") == ("Basketball", "/a/b.mp4", 0, 71)


def test_spec_without_range():
    assert M.parse_spec("Soccer=/a/b.mp4") == ("Soccer", "/a/b.mp4", 0, None)


def test_spec_path_with_colon_but_no_range():
    """A name with a colon must not be eaten as a range."""
    lab, path, s, e = M.parse_spec("X=/a/12:30/b.mp4")
    assert path == "/a/12:30/b.mp4" and e is None


def test_pipe_makes_a_second_label_line():
    """The figure must name the held-out body, not just the task -- that is what
    makes it an illustration of zero-shot transfer to an unseen embodiment."""
    lab, _, _, _ = M.parse_spec("Soccer|sub16 (held out)=/a/b.mp4")
    assert lab == "Soccer\nsub16 (held out)"


def test_backslash_n_also_makes_a_line():
    lab, _, _, _ = M.parse_spec("Soccer\\nsub16=/a/b.mp4")
    assert lab == "Soccer\nsub16"


def test_spec_without_equals_fails_loudly():
    with pytest.raises(SystemExit):
        M.parse_spec("just_a_path.mp4")


def test_spec_with_empty_label_fails_loudly():
    with pytest.raises(SystemExit):
        M.parse_spec("=/a/b.mp4")


# --- end to end --------------------------------------------------------------

def _run(args):
    return subprocess.run([sys.executable, SCRIPT] + args,
                          capture_output=True, text=True)


def test_probe_reads_the_video(video):
    n, fps, w, h = M.probe(video)
    assert n == 400 and round(fps) == 30 and (w, h) == (320, 180)


def test_missing_video_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as e:
        M.probe(str(tmp_path / "nope.mp4"))
    assert "no video at" in str(e.value)


def test_builds_a_png(video, tmp_path):
    out = tmp_path / "fig.png"
    r = _run(["--out", str(out), "--frames", "4", "--height", "120",
              f"A={video}:0-71", f"B={video}:0-71"])
    assert r.returncode == 0, r.stderr
    assert out.is_file() and out.stat().st_size > 1000


def test_builds_a_pdf_for_latex(video, tmp_path):
    out = tmp_path / "fig.pdf"
    r = _run(["--out", str(out), "--frames", "3", f"A={video}:0-60"])
    assert r.returncode == 0, r.stderr
    assert out.is_file() and out.read_bytes()[:4] == b"%PDF"


def test_reports_which_frames_it_sampled(video, tmp_path):
    """The printed indices are how you check the strip is one attempt."""
    r = _run(["--out", str(tmp_path / "f.png"), "--frames", "3",
              f"A={video}:0-71"])
    assert "sampling [0, 36, 71]" in r.stdout, r.stdout


def test_title_is_drawn_and_makes_room_for_itself(video, tmp_path):
    """A title must not overlap the top row -- the figure grows instead."""
    from PIL import Image as I
    plain = tmp_path / "plain.png"
    titled = tmp_path / "titled.png"
    _run(["--out", str(plain), "--frames", "3", "--height", "120", f"A={video}:0-60"])
    _run(["--out", str(titled), "--frames", "3", "--height", "120",
          "--title", "Zero-shot generalization to unseen embodiments",
          f"A={video}:0-60"])
    assert I.open(titled).height > I.open(plain).height


def test_a_serif_font_is_actually_available(video, tmp_path):
    """The figure sits beside LaTeX body text; falling back to the sans default
    silently would make it look like a screenshot."""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    assert have & set(M.SERIF), f"none of {M.SERIF} installed"


# --- the crop -----------------------------------------------------------------
# The camera is static and the backdrop fixed, so what changes between frames is
# the humanoid and the object. The box is the UNION over a row's frames, applied
# identically to each -- a per-frame box would re-centre him every panel, which
# reads as a tracking shot and hides the fact that he is moving through a scene.

def _square_frames(tmp_path, positions, size=40, W=320, H=180):
    from PIL import Image as I
    out = []
    for i, (x, y) in enumerate(positions):
        im = I.new("RGB", (W, H), "black")
        im.paste(I.new("RGB", (size, size), "white"), (x, y))
        p = tmp_path / f"s{i:02d}.png"
        im.save(p)
        out.append(str(p))
    return out


def test_motion_box_covers_every_frames_subject(tmp_path):
    files = _square_frames(tmp_path, [(10, 10), (100, 10), (200, 100)])
    x, y, w, h = M.motion_bbox(files, pad=0.0)
    assert x <= 10 and y <= 10
    assert x + w >= 240 and y + h >= 140        # last square's far corner


def test_motion_box_is_padded(tmp_path):
    files = _square_frames(tmp_path, [(100, 60), (140, 60)])
    tight = M.motion_bbox(files, pad=0.0)
    padded = M.motion_bbox(files, pad=0.25)
    assert padded[2] > tight[2] and padded[3] > tight[3]


def test_motion_box_none_when_nothing_moves(tmp_path):
    """A near-static clip must keep the full frame, not crop to a speck."""
    files = _square_frames(tmp_path, [(100, 60), (100, 60), (100, 60)])
    assert M.motion_bbox(files) is None


def test_motion_box_stays_inside_the_frame(tmp_path):
    files = _square_frames(tmp_path, [(0, 0), (280, 140)])
    x, y, w, h = M.motion_bbox(files, pad=0.5)
    assert x >= 0 and y >= 0 and x + w <= 320 and y + h <= 180


def test_fit_aspect_grows_to_the_target():
    x, y, w, h = M.fit_aspect((100, 50, 40, 80), 1.0, 320, 180)
    assert abs(w / h - 1.0) < 0.05 and w >= 40


def test_fit_aspect_stays_inside_the_frame():
    x, y, w, h = M.fit_aspect((0, 0, 40, 80), 3.0, 320, 180)
    assert x >= 0 and y >= 0 and x + w <= 320 and y + h <= 180


def test_auto_crop_shrinks_the_figure(video, tmp_path):
    """End to end: testsrc moves, so the crop must bite."""
    from PIL import Image as I
    full = tmp_path / "full.png"
    crop = tmp_path / "crop.png"
    _run(["--out", str(full), "--frames", "4", "--height", "120",
          "--no-auto-crop", f"A={video}:0-71"])
    _run(["--out", str(crop), "--frames", "4", "--height", "120",
          f"A={video}:0-71"])
    assert I.open(crop).width <= I.open(full).width


def test_bad_frames_argument_fails_loudly(video, tmp_path):
    r = _run(["--out", str(tmp_path / "f.png"), "--frames", "0", f"A={video}"])
    assert r.returncode != 0

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


def test_bad_frames_argument_fails_loudly(video, tmp_path):
    r = _run(["--out", str(tmp_path / "f.png"), "--frames", "0", f"A={video}"])
    assert r.returncode != 0

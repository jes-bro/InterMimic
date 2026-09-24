"""render_batch.sh: the plan it builds from a batch file, via DRY=1.

A batch of renders is seven near-identical commands differing in three words.
Pasted by hand, that produced files named r2_rev_401.mp4 -- which says the SOURCE
and not the body, so a video could not be identified afterwards. What is pinned:

  * the output name states body, source clip and policy, and is generated rather
    than chosen
  * the arm key picks the policy, so an activity clip cannot be rendered with the
    OMOMO checkpoint by typo
  * renders are dealt round-robin over the cards, one at a time per card (two
    Isaac Gym renders on one card is an OOM at minute nine)
  * a malformed line fails BEFORE anything launches, not after six cards are busy
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("scripts", "render_batch.sh")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(os.path.join(REPO, SCRIPT), tmp_path / SCRIPT)
    # render_pair.sh is not invoked under DRY=1, but its absence should not be
    # what a test passes on, so a stub stands in.
    (tmp_path / "scripts" / "render_pair.sh").write_text("#!/bin/sh\nexit 0\n")
    return tmp_path


def write_batch(repo, text, name="b.txt"):
    p = repo / name
    p.write_text(text)
    return str(p)


def run(repo, batch, **env):
    e = dict(os.environ)
    e.update(DRY="1", GPUS="1 2 3", OUTDIR=str(repo / "out"), LOGDIR=str(repo / "log"))
    e.update(env)
    return subprocess.run(["sh", SCRIPT, batch], cwd=repo, env=e,
                          capture_output=True, text=True)


ONE = "sub13 sub401 sub401_bballd03s01rev015at_002.pt act\n"


def test_output_name_states_body_source_and_policy(repo):
    out = run(repo, write_batch(repo, ONE)).stdout
    assert "body-sub13__src-sub401_bballd03s01rev015at_002__act100k.mp4" in out


def test_omomo_key_gets_the_omomo_tag(repo):
    out = run(repo, write_batch(repo, "sub13 sub15 sub15_largetable_000.pt omomo\n")).stdout
    assert "body-sub13__src-sub15_largetable_000__xf29k.mp4" in out


def test_unknown_arm_key_fails_before_launching(repo):
    r = run(repo, write_batch(repo, "sub13 sub401 c.pt bogus\n"))
    assert r.returncode == 2 and "unknown arm key 'bogus'" in r.stderr


def test_short_line_fails_loudly(repo):
    r = run(repo, write_batch(repo, "sub13 sub401\n"))
    assert r.returncode == 2 and "want <body> <source> <clip> <arm-key>" in r.stderr


def test_extra_field_fails_loudly(repo):
    """A stray word is usually a clip name with a space in the batch file."""
    r = run(repo, write_batch(repo, "sub13 sub401 c.pt act oops\n"))
    assert r.returncode == 2 and "extra field" in r.stderr


def test_a_bad_line_stops_the_whole_batch(repo):
    """Parse-then-launch: the good lines must NOT start if a later line is bad."""
    r = run(repo, write_batch(repo, ONE + "sub13 sub402 c2.pt nope\n"))
    assert r.returncode == 2
    assert "gpu 1" not in r.stdout


def test_comments_and_blank_lines_are_ignored(repo):
    b = write_batch(repo, "# a comment\n\n" + ONE + "\n# another\n")
    out = run(repo, b).stdout
    assert "1 render(s)" in out


def test_empty_batch_fails_loudly(repo):
    r = run(repo, write_batch(repo, "# nothing but comments\n"))
    assert r.returncode == 2 and "no render lines" in r.stderr


def test_missing_batch_file_fails_loudly(repo):
    r = run(repo, str(repo / "nope.txt"))
    assert r.returncode == 2 and "no batch file" in r.stderr


def test_renders_are_dealt_round_robin(repo):
    b = write_batch(repo, ONE * 3)
    out = run(repo, b).stdout
    for g in ("gpu 1", "gpu 2", "gpu 3"):
        assert g in out


def test_more_renders_than_cards_wraps_rather_than_dropping(repo):
    """7 renders over 3 cards is the real case; none may be silently lost."""
    b = write_batch(repo, ONE * 7)
    out = run(repo, b).stdout
    assert "7 render(s) over 3 GPU(s)" in out
    assert out.count("-> body-") == 7


def test_dry_run_launches_nothing(repo):
    run(repo, write_batch(repo, ONE))
    assert not (repo / "out").exists() or list((repo / "out").iterdir()) == []


def test_the_shipped_sub13_batch_parses(repo):
    """The committed batch file must actually be loadable by the script."""
    shutil.copytree(os.path.join(REPO, "scripts", "batches"),
                    repo / "scripts" / "batches")
    r = run(repo, "scripts/batches/sub13_probe.txt")
    assert r.returncode == 0, r.stderr
    assert "7 render(s)" in r.stdout

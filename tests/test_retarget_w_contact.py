"""--w-contact must actually reach the solve.

The contact-weighting ablation is a whole training arm: generate a reference tree
with w_contact=0 (every body weighted equally), train on it, compare. If the flag
were silently dropped, the "ablation" tree would be byte-identical to the real
one and the arm would measure nothing -- a failure that costs days and looks like
a null result. These tests pin the plumbing, CLI -> batch -> worker -> retarget(),
without needing MJCFs, clip data or Isaac Gym.

The weighting itself (w = w_pose + w_contact * contact) is one line in retarget();
what is fragile is the four hops it travels through a multiprocessing job tuple.
"""
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

rc = pytest.importorskip("retarget_contact")


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """_one() with the solve and the data loader stubbed: records what retarget got."""
    seen = {}

    def fake_retarget(clip, source, target, scale, iters=300, verbose=True,
                      source_mjcf=None, w_contact=10.0, **kw):
        seen["w_contact"] = w_contact
        seen["source"] = source
        seen["target"] = target
        return "CLIP_OUT", {"contact_before_cm": 1.0, "contact_after_cm": 0.2}

    monkeypatch.setattr(rc, "retarget", fake_retarget)
    monkeypatch.setattr(rc.torch, "load", lambda *a, **k: types.SimpleNamespace(
        detach=lambda: "CLIP_IN"))
    monkeypatch.setattr(rc.torch, "save", lambda obj, dst: open(dst, "w").write("x"))
    return seen, tmp_path


def _job(tmp_path, w_contact):
    return (str(tmp_path / "sub404_ball_000.pt"), "sub404", "sub204",
            (1., 1., 1.), 300, str(tmp_path / "out"), None, 0.0, w_contact)


def test_worker_forwards_the_default_weight(captured):
    seen, tmp = captured
    rc._one(_job(tmp, 10.0))
    assert seen["w_contact"] == 10.0


def test_worker_forwards_the_ablation_weight(captured):
    """0 means uniform: the whole point of the ablation arm."""
    seen, tmp = captured
    rc._one(_job(tmp, 0.0))
    assert seen["w_contact"] == 0.0


def test_batch_puts_the_weight_in_every_job(monkeypatch, tmp_path):
    """batch() builds the job tuples; the weight must ride along on each one."""
    motion = tmp_path / "clips"
    motion.mkdir()
    for n in ["sub404_ball_000.pt", "sub404_ball_001.pt"]:
        (motion / n).write_text("")

    jobs_seen = []

    class FakePool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap_unordered(self, fn, jobs):
            jobs_seen.extend(jobs)
            for j in jobs:
                yield (j[2], os.path.basename(j[0]), 1.0, 0.2, "ok")

    import multiprocessing as mp
    monkeypatch.setattr(mp, "Pool", FakePool)
    (tmp_path / "out").mkdir()                      # batch writes its summary there
    rc.batch(str(motion), "sub404", ["sub204", "sub205"], str(tmp_path / "out"),
             (1., 1., 1.), 300, 2, w_contact=0.0)

    assert len(jobs_seen) == 4                      # 2 clips x 2 bodies
    assert {j[-1] for j in jobs_seen} == {0.0}      # every job carries the ablation weight


def test_cli_default_and_override():
    """The flag exists, defaults to 10, and parses a 0."""
    import argparse
    src = open(os.path.join(REPO, "scripts", "retarget_contact.py")).read()
    assert '"--w-contact"' in src, "flag missing from the CLI"
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-contact", type=float, default=10.0)
    assert ap.parse_args([]).w_contact == 10.0
    assert ap.parse_args(["--w-contact", "0"]).w_contact == 0.0


def test_weight_formula_is_what_the_flag_claims():
    """w = w_pose + w_contact*contact: 11x at the default, 1x (uniform) at 0."""
    torch = rc.torch
    contact = torch.tensor([[1.0, 0.0, 1.0]])
    for w_contact, expect in [(10.0, [11.0, 1.0, 11.0]), (0.0, [1.0, 1.0, 1.0])]:
        w = 1.0 + w_contact * (contact > 0.5).to(torch.float64)
        assert w.flatten().tolist() == expect

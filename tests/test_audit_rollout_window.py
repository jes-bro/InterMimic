#!/usr/bin/env python3
"""Tests for scripts/audit_rollout_window.py.

The arithmetic here has to match the task exactly, because the whole point is
to detect a condition that produces NO error and NO warning at runtime: a start
window of width 1, where stateInit Hybrid silently behaves as Start.

Pinned against the source:
  start window   max(1, T - rollout_length)   intermimic.py:1269, :1297
  PSI harvest    T >= rollout_length          psi_update.py:104

Run:  python tests/test_audit_rollout_window.py   (exit 0 = all green)
  or: pytest tests/test_audit_rollout_window.py
"""
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import audit_rollout_window as arw  # noqa: E402


def make_clips(root, spec, nested=False):
    """spec: {clip_name: n_frames}. nested=True writes the body-major layout."""
    for name, T in spec.items():
        d = os.path.join(root, "sub2") if nested else root
        os.makedirs(d, exist_ok=True)
        torch.save(torch.zeros(T, 591), os.path.join(d, name))


def test_start_window_matches_the_task_formula():
    # the exact expression at intermimic.py:1269
    assert arw.start_window(202, 300) == 1        # collapsed -> always frame 0
    assert arw.start_window(202, 50) == 152
    assert arw.start_window(101, 50) == 51
    assert arw.start_window(300, 300) == 1
    assert arw.start_window(301, 300) == 1        # T-R == 1 is still one choice
    assert arw.start_window(302, 300) == 2        # first genuinely free case
    print("ok: start window matches max(1, T - R)")


def test_pinning_and_psi_counts():
    lengths = {"a.pt": 147, "b.pt": 202, "c.pt": 309}
    rows = {r["rollout"]: r for r in arw.summarize(lengths, [50, 300])}

    assert rows[50]["n_pinned"] == 0
    assert rows[50]["n_psi"] == 3                 # all clips >= 50

    # at 300: 147 and 202 collapse; 309 keeps a 9-frame window
    assert rows[300]["n_pinned"] == 2
    assert rows[300]["n_psi"] == 1                # only the 309-frame clip
    print("ok: pinned and PSI-eligible counts")


def test_psi_gate_is_inclusive_at_equality():
    # psi_update.py:104 is `mel >= rollout_length`, so T == R still harvests
    rows = {r["rollout"]: r for r in arw.summarize({"a.pt": 300}, [300, 301])}
    assert rows[300]["n_psi"] == 1
    assert rows[301]["n_psi"] == 0
    print("ok: PSI gate is inclusive at T == R")


def test_max_safe_rollout_leaves_a_real_choice_of_start():
    lengths = {"a.pt": 147, "b.pt": 202}
    safe = arw.max_safe_rollout(lengths)
    assert safe == 145                            # min(T) - 2
    assert arw.start_window(147, safe) == 2       # two start frames, not one
    assert arw.summarize(lengths, [safe])[0]["n_pinned"] == 0
    assert arw.summarize(lengths, [safe])[0]["n_psi"] == 2
    # one higher pins the shortest clip
    assert arw.summarize(lengths, [safe + 1])[0]["n_pinned"] == 1
    print("ok: max safe rollout is the largest R that pins nothing")


def test_max_safe_rollout_none_for_a_degenerate_clip():
    assert arw.max_safe_rollout({"a.pt": 2}) is None
    assert arw.max_safe_rollout({}) is None
    print("ok: an unusably short clip yields no safe rollout")


def test_reads_flat_and_body_major_layouts():
    with tempfile.TemporaryDirectory() as tmp:
        flat, nested = os.path.join(tmp, "flat"), os.path.join(tmp, "bodymajor")
        make_clips(flat, {"sub2_a_000.pt": 120, "sub2_a_001.pt": 240})
        make_clips(nested, {"sub100_bball_000.pt": 101}, nested=True)
        assert sorted(arw.clip_lengths(flat).values()) == [120, 240]
        nl = arw.clip_lengths(nested)
        assert list(nl.values()) == [101]
        assert list(nl)[0] == os.path.join("sub2", "sub100_bball_000.pt")
    print("ok: reads flat and body-major layouts")


def test_ragged_clip_lengths_are_flagged():
    """Retargeting must preserve frame count; differing lengths for one clip
    name across bodies is a fault, not a curiosity."""
    with tempfile.TemporaryDirectory() as tmp:
        for body, T in (("sub2", 101), ("sub6", 99)):
            d = os.path.join(tmp, body)
            os.makedirs(d)
            torch.save(torch.zeros(T, 591), os.path.join(d, "sub100_bball_000.pt"))
        text = arw.report(tmp, arw.clip_lengths(tmp), [50])
        assert "WARNING" in text and "differing frame" in text
        assert "[99, 101]" in text
    print("ok: ragged per-body clip lengths are flagged")


def test_report_marks_the_silent_pinning_case():
    lengths = {"a.pt": 202}
    text = arw.report("d", lengths, [50, 300])
    assert "Hybrid silently == Start" in text     # the flag only on the bad row
    assert text.count("Hybrid silently == Start") == 1
    print("ok: report flags the pinned rows explicitly")


def test_main_end_to_end_and_missing_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "m")
        make_clips(d, {"sub2_a_000.pt": 147, "sub2_a_001.pt": 309})
        assert arw.main([d]) == 0
        assert arw.main([d, "--rollouts", "50", "--per-clip"]) == 0
        assert arw.main([os.path.join(tmp, "nope")]) == 1
        assert arw.main([d, "--rollouts", "abc"]) == 1
    print("ok: main runs end to end and errors on a missing dir")


def test_local_omomo_bundle_reproduces_the_measurement():
    """Against the repo's bundled sub2 clips: R=300 pins nearly all of them and
    kills PSI, R=50 does neither. If this ever stops holding, the claim the
    g3_union rolloutLength was chosen from has changed."""
    d = os.path.join(REPO, "InterAct/OMOMO_new")
    if not os.path.isdir(d):
        print("skip: InterAct/OMOMO_new not present")
        return
    lengths = arw.clip_lengths(d)
    if not lengths:
        print("skip: no clips in InterAct/OMOMO_new")
        return
    rows = {r["rollout"]: r for r in arw.summarize(lengths, [50, 300])}
    n = rows[50]["n"]
    assert rows[50]["n_pinned"] == 0
    assert rows[50]["n_psi"] == n
    assert rows[300]["n_pinned"] > 0.8 * n, rows[300]
    assert rows[300]["n_psi"] < 0.2 * n, rows[300]
    print(f"ok: bundled OMOMO ({n} clips) -- R=300 pins {rows[300]['n_pinned']}, "
          f"R=50 pins 0")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")

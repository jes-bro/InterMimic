#!/usr/bin/env python3
"""Tests for scripts/cfg_diff.py.

The bug this tool exists to prevent: a key that is IDENTICAL in both configs is
not printed, and someone then fills that row of a comparison table from memory
(or from a different comparison) and gets it backwards. So the tests pin both
halves -- that equal keys are counted and reportable, and that absent is
reported as absent rather than as a default.

Run:  python tests/test_cfg_diff.py   (exit 0 = all green)
  or: pytest tests/test_cfg_diff.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import cfg_diff as cd  # noqa: E402


def write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


A = """
env:
  numEnvs: 4096
  rolloutLength: 300
  rewardTerms:
    pose:
      enable: true
      lambda: 0.02
  bodyNormalizedReward: true
  dataSub: ['sub2']
"""

B = """
env:
  numEnvs: 4096
  rolloutLength: 50
  rewardTerms:
    pose:
      enable: true
      lambda: 0.02
  dataSub: ['sub100']
  rewardShape: geometric
"""


def test_flatten_nests_with_dots():
    flat = cd.flatten({"env": {"rewardTerms": {"pose": {"enable": True}}, "numEnvs": 4}})
    assert flat == {"env.rewardTerms.pose.enable": True, "env.numEnvs": 4}
    print("ok: flatten produces dotted keys")


def test_equal_keys_are_counted_not_dropped():
    """THE regression: pose is identical in both, so it must be counted as
    identical -- never silently unexamined."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = cd.load(write(tmp, "a.yaml", A)), cd.load(write(tmp, "b.yaml", B))
        rows, n_compared, n_same = cd.compare(a, b)
        by_key = {k: (va, vb, d) for k, va, vb, d in rows}

        assert by_key["env.rewardTerms.pose.enable"] == (True, True, False)
        assert by_key["env.rewardTerms.pose.lambda"] == (0.02, 0.02, False)
        assert by_key["env.numEnvs"][2] is False
        assert n_same == 3          # numEnvs + pose.enable + pose.lambda
        assert n_compared == len(rows)
    print("ok: identical keys are compared and counted, not dropped")


def test_absent_is_reported_as_absent():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = cd.load(write(tmp, "a.yaml", A)), cd.load(write(tmp, "b.yaml", B))
        rows, _, _ = cd.compare(a, b)
        by_key = {k: (va, vb) for k, va, vb, _ in rows}
        # only in A
        assert by_key["env.bodyNormalizedReward"] == (True, cd.ABSENT)
        # only in B
        assert by_key["env.rewardShape"] == (cd.ABSENT, "geometric")
    print("ok: one-sided keys report <absent>, not a guessed default")


def test_lists_compare_by_value():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = cd.load(write(tmp, "a.yaml", A)), cd.load(write(tmp, "b.yaml", B))
        rows, _, _ = cd.compare(a, b)
        by_key = {k: (va, vb, d) for k, va, vb, d in rows}
        assert by_key["env.dataSub"] == (["sub2"], ["sub100"], True)
    print("ok: list values compare by value")


def test_explicit_keys_include_a_key_absent_from_both():
    """Asking about a key that exists in neither file must SAY so, not return
    an empty result that reads like 'no difference'."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = cd.load(write(tmp, "a.yaml", A)), cd.load(write(tmp, "b.yaml", B))
        rows, n_compared, _ = cd.compare(a, b, keys=["env.freeFlightGate.enable",
                                                     "env.rolloutLength"])
        assert n_compared == 2
        by_key = {k: (va, vb) for k, va, vb, _ in rows}
        assert by_key["env.freeFlightGate.enable"] == (cd.ABSENT, cd.ABSENT)
        assert by_key["env.rolloutLength"] == (300, 50)
    print("ok: explicitly requested keys report even when absent from both")


def test_identical_files_report_zero_differences():
    with tempfile.TemporaryDirectory() as tmp:
        a = cd.load(write(tmp, "a.yaml", A))
        rows, n_compared, n_same = cd.compare(a, dict(a))
        assert n_same == n_compared
        assert not any(d for *_, d in rows)
    print("ok: identical files report zero differences")


def test_fmt_truncates_long_values():
    long = [f"sub{i}" for i in range(43)]
    out = cd.fmt(long, 30)
    assert len(out) == 30 and out.endswith("…")
    assert cd.fmt(5, 30) == "5"
    print("ok: long values truncate to one line")


def test_real_repo_pair_pins_the_transcription_bug():
    """Against the actual configs: pose is ON in BOTH r8 and ret_stock, while
    bodyNormalizedReward is ret_stock-only. This is the exact pair that was
    reported backwards by hand."""
    r8 = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg/"
                            "omomo_cari4d_bball_r8_horiz_train.yaml")
    ret = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg/"
                             "omomo_teacher_g2_mlp_ret_stock__f0.yaml")
    if not (os.path.isfile(r8) and os.path.isfile(ret)):
        print("skip: repo configs not present")
        return
    rows, _, _ = cd.compare(cd.load(r8), cd.load(ret),
                            keys=["env.rewardTerms.pose.enable",
                                  "env.rewardTerms.pose.lambda",
                                  "env.bodyNormalizedReward",
                                  "env.physicalBufferSize",
                                  "env.rewardShape"])
    by_key = {k: (va, vb, d) for k, va, vb, d in rows}
    assert by_key["env.rewardTerms.pose.enable"] == (True, True, False), "pose is ON in both"
    assert by_key["env.rewardTerms.pose.lambda"] == (0.02, 0.02, False)
    assert by_key["env.bodyNormalizedReward"] == (cd.ABSENT, True, True)
    assert by_key["env.physicalBufferSize"] == (cd.ABSENT, 3, True)
    assert by_key["env.rewardShape"] == ("geometric", cd.ABSENT, True)
    print("ok: real r8-vs-ret_stock pair matches the verified values")


def test_main_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = write(tmp, "a.yaml", A), write(tmp, "b.yaml", B)
        assert cd.main([a, b]) == 0
        assert cd.main([a, b, "--all"]) == 0
        assert cd.main([a, b, "--keys", "env.rolloutLength"]) == 0
        assert cd.main([os.path.join(tmp, "nope.yaml"), b]) == 1   # missing file -> 1
    print("ok: main runs end to end and errors on a missing file")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")

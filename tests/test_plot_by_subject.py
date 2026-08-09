#!/usr/bin/env python3
"""Fixture tests for scripts/plot_by_subject.py -- the data selection and
aggregation, which are the parts that can be silently wrong (a dropped CSV or a
bad mean still renders a plausible chart). Covers:

  1. load_teacher: latest-checkpoint-per-run default, @<step>k labels under
     --all-checkpoints, and same-run-same-step dedupe keeping the fuller CSV.
  2. per_body: curriculum matrices average over sources; teacher rows pass through.
  3. end-to-end main(): writes the 3 teacher figures + the curriculum figure;
     >6 curriculum runs fails loudly instead of cycling palette hues.

Run:  python tests/test_plot_by_subject.py   (exit 0 = all green)
"""
import csv
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import plot_by_subject as pbs  # noqa: E402

FIELDS = ["body", "source", "is_identity", "avg_steps", "human_pose_error",
          "object_pose_error", "success_rate", "success_count", "success_total",
          "exit_code", "timed_out", "checkpoint"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "0" for k in FIELDS}, "is_identity": "False",
                        "timed_out": "False", **r})


def teacher_row(body, run, step, success, src="sub2"):
    return dict(body=body, source=src, success_rate=success,
                human_pose_error=0.1, object_pose_error=0.2,
                checkpoint=f"checkpoints/smplx_teacher_{run}/nn/mimic_{step:08d}.pth")


def make_fixtures(d):
    # runA at two steps (latest = 200), runB once; runA@200 ALSO exists as a
    # thinner duplicate that dedupe must drop in favor of the 2-row file.
    write_csv(os.path.join(d, "smplx_teacher_runA__mimic_00000100__x.csv"),
              [teacher_row("sub1", "runA", 100, 10)])
    write_csv(os.path.join(d, "smplx_teacher_runA__mimic_00000200__x.csv"),
              [teacher_row("sub1", "runA", 200, 30),
               teacher_row("sub10", "runA", 200, 20)])
    write_csv(os.path.join(d, "smplx_teacher_runA__mimic_00000200__thin.csv"),
              [teacher_row("sub1", "runA", 200, 99)])
    write_csv(os.path.join(d, "smplx_teacher_runB__mimic_00000500__x.csv"),
              [teacher_row("sub1", "runB", 500, 50)])
    # curriculum matrix: sub1's mean over its two sources must be (40+60)/2.
    write_csv(os.path.join(d, "currX__full.csv"),
              [dict(body="sub1", source="sub1", success_rate=40,
                    human_pose_error=0.1, object_pose_error=0.1, checkpoint="c"),
               dict(body="sub1", source="sub2", success_rate=60,
                    human_pose_error=0.3, object_pose_error=0.1, checkpoint="c"),
               dict(body="sub2", source="sub1", success_rate=80,
                    human_pose_error=0.2, object_pose_error=0.1, checkpoint="c"),
               # a crashed pair (empty metrics, exit_code=1): must be dropped
               # loudly, not averaged in and not fatal
               dict(body="sub2", source="sub2", success_rate="",
                    human_pose_error="", object_pose_error="", exit_code=1,
                    checkpoint="c")])


def test_teacher_selection(d):
    runs = pbs.load_teacher(d)                       # default: latest per run
    assert set(runs) == {"runA", "runB"}, runs.keys()
    assert runs["runA"]["step"] == 200
    # dedupe kept the 2-row file (success 30), not the 1-row thin one (99)
    assert pbs.per_body(runs["runA"]["rows"], "success_rate") == {"sub1": 30.0, "sub10": 20.0}

    # Both steps kept and separately labeled -- steps 100/200 share the 0k
    # bucket, so the collision fallback must use exact-step labels.
    both = pbs.load_teacher(d, all_checkpoints=True)
    assert set(both) == {"runA@100", "runA@200", "runB"}, sorted(both)
    print("ok: teacher selection (latest-per-run, dedupe, @step labels)")


def test_curriculum_mean(d):
    cur = pbs.load_curriculum(d)
    assert set(cur) == {"currX"}
    means = pbs.per_body(cur["currX"], "success_rate")
    assert means == {"sub1": 50.0, "sub2": 80.0}, means
    hp = pbs.per_body(cur["currX"], "human_pose_error")
    assert abs(hp["sub1"] - 0.2) < 1e-9, hp
    print("ok: curriculum per-body mean over sources")


def test_end_to_end(d):
    out = os.path.join(d, "figs")
    pbs.main(["--in", d, "--out", out])
    expect = [f"teacher_by_subject_{m}.png" for m, _, _ in pbs.METRICS] \
        + ["curriculum_by_subject.png"]
    for f in expect:
        p = os.path.join(out, f)
        assert os.path.exists(p) and os.path.getsize(p) > 0, f"missing {f}"
    print("ok: end-to-end figures written:", ", ".join(expect))


def test_palette_cap(d):
    # 7 curriculum runs must fail loudly, not cycle hues.
    for i in range(7):
        write_csv(os.path.join(d, f"cap{i}__full.csv"),
                  [dict(body="sub1", source="sub1", success_rate=1,
                        human_pose_error=0.1, object_pose_error=0.1, checkpoint="c")])
    try:
        pbs.main(["--in", d, "--out", os.path.join(d, "figs2")])
    except SystemExit as e:
        assert "palette" in str(e), e
        print("ok: >6 curriculum runs fails loudly")
    else:
        raise AssertionError("expected SystemExit on 7 curriculum runs")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        make_fixtures(d)
        test_teacher_selection(d)
        test_curriculum_mean(d)
        test_end_to_end(d)
    with tempfile.TemporaryDirectory() as d:
        test_palette_cap(d)
    print("ALL GREEN")

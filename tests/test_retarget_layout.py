#!/usr/bin/env python3
"""Tests for scripts/retarget_layout.py.

What must never happen: a body silently absent from the training set. So the
tests pin every path by which a body can leave the list -- FAILED, ERROR, SKIP,
never-mentioned -- and assert each one is reported and each one blocks the run
unless it was excluded on purpose.

Run:  python tests/test_retarget_layout.py   (exit 0 = all green)
  or: pytest tests/test_retarget_layout.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import retarget_layout as rl  # noqa: E402


# A log with one of each outcome: a good solve, a solve that made things worse,
# a solve that died after printing its header, and a body with no MJCF.
LOG = """\
[bball-retarget] host=simurgh1 job=123
[retarget] sub100_bball_000.pt  sub100 -> sub2  object_scale=(1.0, 1.0, 1.0)
  contact err 4.81 -> 3.02 cm | all-body 6.10 -> 5.44 cm
  wrote InterAct/behave_cari4d_optj3d_cf2_sub2/sub100_bball_000.pt
[retarget] sub100_bball_000.pt  sub100 -> sub131  object_scale=(1.0, 1.0, 1.0)
  contact err 4.81 -> 5.10 cm | all-body 6.10 -> 6.55 cm
  wrote InterAct/behave_cari4d_optj3d_cf2_sub131/sub100_bball_000.pt
[retarget] sub100_bball_000.pt  sub100 -> sub14  object_scale=(1.0, 1.0, 1.0)
[bball-retarget] SKIP sub17: no MJCF at .../smplx_omomo_sub17.xml
"""


def test_parse_all_four_statuses():
    v = rl.parse_log(LOG)
    assert v["sub2"]["status"] == "PASS"
    assert (v["sub2"]["contact_before_cm"], v["sub2"]["contact_after_cm"]) == (4.81, 3.02)
    assert (v["sub2"]["all_before_cm"], v["sub2"]["all_after_cm"]) == (6.10, 5.44)

    assert v["sub131"]["status"] == "FAILED"        # 4.81 -> 5.10, worse
    assert "raise --iters" in v["sub131"]["note"]

    assert v["sub14"]["status"] == "ERROR"          # header, no result line
    assert v["sub14"]["contact_after_cm"] is None

    assert v["sub17"]["status"] == "SKIP"
    assert "no MJCF" in v["sub17"]["note"]
    print("ok: PASS / FAILED / ERROR / SKIP all parsed")


def test_a_dead_solve_does_not_shift_later_numbers():
    """The original verdict block zipped two independent lists; a solve that
    died after its header would slide every later body's numbers up by one."""
    log = LOG + ("[retarget] sub100_bball_000.pt  sub100 -> sub15  object_scale=(1.0,)\n"
                 "  contact err 4.81 -> 2.00 cm\n")
    v = rl.parse_log(log)
    assert v["sub14"]["status"] == "ERROR"                  # still no numbers
    assert v["sub15"]["contact_after_cm"] == 2.00           # numbers land on sub15
    assert v["sub15"]["status"] == "PASS"
    print("ok: a crashed solve does not shift later bodies' numbers")


def test_equal_error_counts_as_failed():
    """'did not improve' means not strictly better -- a wash is not a pass."""
    log = ("[retarget] c.pt  sub100 -> sub9  object_scale=(1.0,)\n"
           "  contact err 4.00 -> 4.00 cm\n")
    assert rl.parse_log(log)["sub9"]["status"] == "FAILED"
    print("ok: an unchanged contact error is FAILED, not PASS")


def test_plan_reports_every_drop_with_a_reason():
    v = rl.parse_log(LOG)
    included, dropped, unaccounted = rl.plan(v, ["sub2", "sub131", "sub14", "sub17"])
    assert included == ["sub2"]
    assert {b for b, _, _ in dropped} == {"sub131", "sub14", "sub17"}
    assert all(why for _, _, why in dropped), "every drop must carry a reason"
    assert unaccounted == []
    print("ok: every dropped body carries a status and a reason")


def test_never_mentioned_body_is_unaccounted_not_dropped():
    """A body the log never names cannot be called failed OR passed."""
    v = rl.parse_log(LOG)
    included, dropped, unaccounted = rl.plan(v, ["sub2", "sub999"])
    assert included == ["sub2"]
    assert unaccounted == ["sub999"]
    assert "sub999" not in {b for b, _, _ in dropped}
    print("ok: a never-mentioned body is UNACCOUNTED, distinct from dropped")


def test_explicit_exclusion_moves_a_body_out_of_the_way():
    v = rl.parse_log(LOG)
    included, dropped, _ = rl.plan(v, ["sub2", "sub131"], exclude=["sub131"])
    assert included == ["sub2"]
    assert ("sub131", "FAILED", "excluded explicitly on the command line") in dropped
    print("ok: explicit exclusion is recorded as a deliberate act")


def test_a_passing_body_can_also_be_excluded_on_purpose():
    v = rl.parse_log(LOG)
    included, dropped, _ = rl.plan(v, ["sub2"], exclude=["sub2"])
    assert included == []
    assert dropped[0][0] == "sub2"
    print("ok: exclusion applies to passing bodies too")


def test_table_lists_every_requested_body():
    v = rl.parse_log(LOG)
    table = rl.format_table(v, ["sub2", "sub131", "sub14", "sub17", "sub999"])
    for body in ("sub2", "sub131", "sub14", "sub17", "sub999"):
        assert body in table
    assert "UNACCOUNTED" in table
    assert "4.81 ->   3.02 cm" in table
    print("ok: the printed table covers every requested body")


def test_manifest_round_trips_including_unaccounted():
    v = rl.parse_log(LOG)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, rl.MANIFEST_NAME)
        rl.write_manifest(path, v, ["sub2", "sub131", "sub999"])
        rows = [l.split("\t") for l in open(path).read().strip().split("\n")]
        assert rows[0] == rl.MANIFEST_COLS
        by_body = {r[0]: r for r in rows[1:]}
        assert by_body["sub2"][1] == "PASS"
        assert by_body["sub131"][1] == "FAILED"
        assert by_body["sub999"][1] == "UNACCOUNTED"   # written, not omitted
        assert len(rows) == 4
    print("ok: manifest records every body, unaccounted included")


def test_assemble_builds_body_major_and_errors_on_a_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        clip = "sub100_bball_000.pt"
        for b in ("sub2", "sub6"):
            d = os.path.join(tmp, f"flat_{b}")
            os.makedirs(d)
            with open(os.path.join(d, clip), "w") as fh:
                fh.write(b)
        out = os.path.join(tmp, "bodymajor")
        written = rl.assemble(os.path.join(tmp, "flat"), out, ["sub2", "sub6"], clip)
        assert len(written) == 2
        # the layout intermimic.py:302 requires: <dir>/<body>/<clip>
        assert open(os.path.join(out, "sub2", clip)).read() == "sub2"
        assert open(os.path.join(out, "sub6", clip)).read() == "sub6"

        try:
            rl.assemble(os.path.join(tmp, "flat"), out, ["sub9"], clip)
        except rl.LayoutError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("a PASS body with no file on disk must error")
    print("ok: assemble builds body-major layout and errors on a missing clip")


def _fixture_tree(tmp):
    """A log + flat dirs + a body-list YAML, wired together."""
    clip = "sub100_bball_000.pt"
    for b in ("sub2", "sub131"):
        d = os.path.join(tmp, f"flat_{b}")
        os.makedirs(d)
        with open(os.path.join(d, clip), "w") as fh:
            fh.write(b)
    log = os.path.join(tmp, "r.out")
    with open(log, "w") as fh:
        fh.write(LOG)
    cfg = os.path.join(tmp, "arm.yaml")
    with open(cfg, "w") as fh:
        fh.write("env:\n  subjectBodies: ['sub2', 'sub131']\n")
    return log, cfg, os.path.join(tmp, "flat"), os.path.join(tmp, "out"), clip


def test_main_refuses_an_unstated_failure_then_accepts_the_exclusion():
    with tempfile.TemporaryDirectory() as tmp:
        log, cfg, flat, out, clip = _fixture_tree(tmp)
        common = ["--log", log, "--flat-prefix", flat, "--out-dir", out,
                  "--clip", clip, "--bodies-from", cfg]

        # sub131 FAILED and was not excluded -> refuse, write nothing
        assert rl.main(common) == 1
        assert not os.path.exists(os.path.join(out, rl.MANIFEST_NAME))

        # named explicitly -> proceeds
        assert rl.main(common + ["--exclude", "sub131"]) == 0
        assert os.path.isfile(os.path.join(out, "sub2", clip))
        assert not os.path.exists(os.path.join(out, "sub131"))
        # the manifest still records the excluded body and why it failed
        text = open(os.path.join(out, rl.MANIFEST_NAME)).read()
        assert "sub131" in text and "FAILED" in text
    print("ok: refuses an unstated failure, proceeds once it is stated")


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        log, cfg, flat, out, clip = _fixture_tree(tmp)
        rc = rl.main(["--log", log, "--flat-prefix", flat, "--out-dir", out,
                      "--clip", clip, "--bodies-from", cfg,
                      "--exclude", "sub131", "--dry-run"])
        assert rc == 0
        assert not os.path.exists(out)
    print("ok: dry run writes nothing")


def test_bodies_from_cfg_reads_the_real_f0_arm():
    cfg = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg/"
                             "omomo_teacher_g2_mlp_ret_stock__f0.yaml")
    if not os.path.isfile(cfg):
        print("skip: f0 config not present")
        return
    bodies = rl.bodies_from_cfg(cfg)
    assert len(bodies) == 43, len(bodies)
    assert bodies[0] == "sub1" and "sub100" in bodies
    print("ok: reads the real f0 arm's 43 bodies")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")

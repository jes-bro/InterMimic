#!/usr/bin/env python3
"""Fixture tests for scripts/experiment_timeline.py.

The script's whole job is stitching many slurm jobs into one experiment row, so
the things worth pinning are the stitching and the honesty of the fallbacks:

  1. a job id is tied to an experiment by the launcher's OWN output line, not by
     guessing from the abbreviated job name (bball-r2_warm != smplx_..._r2_warm)
  2. several jobs collapse into ONE row: earliest start, latest end, summed
     compute -- and wall span must include the gap between resubmits, because
     that gap is real elapsed time the experiment was not running
  3. a job sacct has forgotten is flagged approx=YES, never silently dropped and
     never quietly mixed in with real accounting data
  4. an experiment with no logs is reported as NO LOGS, not omitted -- a typo'd
     name must be visible, since silently returning 16 rows for 17 requested
     names is exactly how a missing run goes unnoticed

Fixtures are written to a temp dir; nothing touches the real repo or sacct.

Run:  python tests/test_experiment_timeline.py   (exit 0 = all green)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
from experiment_timeline import (  # noqa: E402
    parse_sacct, parse_elapsed, parse_time, scan_logs, summarise, humanise,
    parse_progress,
)

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def write_log(d, name, body):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write(body)
    return p


LAUNCH = ("[bball-{tag}] invocation: python -u -m intermimic.run ...\n"
          "[bball-{tag}] host=simurgh2.stanford.edu job={job} "
          "-> checkpoints/{exp}/nn/\n"
          "Setting seed: 9198\n")


def test_scan_logs():
    print("1. scan_logs -- job id tied to experiment by the log's own line:")
    with tempfile.TemporaryDirectory() as d:
        write_log(d, "cari4d-bball-r2_warm-17021158.out",
                  LAUNCH.format(tag="r2_warm", job="17021158",
                                exp="smplx_cari4d_bball_r2_warm"))
        write_log(d, "cari4d-bball-r2_warm-17030000.out",
                  LAUNCH.format(tag="r2_warm", job="17030000",
                                exp="smplx_cari4d_bball_r2_warm"))
        write_log(d, "cari4d-bball-r3_roll30-17037568.out",
                  LAUNCH.format(tag="r3_roll30", job="17037568",
                                exp="smplx_cari4d_bball_r3_roll30"))
        # a log that is not an experiment launch at all
        write_log(d, "bball-render-17040000.out", "[bball-render] done:\n")
        found = scan_logs(d)

        check("two experiments discovered", set(found) == {
            "smplx_cari4d_bball_r2_warm", "smplx_cari4d_bball_r3_roll30"},
            f"(got {sorted(found)})")
        check("r2_warm's TWO resubmits grouped under one experiment",
              set(found.get("smplx_cari4d_bball_r2_warm", {})) ==
              {"17021158", "17030000"})
        check("a render log is not mistaken for an experiment",
              all("render" not in e for e in found))

    print("\n2. scan_logs -- job id recovered from the filename if the line lacks it:")
    with tempfile.TemporaryDirectory() as d:
        write_log(d, "cari4d-bball-old-16000001.out",
                  "[run] task=InterMimic experiment=smplx_cari4d_bball_looseterm "
                  "-> checkpoints/...\n")
        found = scan_logs(d)
        check("experiment found via the run.py banner",
              "smplx_cari4d_bball_looseterm" in found, f"(got {sorted(found)})")
        check("job id taken from the filename",
              set(found.get("smplx_cari4d_bball_looseterm", {})) == {"16000001"})


def test_parsers():
    print("\n3. sacct parsing:")
    text = ("17021158|bball-r2_warm|2026-08-24T14:01:33|2026-08-24T14:02:10|"
            "00:00:37|FAILED|1:0\n"
            "17030000|bball-r2_warm|2026-08-24T16:00:00|2026-08-25T16:00:00|"
            "1-00:00:00|TIMEOUT|0:0\n")
    rows = parse_sacct(text)
    check("both rows parsed", set(rows) == {"17021158", "17030000"})
    check("elapsed 00:00:37 -> 37s", parse_elapsed("00:00:37") == 37)
    check("elapsed 1-00:00:00 -> 86400s", parse_elapsed("1-00:00:00") == 86400)
    check("running job's End=Unknown -> None", parse_time("Unknown") is None)
    check("humanise(86400) == 24h00m", humanise(86400) == "24h00m",
          f"(got {humanise(86400)})")


def test_summarise_stitches_jobs():
    print("\n4. summarise -- many jobs, one row:")
    sacct = parse_sacct(
        "17021158|bball-r2_warm|2026-08-24T14:00:00|2026-08-24T15:00:00|"
        "01:00:00|FAILED|1:0\n"
        "17030000|bball-r2_warm|2026-08-24T20:00:00|2026-08-25T02:00:00|"
        "06:00:00|TIMEOUT|0:0\n")
    jobs = {"17021158": "/tmp/a.out", "17030000": "/tmp/b.out"}
    r = summarise("smplx_cari4d_bball_r2_warm", jobs, sacct)

    check("job count is 2", r["jobs"] == 2)
    check("first_start is the EARLIER job's start",
          r["first_start"] == "2026-08-24 14:00", f"(got {r['first_start']})")
    check("last_end is the LATER job's end",
          r["last_end"] == "2026-08-25 02:00", f"(got {r['last_end']})")
    # 14:00 -> 02:00 next day = 12h wall, but only 7h of it was computing:
    # the 5h queue gap between resubmits is real and must not be hidden.
    check("wall_span spans the queue gap (12h)", r["wall_span"] == "12h00m",
          f"(got {r['wall_span']})")
    check("compute is the SUM of Elapsed (7h), not the span",
          r["compute"] == "7h00m", f"(got {r['compute']})")
    check("not flagged approximate when sacct knew every job", r["approx"] == "")


def test_missing_sacct_is_flagged():
    print("\n5. a job sacct forgot is flagged, not dropped:")
    with tempfile.TemporaryDirectory() as d:
        p = write_log(d, "cari4d-bball-old-16000001.out", "x\n")
        # sacct knows nothing about this job
        r = summarise("smplx_cari4d_bball_rectinj3", {"16000001": p}, {})
        check("row still produced (not silently dropped)", r["jobs"] == 1)
        check("flagged approx=YES", r["approx"] == "YES", f"(got {r['approx']!r})")
        check("job state records NO-SACCT", "NO-SACCT" in r["job_states"],
              f"(got {r['job_states']})")
        check("end time falls back to the log mtime", r["last_end"] != "(running?)")
        # No start time can be honestly invented, so it must stay blank rather
        # than being guessed from the mtime.
        check("start time is NOT invented", r["first_start"] == "-",
              f"(got {r['first_start']})")


def test_running_job():
    print("\n6. a still-running job:")
    sacct = parse_sacct("17037568|bball-r3_roll30|2026-08-24T14:01:00|Unknown|"
                        "06:23:00|RUNNING|0:0\n")
    r = summarise("smplx_cari4d_bball_r3_roll30", {"17037568": "/tmp/x.out"}, sacct)
    check("start is known", r["first_start"] == "2026-08-24 14:01")
    check("end shows as still running", r["last_end"] == "(running?)",
          f"(got {r['last_end']})")
    check("compute still counted from Elapsed", r["compute"] == "6h23m",
          f"(got {r['compute']})")


PROGRESS_LOG = """[warm-start] Successfully restored from checkpoints/smplx_teachers_new/sub2.pth; resuming at epoch 12970
[bball-r3_roll30] host=x job=17037568 -> checkpoints/smplx_cari4d_bball_r3_roll30/nn/
epoch_num:12971 mean_rewards:[0.10] fps step: 20000.0 fps total: 16000.0
epoch_num:12972 mean_rewards:[0.20] fps step: 20100.0 fps total: 16200.0
epoch_num:14304 mean_rewards:[0.30] fps step: 20167.8 fps total: 16551.5
"""

FRESH_LOG = """[bball-rectinj3] host=x job=16000002 -> checkpoints/smplx_cari4d_bball_rectinj3/nn/
epoch_num:1 mean_rewards:[0.01] fps step: 19000.0 fps total: 15000.0
epoch_num:2000 mean_rewards:[0.05] fps step: 19500.0 fps total: 15500.0
"""


def test_parse_progress():
    print("\n7. progress parsing -- epochs must not be inflated by the warm start:")
    with tempfile.TemporaryDirectory() as d:
        p = write_log(d, "cari4d-bball-r3_roll30-17037568.out", PROGRESS_LOG)
        pr = parse_progress(p)
        check("last epoch_num read", pr["last_epoch"] == 14304, f"(got {pr['last_epoch']})")
        check("warm-start epoch captured", pr["warm_epoch"] == 12970,
              f"(got {pr['warm_epoch']})")
        check("reward is the tail MEAN, not one noisy value",
              abs(pr["reward_tail"] - 0.20) < 1e-9, f"(got {pr['reward_tail']})")
        check("fps total parsed", abs(pr["fps_total"] - 16250.5) < 1e-6,
              f"(got {pr['fps_total']})")

        r = summarise("smplx_cari4d_bball_r3_roll30", {"17037568": p}, {})
        # 14304 - 12970 = 1334 epochs actually trained, NOT 14304.
        check("epochs = final - warm_from (1334, not 14304)", r["epochs"] == 1334,
              f"(got {r['epochs']})")
        check("raw epoch_num still reported", r["epoch_num"] == 14304)
        check("warm_from surfaced so the offset is visible", r["warm_from"] == 12970)
        check("latest_job is the job id", r["latest_job"] == "17037568")

    print("\n8. a FRESH run has no warm-start offset:")
    with tempfile.TemporaryDirectory() as d:
        p = write_log(d, "cari4d-bball-rectinj3-16000002.out", FRESH_LOG)
        r = summarise("smplx_cari4d_bball_rectinj3", {"16000002": p}, {})
        check("epochs == epoch_num when never warm-started", r["epochs"] == 2000,
              f"(got {r['epochs']})")
        check("warm_from blank for a fresh run", r["warm_from"] == "",
              f"(got {r['warm_from']!r})")

    print("\n9. resubmits: epochs measured from the FIRST job's baseline:")
    with tempfile.TemporaryDirectory() as d:
        a = write_log(d, "cari4d-bball-r2_warm-17021158.out", PROGRESS_LOG)
        # the resubmit resumes from the run's OWN checkpoint at a later epoch
        b = write_log(d, "cari4d-bball-r2_warm-17030000.out",
                      "[warm-start] Successfully restored from checkpoints/x/nn/mimic.pth; "
                      "resuming at epoch 14304\n"
                      "epoch_num:15907 mean_rewards:[0.31] fps step: 20000.0 fps total: 16600.0\n")
        r = summarise("smplx_cari4d_bball_r2_warm", {"17021158": a, "17030000": b}, {})
        # baseline is the FIRST job's 12970, not the resubmit's 14304
        check("epochs sums both jobs' deltas (1334+1603=2937)", r["epochs"] == 2937,
              f"(got {r['epochs']})")
        check("latest_job is the newer id", r["latest_job"] == "17030000")
        check("reward comes from the LATEST job", r["reward"] == "0.310",
              f"(got {r['reward']})")


def test_non_monotonic_epochs():
    """The rectinj3 case: one job resumes high, another runs a separate range.

    max(last_epoch) - first_baseline gave 35 epochs for 24h of compute, because
    the second job's entire range was invisible to max(). Per-job deltas are
    immune: each job's advance is well-defined no matter what the others did.
    """
    print("\n10. non-monotonic epoch counters across jobs (the rectinj3 bug):")
    with tempfile.TemporaryDirectory() as d:
        # job A resumed at 13000, advanced 35 epochs, then died
        a = write_log(d, "cari4d-bball-rectinj3-16864000.out",
                      "[warm-start] Successfully restored from checkpoints/x/nn/mimic.pth; "
                      "resuming at epoch 13000\n"
                      "[bball-rectinj3] host=x job=16864000 -> checkpoints/smplx_cari4d_bball_rectinj3/nn/\n"
                      "epoch_num:13035 mean_rewards:[1.40] fps step: 22000.0 fps total: 18500.0\n")
        # job B started FRESH and ran to 12000 -- a lower number than job A's end
        b = write_log(d, "cari4d-bball-rectinj3-16864697.out",
                      "[bball-rectinj3] host=x job=16864697 -> checkpoints/smplx_cari4d_bball_rectinj3/nn/\n"
                      "epoch_num:12000 mean_rewards:[1.48] fps step: 22100.0 fps total: 18561.0\n")
        r = summarise("smplx_cari4d_bball_rectinj3",
                      {"16864000": a, "16864697": b}, {})
        # old logic: max(13035,12000) - 13000 = 35.  new: 35 + 12000 = 12035.
        check("epochs sums per-job deltas (12035, not 35)", r["epochs"] == 12035,
              f"(got {r['epochs']})")
        check("raw epoch_num still shows the max seen", r["epoch_num"] == 13035)
        check("warm_from still reports the first job's baseline",
              r["warm_from"] == 13000)
        # Job B began at 0 while job A had already ended at 13035 -- the chain is
        # not one run. RESTART is the stronger, more accurate flag here.
        check("flagged RESTART (cfg changed mid-experiment)",
              r["epoch_flag"] == "RESTART", f"(got {r['epoch_flag']!r})")

    print("\n11. a clean resubmit chain is unaffected by the fix:")
    with tempfile.TemporaryDirectory() as d:
        a = write_log(d, "cari4d-bball-r2_warm-17021158.out", PROGRESS_LOG)
        b = write_log(d, "cari4d-bball-r2_warm-17030000.out",
                      "[warm-start] Successfully restored from checkpoints/x/nn/mimic.pth; "
                      "resuming at epoch 14304\n"
                      "epoch_num:15907 mean_rewards:[0.31] fps step: 20000.0 fps total: 16600.0\n")
        r = summarise("smplx_cari4d_bball_r2_warm", {"17021158": a, "17030000": b}, {})
        # 1334 + 1603 = 2937, same as the old final-minus-baseline answer
        check("monotonic chain still gives 2937", r["epochs"] == 2937,
              f"(got {r['epochs']})")
        check("no SUSPECT flag when every job has a baseline",
              r["epoch_flag"] == "", f"(got {r['epoch_flag']!r})")


def main():
    test_scan_logs()
    test_parsers()
    test_summarise_stitches_jobs()
    test_missing_sacct_is_flagged()
    test_running_job()
    test_parse_progress()
    test_non_monotonic_epochs()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

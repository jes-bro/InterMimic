"""Fixture tests for scripts/summarize_evals.py.

Two bugs this pins against, both of which produced a plausible-looking number:

  * the per-body value was once whichever source's row came LAST in the file,
    so a body x source CSV reported one source per body (11% vs 81% on a real
    example). Fixed by averaging over sources -- which then made every SOURCE
    count equally, so a 1-clip source outweighed a 14-clip one per clip.
  * neither weighting matched InterMimic's, which pools: successes / clips.

The table now pools over clips, so these tests check that the weighting really
is per clip, that crashed rows leave the denominator as well as the numerator,
and that pose errors are clip-weighted means rather than averages of averages.
"""
import csv
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "summarize_evals.py")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from summarize_evals import per_body_totals, pool  # noqa: E402

HEADER = ["body", "source", "is_identity", "avg_steps", "human_pose_error",
          "object_pose_error", "success_rate", "success_count", "success_total",
          "exit_code", "timed_out", "checkpoint"]
CKPT = "checkpoints/smplx_teacher_g3_bball7_geoall__f0/nn/mimic_00020000.pth"


def row(body, source, count=None, total=None, hpe="0.20", ope="0.30", crashed=False):
    """One CSV row. crashed=True is the empty-metric shape eval_per_pair.py
    writes for a pair that died: no numbers at all, exit_code 1."""
    if crashed:
        return [body, source, "False", "", "", "", "", "", "", "1", "False", CKPT]
    rate = 100.0 * count / total
    return [body, source, "False", "100", hpe, ope, f"{rate}", str(count), str(total),
            "0", "False", CKPT]


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)


def _rows(*tuples):
    return [dict(zip(HEADER, r)) for r in tuples]


# ----------------------------------------------------------------- pooling
def test_success_pools_over_clips_not_over_sources():
    """A 14-clip source and a 1-clip source must NOT count equally.
    Source-averaged this body is (50 + 100)/2 = 75; pooled it is 8/15 = 53.3."""
    rows = _rows(row("sub10", "sub409", count=7, total=14),
                 row("sub10", "sub458", count=1, total=1))
    totals = per_body_totals(rows, "success_rate")
    assert totals["sub10"] == (800, 15)          # count * 100, so the ratio is a percentage
    val, clips = pool(totals, ["sub10"])
    assert clips == 15 and abs(val - 100 * 8 / 15) < 1e-9


def test_group_pools_across_bodies_too():
    rows = _rows(row("sub10", "sub401", count=3, total=10),
                 row("sub13", "sub401", count=0, total=10),
                 row("sub16", "sub401", count=5, total=10))
    totals = per_body_totals(rows, "success_rate")
    val, clips = pool(totals, ["sub10", "sub13", "sub16"])
    assert clips == 30 and abs(val - 100 * 8 / 30) < 1e-9


def test_pose_error_is_clip_weighted_not_an_average_of_averages():
    """0.10 over 1 clip and 0.50 over 9 clips is 0.46 pooled, 0.30 unweighted."""
    rows = _rows(row("sub10", "sub458", count=1, total=1, hpe="0.10"),
                 row("sub10", "sub409", count=5, total=9, hpe="0.50"))
    totals = per_body_totals(rows, "human_pose_error")
    val, clips = pool(totals, ["sub10"])
    assert clips == 10 and abs(val - (0.10 * 1 + 0.50 * 9) / 10) < 1e-9


# ----------------------------------------------------------------- crashes
def test_crashed_rows_leave_the_denominator_as_well_as_the_numerator():
    """Counting an unknown outcome's clips as failures would be a guess."""
    rows = _rows(row("sub10", "sub401", count=5, total=10),
                 row("sub10", "sub409", crashed=True))
    totals = per_body_totals(rows, "success_rate")
    assert totals["sub10"] == (500, 10)          # not (500, 24)
    val, clips = pool(totals, ["sub10"])
    assert clips == 10 and abs(val - 50.0) < 1e-9


def test_body_with_only_crashed_rows_is_absent_not_zero():
    rows = _rows(row("sub10", "sub401", count=5, total=10),
                 row("sub13", "sub401", crashed=True))
    totals = per_body_totals(rows, "success_rate")
    assert "sub13" not in totals
    assert pool(totals, ["sub13"]) [1] == 0


# ----------------------------------------------------------------- the table
def test_table_reports_pooled_values_and_flags_crashes(tmp_path):
    p = tmp_path / "smplx_teacher_g3_bball7_geoall__f0__mimic_00020000__x.csv"
    write_csv(p, [row("sub10", "sub409", count=7, total=14),
                  row("sub10", "sub458", count=1, total=1),
                  row("sub13", "sub409", count=0, total=14),
                  row("sub16", "sub409", crashed=True),
                  row("sub2", "sub409", count=14, total=14)])
    out = subprocess.run([sys.executable, str(SCRIPT), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "POOLED OVER CLIPS" in out
    line = [l for l in out.splitlines() if "sub10=" in l][0]
    assert "sub10=53.3" in line                  # 8/15, not the 75.0 source-average
    assert "sub13=0.0" in line
    assert "sub16=" not in line                  # crashed only
    assert "1 crashed row(s) dropped" in line
    # held-out group = sub10 + sub13 pooled = 8 / 29
    assert f"{100 * 8 / 29:.1f}" in line
    assert "20,000" in line


def test_counts_flag_shows_successes_over_clips(tmp_path):
    p = tmp_path / "smplx_teacher_g3_bball7_geoall__f0__mimic_00020000__x.csv"
    write_csv(p, [row("sub10", "sub401", count=3, total=10),
                  row("sub13", "sub401", count=5, total=10)])
    out = subprocess.run([sys.executable, str(SCRIPT), "--counts", str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "held 8/20" in out


def test_pose_error_prints_enough_decimals(tmp_path):
    """The old table formatted every metric to 1 dp, so every pose error read
    as 0.2 or 0.3 and --metric was useless."""
    p = tmp_path / "smplx_teacher_g3_bball7_geoall__f0__mimic_00020000__x.csv"
    write_csv(p, [row("sub10", "sub401", count=3, total=10, hpe="0.2449")])
    out = subprocess.run([sys.executable, str(SCRIPT), "--metric", "human_pose_error", str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "0.2449" in out


def test_student_checkpoints_are_labelled_by_experiment(tmp_path):
    p = tmp_path / "s__mimic_00018000__x.csv"
    r = row("sub10", "sub401", count=3, total=10)
    r[-1] = "checkpoints/smplx_student_g3_act_xf_ret_nvadlr__f0/nn/mimic_00018000.pth"
    write_csv(p, [r])
    out = subprocess.run([sys.executable, str(SCRIPT), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "g3_act_xf_ret_nvadlr__f0" in out and "18,000" in out


def test_rolling_checkpoint_reports_rolling_not_an_invented_epoch(tmp_path):
    p = tmp_path / "r__f0__x.csv"
    r = row("sub10", "sub401", count=3, total=10)
    r[-1] = "checkpoints/smplx_teacher_g3_bball7_geoall__f0/nn/mimic.pth"
    write_csv(p, [r])
    out = subprocess.run([sys.executable, str(SCRIPT), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "rolling" in out

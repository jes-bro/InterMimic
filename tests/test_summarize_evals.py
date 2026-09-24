"""Fixture tests for scripts/summarize_evals.py.

The bug these exist to prevent: the summarizer built {body: value} directly
from the CSV rows, so a body with several sources (every body x source CSV)
was reported from whichever source's row came LAST in the file. The per-body
number must be the mean over that body's sources, with crashed pairs (empty
metric) left out rather than counted as zero.
"""
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "summarize_evals.py"
sys.path.insert(0, str(REPO / "scripts"))
from summarize_evals import per_body_means  # noqa: E402

HEADER = ["body", "source", "is_identity", "avg_steps", "human_pose_error",
          "object_pose_error", "success_rate", "success_count", "success_total",
          "exit_code", "timed_out", "checkpoint"]
CKPT = "checkpoints/smplx_teacher_g3_bball7_geoall__f0/nn/mimic_00020000.pth"


def row(body, source, sr, crashed=False):
    """One CSV row. crashed=True writes the empty-metric shape eval_per_pair.py
    emits for a pair that died (exit_code 1, no numbers)."""
    if crashed:
        return [body, source, "False", "", "", "", "", "", "", "1", "False", CKPT]
    return [body, source, "False", "100", "0.05", "0.10", f"{sr}", "1", "1024",
            "0", "False", CKPT]


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)


def test_per_body_mean_averages_over_sources_and_skips_crashes():
    rows = [dict(zip(HEADER, r)) for r in [
        row("sub10", "sub401", 10.0),
        row("sub10", "sub402", 30.0),
        row("sub10", "sub409", 0.0, crashed=True),   # must not drag the mean down
        row("sub13", "sub401", 50.0),
    ]]
    d = per_body_means(rows, "success_rate")
    assert d == {"sub10": 20.0, "sub13": 50.0}


def test_table_uses_source_mean_not_last_row(tmp_path):
    # Last-row-wins would report sub10 = 2.0 and sub13 = 90.0. The true source
    # means are 50.0 and 60.0, so the held-out group mean must be 55.0.
    p = tmp_path / "smplx_teacher_g3_bball7_geoall__f0__mimic_00020000__x.csv"
    write_csv(p, [
        row("sub10", "sub401", 98.0),
        row("sub10", "sub402", 2.0),
        row("sub13", "sub401", 30.0),
        row("sub13", "sub402", 90.0),
        row("sub16", "sub401", 40.0, crashed=True),
    ])
    out = subprocess.run([sys.executable, str(SCRIPT), str(p)],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.startswith("g3_bball7_geoall__f0")]
    # sub16 has only a crashed row, so the script first warns that the held-out
    # body is absent, then prints the table row. Read the table row.
    assert any("held-out bodies ['sub16'] not in this CSV" in l for l in lines)
    line = [l for l in lines if "sub10=" in l][0]
    assert "sub10=50.0" in line
    assert "sub13=60.0" in line
    assert "sub16=" not in line                     # crashed-only body: no number
    assert "55.0" in line                           # held-out group mean
    assert "1 crashed row(s) dropped" in line


# --- decimal places ---------------------------------------------------------
# Pose errors are metres: at 1 dp, two policies 3 mm apart print the same number,
# which is how a real difference disappears into the table. success_rate is a
# percentage and 1 dp is right for it, so the default depends on the metric.

def _err_csv(tmp_path):
    p = tmp_path / "errs.csv"
    p.write_text(
        "body,source,is_identity,avg_steps,human_pose_error,object_pose_error,"
        "success_rate,success_count,success_total,exit_code,timed_out,checkpoint\n"
        "sub10,sub1,False,100,0.11234,0.09876,50.0,1,2,0,False,nn/mimic_00010000.pth\n"
        "sub13,sub1,False,100,0.11534,0.09276,50.0,1,2,0,False,nn/mimic_00010000.pth\n"
        "sub16,sub1,False,100,0.11834,0.09076,50.0,1,2,0,False,nn/mimic_00010000.pth\n")
    return p


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT)] + args,
                          capture_output=True, text=True).stdout


def test_pose_error_defaults_to_four_decimals(tmp_path):
    out = _run([str(_err_csv(tmp_path)), "--metric", "human_pose_error"])
    assert "sub10=0.1123" in out and "sub16=0.1183" in out


def test_success_rate_still_defaults_to_one_decimal(tmp_path):
    out = _run([str(_err_csv(tmp_path)), "--metric", "success_rate"])
    assert "sub10=50.0" in out


def test_decimals_is_overridable(tmp_path):
    out = _run([str(_err_csv(tmp_path)), "--metric", "object_pose_error", "--decimals", "2"])
    assert "sub10=0.10" in out

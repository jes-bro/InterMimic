"""Tests for eval_per_pair.load_resumable -- the --resume selection rule.

A 16-body eval is ~80 minutes, so a job that hits its walltime three quarters of
the way through should keep what it paid for. But "already there" is not the
same as "succeeded": a job on a failing GPU writes a full CSV of exit_code=1
rows with empty metrics, and resuming over those would preserve the hole rather
than fill it.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_per_pair", REPO / "scripts" / "eval_per_pair.py")
epp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(epp)

FIELDS = ["body", "source", "is_identity", "avg_steps", "human_pose_error",
          "object_pose_error", "success_rate", "success_count", "success_total",
          "exit_code", "timed_out", "checkpoint"]
CKPT = "checkpoints/smplx_teacher_g2_mlp_ret_stock__f0/nn/mimic_00054600.pth"


def row(body, ok=True, ckpt=CKPT, timed_out=False):
    r = dict.fromkeys(FIELDS, "")
    r.update(body=body, source="sub2", is_identity=body == "sub2",
             exit_code="0" if ok else "1", timed_out=timed_out, checkpoint=ckpt)
    if ok:
        r.update(avg_steps="180.0", human_pose_error="0.15",
                 object_pose_error="0.12", success_rate="50.0",
                 success_count="26", success_total="52")
    return r


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def test_missing_file_resumes_nothing(tmp_path):
    assert epp.load_resumable(tmp_path / "nope.csv", CKPT) == {}


def test_keeps_only_the_pairs_that_succeeded(tmp_path):
    p = write(tmp_path / "e.csv",
              [row("sub1"), row("sub2"), row("sub3", ok=False), row("sub5")])
    keep = epp.load_resumable(p, CKPT)
    assert set(keep) == {("sub1", "sub2"), ("sub2", "sub2"), ("sub5", "sub2")}


def test_all_failed_csv_resumes_nothing(tmp_path):
    """The simurgh6 ECC case: 16 rows, every one exit_code=1, no metrics."""
    p = write(tmp_path / "e.csv",
              [row(f"sub{i}", ok=False) for i in range(1, 17)])
    assert epp.load_resumable(p, CKPT) == {}


def test_exit_zero_with_empty_metrics_is_not_kept(tmp_path):
    """Exit status alone is not proof: a parse miss leaves the metrics blank."""
    r = row("sub1")
    r["success_rate"] = ""
    p = write(tmp_path / "e.csv", [r])
    assert epp.load_resumable(p, CKPT) == {}


def test_timed_out_pair_is_retried(tmp_path):
    p = write(tmp_path / "e.csv", [row("sub1", ok=False, timed_out=True)])
    assert epp.load_resumable(p, CKPT) == {}


def test_different_checkpoint_is_a_hard_error(tmp_path):
    """One CSV must never describe two policies -- nothing downstream could tell."""
    other = "checkpoints/smplx_teacher_g2_mlp_ret_stock__f0/nn/mimic_00027000.pth"
    p = write(tmp_path / "e.csv", [row("sub1", ckpt=other)])
    with pytest.raises(SystemExit) as e:
        epp.load_resumable(p, CKPT)
    assert "different checkpoint" in str(e.value)
    assert "mimic_00027000.pth" in str(e.value)


def test_kept_rows_carry_their_metrics_through(tmp_path):
    p = write(tmp_path / "e.csv", [row("sub7")])
    keep = epp.load_resumable(p, CKPT)
    assert keep[("sub7", "sub2")]["success_rate"] == "50.0"
    assert keep[("sub7", "sub2")]["human_pose_error"] == "0.15"

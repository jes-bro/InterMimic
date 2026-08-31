"""Fixture tests for scripts/plot_gen2_by_subject.py.

The bug these exist to prevent: a too-wide --include glob pulled in a second
CSV for the same run at a DIFFERENT checkpoint, which silently overwrote the
newer numbers and left the fold groups mislabelled. The plotter must now refuse
that outright, and must classify held-out vs in-distribution from the fold
design rather than from whatever bodies happen to be in the file.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "plot_gen2_by_subject.py"

HEADER = ["body", "source", "is_identity", "avg_steps", "human_pose_error",
          "object_pose_error", "success_rate", "success_count", "success_total",
          "exit_code", "timed_out", "checkpoint"]

# The real fold design, from omomo_teacher_g2_*__f{0,1}.yaml env.subjectBodies.
REAL_BODIES = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
               "sub10", "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
               "sub17"]
HELDOUT = {"f0": {"sub10", "sub13", "sub16"}, "f1": {"sub5", "sub7", "sub12"}}


def write_csv(path, ckpt, bodies, sr=50.0):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, b in enumerate(bodies):
            w.writerow([b, "sub2", b == "sub2", 180.0, 0.15, 0.12,
                        sr + i, 26, 52, 0, False, ckpt])


def ckpt_for(cfg, fold, epoch="00054600"):
    return f"checkpoints/smplx_teacher_g2_{cfg}__{fold}/nn/mimic_{epoch}.pth"


def run(args):
    return subprocess.run([sys.executable, str(SCRIPT)] + args,
                          capture_output=True, text=True)


@pytest.fixture
def evaldir(tmp_path):
    """One CSV per (config, fold): all 16 real bodies, one checkpoint each."""
    d = tmp_path / "eval"
    d.mkdir()
    for cfg in ["mlp_plain_stock", "mlp_ret_stock"]:
        for fold in ["f0", "f1"]:
            write_csv(d / f"g2_{cfg}__{fold}.csv", ckpt_for(cfg, fold), REAL_BODIES)
    return d


def test_all_real_bodies_plotted_and_sub4_absent(evaldir, tmp_path):
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.png").exists()
    for b in REAL_BODIES:
        assert f" {b:<7}" in r.stdout, f"{b} missing from the per-body table"
    assert "sub4 " not in r.stdout


def test_heldout_flagged_per_fold_not_per_file(evaldir, tmp_path):
    """Every CSV holds all 16 bodies, so 'held out' can only come from the fold."""
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    marked = {"f0": set(), "f1": set()}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in marked and parts[1].startswith("sub"):
            if parts[2] == "yes":
                marked[parts[0]].add(parts[1])
    assert marked == HELDOUT


def test_refuses_to_mix_checkpoints(evaldir, tmp_path):
    """A second CSV for the same run at another epoch must be a hard stop."""
    write_csv(evaldir / "smplx_teacher_g2_mlp_plain_stock__f0__old.csv",
              ckpt_for("mlp_plain_stock", "f0", "00027000"), REAL_BODIES, sr=10.0)
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png")])
    assert r.returncode != 0
    assert "more than one checkpoint" in r.stderr
    assert "mimic_00027000.pth" in r.stderr and "mimic_00054600.pth" in r.stderr


def test_include_glob_isolates_one_checkpoint(evaldir, tmp_path):
    """...and narrowing --include is the documented way out of that stop."""
    write_csv(evaldir / "smplx_teacher_g2_mlp_plain_stock__f0__old.csv",
              ckpt_for("mlp_plain_stock", "f0", "00027000"), REAL_BODIES, sr=10.0)
    r = run(["--in", str(evaldir), "--include", "g2_*",
             "--out", str(tmp_path / "o.png"), "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr


def test_synthetic_bodies_dropped_by_default(evaldir, tmp_path):
    write_csv(evaldir / "g2_mlp_plain_stock__f0.csv",
              ckpt_for("mlp_plain_stock", "f0"), REAL_BODIES + ["sub100", "sub115"])
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    assert "dropped 2 synthetic-body rows" in r.stdout
    # Check the per-body TABLE, not the whole log -- the note itself says "sub100+".
    assert f" {'sub100':<7}" not in r.stdout

    r = run(["--in", str(evaldir), "--synthetic", "--out", str(tmp_path / "s.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    assert f" {'sub100':<7}" in r.stdout


def test_unknown_fold_is_a_hard_error(evaldir, tmp_path):
    """f2 has no held-out definition; guessing would mislabel generalization."""
    write_csv(evaldir / "g2_mlp_plain_stock__f2.csv",
              ckpt_for("mlp_plain_stock", "f2"), REAL_BODIES)
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png")])
    assert r.returncode != 0
    assert "not in the known fold design" in r.stderr


def test_missing_cells_are_reported_not_silent(evaldir, tmp_path):
    write_csv(evaldir / "g2_mlp_ret_stock__f1.csv",
              ckpt_for("mlp_ret_stock", "f1"), ["sub1", "sub2"])
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    assert "have no eval" in r.stdout


def test_arch_kept_in_config_label(evaldir, tmp_path):
    """MLP and transformer runs must not collapse onto the same bar."""
    for fold in ["f0", "f1"]:
        write_csv(evaldir / f"g2_xf_plain_stock__{fold}.csv",
                  ckpt_for("xf_plain_stock", fold), REAL_BODIES)
    r = run(["--in", str(evaldir), "--out", str(tmp_path / "o.png"),
             "--metric", "success_rate"])
    assert r.returncode == 0, r.stderr
    assert "mlp_plain_stock" in r.stdout and "xf_plain_stock" in r.stdout

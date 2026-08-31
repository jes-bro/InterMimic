"""Fixture tests for scripts/eval_gen2_allbodies.sh.

Exercised with DRY=1 against fake checkpoint trees, so nothing is submitted and
no cluster is needed. What matters here is that a missing checkpoint is REPORTED
rather than turned into a job that dies in three seconds -- which is exactly how
the last round of eval failures went unnoticed.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_gen2_allbodies.sh"
CELLS = ["plain_stock", "plain_nvadlr", "ret_stock", "ret_nvadlr"]


def make_ckpts(root, fold, arch="mlp", cells=CELLS, epoch="00054600"):
    """A fake checkpoint tree shaped like the real one for that fold."""
    for c in cells:
        nn = root / f"smplx_teacher_g2_{arch}_{c}__{fold}" / "nn"
        nn.mkdir(parents=True, exist_ok=True)
        name = "mimic.pth" if fold == "f1" else f"mimic_{epoch}.pth"
        (nn / name).write_bytes(b"")


def run(env=None, args=()):
    e = dict(os.environ, DRY="1")
    e.update(env or {})
    return subprocess.run(["sh", str(SCRIPT), *args], cwd=REPO, env=e,
                          capture_output=True, text=True)


@pytest.fixture
def roots(tmp_path):
    f0, f1 = tmp_path / "checkpoints", tmp_path / "collab"
    make_ckpts(f0, "f0")
    make_ckpts(f1, "f1")
    return {"F0_ROOT": str(f0), "F1_ROOT": str(f1)}


def test_submits_every_cell_and_fold(roots):
    r = run(roots, ["mlp"])
    assert r.returncode == 0, r.stderr
    assert "8 submitted, 0 skipped" in r.stdout
    for c in CELLS:
        assert f"g2_mlp_{c}__f0 ->" in r.stdout
        assert f"g2_mlp_{c}__f1 ->" in r.stdout


def test_all_sixteen_bodies_and_no_sub4(roots):
    r = run(roots, ["mlp"])
    assert r.returncode == 0, r.stderr
    # eval_one.sh echoes the caller's body override back.
    line = next(l for l in r.stdout.splitlines() if "BODIES" in l and "sub1" in l)
    bodies = [w for w in line.split() if w.startswith("sub")]
    assert len(bodies) == 16
    assert "sub4" not in bodies
    assert bodies[0] == "sub1" and "sub17" in bodies


def test_missing_checkpoint_is_reported_not_submitted(tmp_path):
    """The transformer arms may not exist; that must be visible, not a dead job."""
    f0 = tmp_path / "checkpoints"
    make_ckpts(f0, "f0", arch="mlp")          # xf tree deliberately absent
    r = run({"F0_ROOT": str(f0), "F1_ROOT": str(tmp_path / "nope"),
             "FOLDS": "f0"}, ["xf"])
    assert r.returncode == 0, r.stderr
    assert "0 submitted, 4 skipped" in r.stdout
    assert "SKIP g2_xf_plain_stock__f0: no checkpoint at" in r.stdout


def test_wrong_epoch_is_reported(roots):
    r = run({**roots, "F0_EPOCH": "00099999", "FOLDS": "f0"}, ["mlp"])
    assert r.returncode == 0, r.stderr
    assert "0 submitted, 4 skipped" in r.stdout
    assert "mimic_00099999.pth" in r.stdout


def test_latest_epoch_is_printed_not_silent(tmp_path):
    f0 = tmp_path / "checkpoints"
    make_ckpts(f0, "f0", cells=["ret_stock"], epoch="00054600")
    nn = f0 / "smplx_teacher_g2_mlp_ret_stock__f0" / "nn"
    (nn / "mimic_00061200.pth").write_bytes(b"")
    r = run({"F0_ROOT": str(f0), "F0_EPOCH": "latest", "FOLDS": "f0",
             "CELLS": "ret_stock"}, ["mlp"])
    assert r.returncode == 0, r.stderr
    assert "resolved to mimic_00061200.pth" in r.stdout


HDR = ("body,source,is_identity,avg_steps,human_pose_error,object_pose_error,"
       "success_rate,success_count,success_total,exit_code,timed_out,checkpoint\n")
GOOD = "sub1,sub2,False,180.0,0.15,0.12,50.0,26,52,0,False,x.pth\n"
# What a bad GPU writes: a full CSV whose every row failed. 17 lines, no result.
FAILED = "sub1,sub2,False,,,,,,,1,False,x.pth\n"


BODIES16 = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
            "sub10", "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
            "sub17"]


def good_rows(bodies):
    return "".join(f"{b},sub2,False,180.0,0.15,0.12,50.0,26,52,0,False,x.pth\n"
                   for b in bodies)


def test_complete_csv_is_skipped(roots, tmp_path):
    out = REPO / "eval_results" / "g2_mlp_ret_stock__f0_ep00054600_pytest.csv"
    out.parent.mkdir(exist_ok=True)
    out.write_text(HDR + good_rows(BODIES16))
    try:
        r = run({**roots, "FOLDS": "f0", "CELLS": "ret_stock", "TAG": "pytest"},
                ["mlp"])
        assert r.returncode == 0, r.stderr
        assert "is complete (16/16)" in r.stdout
        assert "0 submitted, 1 skipped" in r.stdout
    finally:
        out.unlink()


def test_partial_csv_is_resumed_not_restarted(roots, tmp_path):
    """A walltime timeout part way through must not throw away paid-for pairs."""
    out = REPO / "eval_results" / "g2_mlp_ret_stock__f0_ep00054600_pytest.csv"
    out.parent.mkdir(exist_ok=True)
    out.write_text(HDR + good_rows(BODIES16[:12]))
    try:
        r = run({**roots, "FOLDS": "f0", "CELLS": "ret_stock", "TAG": "pytest"},
                ["mlp"])
        assert r.returncode == 0, r.stderr
        assert "has 12/16 -- running the remaining 4" in r.stdout
        assert "1 submitted, 0 skipped" in r.stdout
        assert out.exists()          # kept, not deleted
    finally:
        out.unlink()


def test_all_failed_csv_is_redone_not_skipped(roots, tmp_path):
    """A simurgh6 ECC job wrote a full CSV of exit_code=1 rows. That is a hole,
    not a result, and skipping over it is how it stays invisible."""
    out = REPO / "eval_results" / "g2_mlp_ret_stock__f0_ep00054600_pytest.csv"
    out.parent.mkdir(exist_ok=True)
    out.write_text(HDR + FAILED * 16)
    try:
        r = run({**roots, "FOLDS": "f0", "CELLS": "ret_stock", "TAG": "pytest"},
                ["mlp"])
        assert r.returncode == 0, r.stderr
        assert "has 0/16 usable rows (every pair failed) -- replacing" in r.stdout
        assert "1 submitted, 0 skipped" in r.stdout
        assert not out.exists()          # removed so the rerun can write it
    finally:
        if out.exists():
            out.unlink()


def test_unknown_fold_is_a_hard_error(roots):
    r = run({**roots, "FOLDS": "f7"}, ["mlp"])
    assert r.returncode != 0
    assert "unknown fold 'f7'" in r.stderr

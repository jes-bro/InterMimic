#!/usr/bin/env python3
"""eval_one.sh WALLTIME passthrough.

slurm_eval_curriculum.sh's header asks for 08:00:00. That covers a held-out-only
eval (~50 pairs) and nothing bigger; a 13-source arm's default body set is 286
pairs (29-43 h) and the job was killed at 8 h and RESUME=1'd repeatedly.
WALLTIME=<slurm time> must reach sbatch as a CLI --time (which overrides the
header), and an unset WALLTIME must leave the sbatch line exactly as it was.

The test replaces `sbatch` with a stub on PATH that echoes its arguments, so the
real submission line is exercised without a cluster.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_eval_one_walltime.py -q
"""
import os
import stat
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = "g3_bball7_geoall__f0"                      # any arm with an eval cfg


def _fake_sbatch(tmp_path):
    d = tmp_path / "bin"; d.mkdir()
    s = d / "sbatch"
    s.write_text('#!/bin/sh\necho "SBATCH_ARGV: $*"\n')
    s.chmod(s.stat().st_mode | stat.S_IEXEC)
    return str(d)


def _fake_ckpt(tmp_path):
    ck = tmp_path / f"smplx_teacher_{ARM}" / "nn" / "mimic_00020000.pth"
    ck.parent.mkdir(parents=True); ck.write_bytes(b"")
    return str(ck)


def _run(tmp_path, env):
    e = {**os.environ, **env, "PATH": _fake_sbatch(tmp_path) + os.pathsep + os.environ["PATH"],
         "OUT": str(tmp_path / "out.csv"), "EXCLUDE_NODES": "simurgh6,simurgh2"}
    return subprocess.run(["sh", "scripts/eval_one.sh", ARM, _fake_ckpt(tmp_path)],
                          cwd=REPO, env=e, capture_output=True, text=True)


def _sbatch_argv(stdout):
    return next(l for l in stdout.splitlines() if l.startswith("SBATCH_ARGV:"))


def test_walltime_reaches_sbatch_as_a_cli_time_flag(tmp_path):
    r = _run(tmp_path, {"WALLTIME": "2-00:00:00"})
    assert r.returncode == 0, r.stderr
    argv = _sbatch_argv(r.stdout)
    assert "--time=2-00:00:00" in argv
    assert "--exclude=simurgh6,simurgh2" in argv                 # the existing flag still passes
    assert argv.endswith("slurm_eval_curriculum.sh")
    assert "walltime   : 2-00:00:00" in r.stdout                 # recorded in the submission log


def test_unset_walltime_leaves_the_header_in_charge(tmp_path):
    r = _run(tmp_path, {})
    assert r.returncode == 0, r.stderr
    argv = _sbatch_argv(r.stdout)
    assert "--time" not in argv
    assert "   walltime   :" not in r.stdout         # the exact log line, not the word
                                                     # (pytest's tmp dir carries the test name)
    assert argv == "SBATCH_ARGV: --exclude=simurgh6,simurgh2 slurm_eval_curriculum.sh"

"""eval_local.sh resolves the plan through eval_one.sh (EMIT) and would run
slurm_eval_curriculum.sh inline with the same variables; it refuses to
overwrite an existing CSV. DRY=1 stops before running anything.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_eval_local.py -q
"""
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, env=None, cwd=REPO):
    return subprocess.run(["sh", "scripts/eval_local.sh", *args], cwd=cwd,
                          env={**os.environ, **(env or {})}, capture_output=True, text=True)


def _fake_ckpt(tmp_path, exp, name="mimic_00020000.pth"):
    ck = tmp_path / exp / "nn" / name
    ck.parent.mkdir(parents=True); ck.write_bytes(b"")
    return str(ck)


def test_dry_prints_the_same_plan_eval_one_resolves(tmp_path):
    ck = _fake_ckpt(tmp_path, "smplx_teacher_g3_bball7_geoall_nogate__f0")
    out = str(tmp_path / "res.csv")
    r = _run(["g3_bball7_geoall_nogate__f0+gatedscore", ck], env={"DRY": "1", "OUT": out})
    assert r.returncode == 0, r.stderr
    assert "omomo_eval_g3_bball7_geoall_nogate_gatedscore__f0.yaml" in r.stdout
    assert "intermimic.run / InterMimic" in r.stdout
    assert f"-> csv     : {out}" in r.stdout and "(DRY=1: not running)" in r.stdout


def test_student_plan_uses_the_student_path(tmp_path):
    ck = _fake_ckpt(tmp_path, "smplx_student_g3_act_xf_ret_nvadlr__f0", "mimic_00009000.pth")
    r = _run(["student_g3_act_xf_ret_nvadlr__f0", ck],
             env={"DRY": "1", "OUT": str(tmp_path / "s.csv"), "BODIES": "sub10 sub13 sub16"})
    assert r.returncode == 0, r.stderr
    assert "intermimic.run_distill / InterMimicDistillG3" in r.stdout
    assert "bodies     : sub10 sub13 sub16" in r.stdout


def test_refuses_to_overwrite_an_existing_csv(tmp_path):
    ck = _fake_ckpt(tmp_path, "smplx_teacher_g3_bball7_geoall__f0")
    out = tmp_path / "exists.csv"; out.write_text("body,source\nsub2,sub401\n")
    r = _run(["g3_bball7_geoall__f0", ck], env={"DRY": "1", "OUT": str(out)})
    assert r.returncode == 2 and "Refusing to overwrite" in r.stderr
    r = _run(["g3_bball7_geoall__f0", ck], env={"DRY": "1", "OUT": str(out), "RESUME": "1"})
    assert r.returncode == 0, r.stderr                        # RESUME fills in, never discards


def test_unknown_run_fails_loudly():
    r = _run(["g9_nope__f0"], env={"DRY": "1"})
    assert r.returncode != 0


def test_it_runs_the_slurm_script_inline_with_the_plan_vars():
    src = open(os.path.join(REPO, "scripts", "eval_local.sh")).read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "bash slurm_eval_curriculum.sh" in code and "sbatch" not in code   # inline, never queued
    for v in ("CHECKPOINT", "OUT", "ENV_YAML", "TRAIN_YAML", "SOURCES", "BODIES", "EVAL_ENTRY", "EVAL_TASK"):
        assert f'{v}="$' in src, v

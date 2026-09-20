"""The `_tfinal` OMOMO XF student = the Sep-16 OMOMO XF student distilled from the
FINAL teacher checkpoints. It must differ from the original in exactly: its
experiment name, and teacherPolicy (its own teacher dir). Anything else that
drifts would make the comparison "teachers + something", not "teachers".

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_tfinal_cfgs.py -q
"""
import os

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")
BASE = "omomo_xf_ret_nvadlr"


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _env(n):
    return _flat(yaml.safe_load(open(os.path.join(CFG, f"omomo_student_g3_{n}__f0.yaml"))))


def _train(n):
    return _flat(yaml.safe_load(open(os.path.join(RLG, f"omomo_student_g3_{n}__f0.yaml"))))


def test_env_differs_only_in_teacher_dir():
    b, a = _env(BASE), _env(f"{BASE}_tfinal")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"env.teacherPolicy"}
    assert a["env.teacherPolicy"] == "checkpoints/teachers/g3_omomo_tfinal"
    assert b["env.teacherPolicy"] == "checkpoints/teachers/g3_omomo"       # the Sep-16 dir, untouched


def test_train_cfg_differs_only_in_name():
    b, a = _train(BASE), _train(f"{BASE}_tfinal")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"params.config.full_experiment_name"}
    assert a["params.config.full_experiment_name"] == f"smplx_student_g3_{BASE}_tfinal__f0"


def test_launcher_points_at_tfinal_everything():
    s = open(os.path.join(ROOT, f"slurm_student_g3_{BASE}_tfinal__f0.sh")).read()
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_student_g3_{BASE}_tfinal__f0.yaml" in s
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_student_g3_{BASE}_tfinal__f0.yaml" in s
    assert f'--job-name="stu-g3_{BASE}_tfinal__f0"' in s
    assert "--out checkpoints/teachers/g3_omomo_tfinal" in s                 # header's collect step
    assert "--out checkpoints/teachers/g3_omomo\n" not in s                 # no stale pointer to the Sep-16 dir
    # every CODE mention of the base experiment name is the tfinal one (the header
    # comment may name the original launcher it was copied from)
    code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#") or l.startswith("#SBATCH"))
    assert f"{BASE}__f0" not in code.replace(f"{BASE}_tfinal__f0", "")


def test_runbook_has_the_tfinal_section():
    s = open(os.path.join(ROOT, "COLLAB_RUNBOOK_g3_students.md")).read()
    assert "g3-distill-tfinal" in s and "checkpoints/teachers/g3_omomo_tfinal" in s
    assert "NUM_ENVS=1024 sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_xf_ret_nvadlr_tfinal__f0.sh" in s

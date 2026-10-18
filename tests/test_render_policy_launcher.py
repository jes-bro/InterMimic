"""slurm_render_policy.sh: a STUDENT arm renders through the entry/task its eval cfg
names (intermimic.run_distill + InterMimicDistillG3/G4) from the student .pth alone,
and a teacher arm keeps the teacher path. The launcher runs no Python here; the
checks are the bash syntax and the resolution rules its text encodes.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_render_policy_launcher.py -q
"""
import os
import re
import subprocess

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO, "slurm_render_policy.sh")
CFG = os.path.join(REPO, "isaacgym", "src", "intermimic", "data", "cfg")
TXT = open(SCRIPT).read()
CODE = "\n".join(l for l in TXT.splitlines() if not l.lstrip().startswith("#"))


def test_bash_syntax():
    r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_train_cfg_follows_the_student_naming_rule():
    # check_eval_cfg.is_student: students are train/rlg/omomo_<arm>.yaml, teachers omomo_teacher_<arm>.yaml
    assert 'student_*) TRAIN="${TRAIN:-$CFGDIR/train/rlg/omomo_${ARM}.yaml}"' in CODE
    assert '*)         TRAIN="${TRAIN:-$CFGDIR/train/rlg/omomo_teacher_${ARM}.yaml}"' in CODE


def test_entry_and_task_come_from_the_eval_cfg_with_teacher_defaults():
    assert "c.get('evalEntry', 'intermimic.run')" in CODE
    assert "c.get('evalTask', 'InterMimic')" in CODE
    # the invocation uses them, and nothing hardcodes the teacher path any more
    assert 'python -u -m "$EVAL_ENTRY" --task "$EVAL_TASK"' in CODE
    assert "python -u -m intermimic.run --task InterMimic" not in CODE


def test_students_render_from_the_pth_alone():
    assert '[ "$EVAL_ENTRY" = intermimic.run_distill ] && SKIP_TEACHERS="DISTILL_SKIP_TEACHERS=1"' in CODE
    assert re.search(r'^env \$SKIP_TEACHERS RECORD_VIDEO="\$OUT"', CODE, flags=re.M)


def test_gen4_student_eval_cfgs_name_the_distill_entry_and_g4_task():
    for arm in ("student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0",
                "student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp_fullep__f0"):
        c = yaml.safe_load(open(os.path.join(CFG, f"omomo_eval_{arm}.yaml")))
        assert c["evalFor"] == [arm]
        assert c["evalEntry"] == "intermimic.run_distill" and c["evalTask"] == "InterMimicDistillG4"
        e = c["env"]
        assert e["stateInit"] == "Start" and e["enableEvaluation"] is True
        assert e["rolloutLength"] == 1000 and e["physicalBufferSize"] == 1 and e["numEnvs"] == 1024
        assert e["studentObsHorizons"] == [1, 4, 7, 10, 13, 16] and e["numObs"] == 9594
        assert e["teacherPolicy"] == "checkpoints/teachers/gen4_omomo_groups_msexp"
        assert ("minEpisodeFrames" in e) == arm.endswith("_fullep__f0")

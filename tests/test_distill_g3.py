"""Pin the g3 distillation path (utils/distill_g3.py + the task/builder wiring).

The task and builder import Isaac Gym / rl_games (unavailable locally), so the
arithmetic and manifest rules live in utils/distill_g3.py (stdlib + yaml only)
and are imported here by path; the wiring is checked by reading the sources.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_distill_g3.py -q
"""
import importlib.util
import os
import re

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
PKG = os.path.join(ROOT, "isaacgym", "src", "intermimic")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dg = _load(os.path.join(PKG, "utils", "distill_g3.py"), "distill_g3")


# ----------------------------------------------------------------- manifest
def _write_manifest(d, entries, touch=True):
    for e in entries:
        if touch:
            open(os.path.join(d, e["file"]), "wb").close()
    with open(os.path.join(d, dg.MANIFEST_NAME), "w") as fh:
        yaml.safe_dump({"teachers": entries}, fh)


def test_manifest_roundtrip(tmp_path):
    _write_manifest(tmp_path, [
        {"file": "sub2.pth", "sources": [2], "from": "x", "epoch": 12000},
        {"file": "bball7.pth", "sources": [401, 402, 404], "epoch": 9000},
    ])
    ents = dg.load_teacher_manifest(str(tmp_path))
    assert [e.file for e in ents] == ["sub2.pth", "bball7.pth"]
    assert ents[1].sources == [401, 402, 404]
    assert ents[0].epoch == 12000 and ents[0].origin == "x"


def test_manifest_missing_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="teachers.yaml"):
        dg.load_teacher_manifest(str(tmp_path))


def test_manifest_missing_file_is_loud(tmp_path):
    _write_manifest(tmp_path, [{"file": "sub2.pth", "sources": [2]}], touch=False)
    with pytest.raises(FileNotFoundError, match="sub2.pth"):
        dg.load_teacher_manifest(str(tmp_path))


def test_manifest_duplicate_source_is_loud(tmp_path):
    _write_manifest(tmp_path, [{"file": "a.pth", "sources": [5]},
                               {"file": "b.pth", "sources": [5, 6]}])
    with pytest.raises(ValueError, match="sub5 claimed by both"):
        dg.load_teacher_manifest(str(tmp_path))


def test_manifest_bad_sources_is_loud(tmp_path):
    _write_manifest(tmp_path, [{"file": "a.pth", "sources": ["sub5"]}])
    with pytest.raises(ValueError, match="non-negative ints"):
        dg.load_teacher_manifest(str(tmp_path))
    _write_manifest(tmp_path, [{"file": "a.pth", "sources": []}])
    with pytest.raises(ValueError, match="non-negative ints"):
        dg.load_teacher_manifest(str(tmp_path))


# ----------------------------------------------------------------- routing
def _entries():
    return [dg.TeacherEntry("sub2.pth", [2]), dg.TeacherEntry("sub5.pth", [5]),
            dg.TeacherEntry("bball7.pth", [401, 402])]


def test_lookup_routes_every_present_source():
    lookup, unused = dg.build_source_lookup(_entries(), [2, 401, 402, 2])
    assert lookup[2] == 0 and lookup[401] == 2 and lookup[402] == 2
    assert lookup[5] == 1            # defined even though no clip needs it
    assert lookup[3] == -1           # never served
    assert unused == ["sub5.pth"]    # reported, not an error


def test_lookup_refuses_uncovered_source():
    with pytest.raises(ValueError, match=r"no teacher: \['sub9'\]"):
        dg.build_source_lookup(_entries(), [2, 9])


# ----------------------------------------------------------------- obs widths
G3_H = [1, 4, 7, 10, 13, 16]


def test_student_width_matches_teacher_six_horizons():
    assert dg.student_obs_width(9594, G3_H, G3_H, False) == 9594


def test_student_width_four_horizons():
    assert dg.student_obs_width(9594, G3_H, [0, 1, 4, 16], False) == 4 * 1599


def test_student_width_refuses_betas_and_bad_division():
    with pytest.raises(ValueError, match="betas"):
        dg.student_obs_width(9594, G3_H, G3_H, True)
    with pytest.raises(ValueError, match="not a multiple"):
        dg.student_obs_width(9595, G3_H, G3_H, False)
    with pytest.raises(ValueError, match="studentObsHorizons"):
        dg.student_obs_width(9594, G3_H, [1, 1], False)


# ----------------------------------------------------------------- transformer tokens
def test_token_layout_defaults_reproduce_old_network():
    assert dg.token_layout(6396, 4, 1) == (1599, 4, 1)
    assert dg.token_layout(6524, 4, 1) == (1631, 4, 1)   # with betas, as before


def test_token_layout_six_tokens():
    assert dg.token_layout(9594, 6, 0) == (1599, 6, 0)


def test_token_layout_refusals():
    with pytest.raises(ValueError, match="not divisible"):
        dg.token_layout(9594, 4, 1)
    with pytest.raises(ValueError, match="out of range"):
        dg.token_layout(9594, 6, 6)
    with pytest.raises(ValueError, match="positive int"):
        dg.token_layout(9594, 0, 0)


# ----------------------------------------------------------------- wiring (source checks)
def _src(*parts):
    return open(os.path.join(PKG, *parts)).read()


def test_builder_no_longer_hardcodes_four_tokens():
    s = _src("learning", "intermimic_transformer_network_builder.py")
    assert "view(obs.shape[0], 4, -1)" not in s
    assert "self.encoder(a_out)[1]" not in s
    assert "view(obs.shape[0], self._num_tokens, -1)" in s
    assert "self.encoder(a_out)[self._readout_token]" in s
    assert "token_layout(" in s


def test_task_is_registered_and_whitelisted():
    assert "from ..env.tasks.intermimic_distill_g3 import InterMimicDistillG3" in _src("utils", "parse_task.py")
    im = _src("env", "tasks", "intermimic.py")
    keys = re.search(r"KNOWN_ENV_KEYS = frozenset\(\{(.*?)\}\)", im, re.S).group(1)
    assert "'studentObsHorizons'" in keys, "new env key must be whitelisted or startup refuses the cfg"


def test_task_keeps_parent_loop():
    """The whole point: no step()/post-loop re-implementation, teacher routed by SOURCE."""
    s = _src("env", "tasks", "intermimic_distill_g3.py")
    assert "def step(" not in s
    assert "super().post_physics_step()" in s
    assert "self.source_subject_index[self.data_id]" in s
    assert "hoi_data_retarget" not in s        # one reference, shared by teacher and student
    assert "motion_file_retarget" not in s

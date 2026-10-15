"""InterMimicDistillG4: the gen4 object-group student's task. Pins the pure
helpers (utils/distill_g4.py: object-keyed manifest + object lookup), the task's
shape (a g3 subclass overriding ONLY the three teacher seams + the per-env
routing), the seams' presence in g3, and the registration in parse_task.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_distill_g4.py -q
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


def _src(*parts):
    with open(os.path.join(PKG, *parts)) as fh:
        return fh.read()


dg4 = _load(os.path.join(PKG, "utils", "distill_g4.py"), "distill_g4")

GROUPS = {
    "boxes.pth": ["largebox", "smallbox", "plasticbox", "suitcase", "trashcan", "monitor"],
    "stands.pth": ["tripod", "clothesstand", "floorlamp"],
    "chairs.pth": ["whitechair", "woodchair"],
    "tables.pth": ["largetable", "smalltable"],
}


# ----------------------------------------------------------------- manifest
def _write(d, entries, touch=True):
    for e in entries:
        if touch:
            open(os.path.join(d, e["file"]), "wb").close()
    with open(os.path.join(d, dg4.MANIFEST_NAME), "w") as fh:
        yaml.safe_dump({"teachers": entries}, fh)


def _four(d):
    _write(d, [{"file": f, "objects": objs, "from": f"x/{f}", "epoch": 35000 + i}
               for i, (f, objs) in enumerate(GROUPS.items())])


def test_manifest_roundtrip(tmp_path):
    _four(tmp_path)
    ents = dg4.load_object_manifest(str(tmp_path))
    assert [e.file for e in ents] == list(GROUPS)
    assert ents[2].objects == ["whitechair", "woodchair"]
    assert ents[0].epoch == 35000 and ents[0].origin == "x/boxes.pth"
    assert "objects=" in repr(ents[0])


def test_manifest_missing_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="teachers.yaml"):
        dg4.load_object_manifest(str(tmp_path))


def test_manifest_missing_file_is_loud(tmp_path):
    _write(tmp_path, [{"file": "chairs.pth", "objects": ["woodchair"]}], touch=False)
    with pytest.raises(FileNotFoundError, match="chairs.pth"):
        dg4.load_object_manifest(str(tmp_path))


def test_manifest_refuses_a_g3_sources_manifest(tmp_path):
    """The wrong-task trap: a per-source manifest must not route by object."""
    _write(tmp_path, [{"file": "sub2.pth", "sources": [2]}])
    with pytest.raises(ValueError, match="keyed by 'sources'.*g3"):
        dg4.load_object_manifest(str(tmp_path))


def test_manifest_refuses_duplicate_object(tmp_path):
    _write(tmp_path, [{"file": "chairs.pth", "objects": ["woodchair"]},
                      {"file": "boxes.pth", "objects": ["largebox", "woodchair"]}])
    with pytest.raises(ValueError, match="object woodchair claimed by both chairs.pth and boxes.pth"):
        dg4.load_object_manifest(str(tmp_path))


@pytest.mark.parametrize("bad", [[], ["wood_chair"], [""], [3], "woodchair"])
def test_manifest_refuses_bad_object_lists(tmp_path, bad):
    _write(tmp_path, [{"file": "chairs.pth", "objects": bad}])
    with pytest.raises(ValueError, match="'objects' must be"):
        dg4.load_object_manifest(str(tmp_path))


def test_manifest_refuses_entry_without_objects(tmp_path):
    _write(tmp_path, [{"file": "chairs.pth"}])
    with pytest.raises(ValueError, match="needs 'objects'"):
        dg4.load_object_manifest(str(tmp_path))


# ----------------------------------------------------------------- lookup
def _entries():
    return [dg4.ObjectTeacherEntry(f, objs) for f, objs in GROUPS.items()]


def test_lookup_aligns_with_the_tasks_sorted_object_list():
    names = sorted(o for objs in GROUPS.values() for o in objs)       # what intermimic.py builds
    lookup, unused = dg4.build_object_lookup(_entries(), names)
    assert len(lookup) == 13 and unused == []
    files = list(GROUPS)
    for o, i in zip(names, lookup):
        assert o in GROUPS[files[i]], (o, files[i])
    assert lookup[names.index("woodchair")] == files.index("chairs.pth")
    assert lookup[names.index("monitor")] == files.index("boxes.pth")


def test_lookup_reports_unused_teachers_on_a_filtered_dataset():
    lookup, unused = dg4.build_object_lookup(_entries(), ["whitechair", "woodchair"])
    assert lookup == [2, 2] and unused == ["boxes.pth", "stands.pth", "tables.pth"]


def test_lookup_refuses_an_uncovered_object():
    with pytest.raises(ValueError, match=r"no teacher: \['hoop'\]"):
        dg4.build_object_lookup(_entries(), ["woodchair", "hoop"])


# ----------------------------------------------------------------- task shape
def test_task_is_a_g3_subclass_overriding_only_the_seams():
    s = _src("env", "tasks", "intermimic_distill_g4.py")
    assert "class InterMimicDistillG4(InterMimicDistillG3)" in s
    defs = re.findall(r"^    def (\w+)\(", s, flags=re.M)
    assert sorted(defs) == ["_build_teacher_routing", "_describe_teacher",
                            "_load_teacher_entries", "_refresh_teacher_indices"]
    assert "def __init__" not in s and "def step(" not in s and "def post_physics_step" not in s
    assert "self.object_id[self.data_id]" in s                 # routed by the clip's OBJECT
    assert "source_subject_index" not in s
    assert "load_object_manifest(teacher_dir)" in s


def test_g3_exposes_the_seams_and_routes_by_source_unchanged():
    s = _src("env", "tasks", "intermimic_distill_g3.py")
    for seam in ("def _load_teacher_entries(self, teacher_dir)", "def _describe_teacher(self, e)",
                 "def _build_teacher_routing(self)", "def _refresh_teacher_indices(self)"):
        assert seam in s, seam
    assert "self._teacher_entries = self._load_teacher_entries(teacher_dir)" in s
    assert "self._build_teacher_routing()" in s
    assert "self.source_subject_index[self.data_id]" in s      # g3 still routes by source
    assert "return load_teacher_manifest(teacher_dir)" in s


def test_task_is_registered():
    assert "from ..env.tasks.intermimic_distill_g4 import InterMimicDistillG4" in _src("utils", "parse_task.py")

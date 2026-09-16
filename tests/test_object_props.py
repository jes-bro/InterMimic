"""utils/object_props.py: the restitution solve and the props-file validation.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_object_props.py -q
"""
import importlib.util
import os

import pytest
import yaml

PATH = os.path.join(os.path.dirname(__file__), "..", "isaacgym", "src", "intermimic", "utils", "object_props.py")
spec = importlib.util.spec_from_file_location("object_props", PATH)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def test_restitution_solve_reproduces_each_teacher_floor_average():
    # teacher pairs from the three activity arms, student plane 0.7
    assert op.teacher_pair_to_student_object(0.85, 0.85, 0.7) == pytest.approx(1.0)   # bball
    assert op.teacher_pair_to_student_object(0.65, 0.65, 0.7) == pytest.approx(0.6)   # soccer
    assert op.teacher_pair_to_student_object(0.05, 0.70, 0.7) == pytest.approx(0.05)  # cpr
    for obj_r, plane_r in [(0.85, 0.85), (0.65, 0.65), (0.05, 0.7)]:
        r = op.teacher_pair_to_student_object(obj_r, plane_r, 0.7)
        assert (r + 0.7) / 2 == pytest.approx((obj_r + plane_r) / 2)


def test_restitution_solve_refuses_infeasible_plane():
    with pytest.raises(ValueError, match="outside"):
        op.teacher_pair_to_student_object(0.05, 0.7, 0.85)   # cpr would need < 0
    with pytest.raises(ValueError, match="outside"):
        op.teacher_pair_to_student_object(0.85, 0.85, 0.5)   # bball would need > 1


def test_validate_props():
    good = {"ball": {"mass": 0.624, "restitution": 1.0}, "manikin": {"mass": 3.7, "restitution": 0.05}}
    assert op.validate_props(good, ["ball", "manikin"]) is good
    assert op.validate_props(good, ["ball"]) is good           # extra entries allowed
    with pytest.raises(ValueError, match=r"no entry: \['cup'\]"):
        op.validate_props(good, ["ball", "cup"])
    with pytest.raises(ValueError, match="mass must be > 0"):
        op.validate_props({"b": {"mass": 0, "restitution": 0.5}}, ["b"])
    with pytest.raises(ValueError, match=r"restitution must be in \[0, 1\]"):
        op.validate_props({"b": {"mass": 1, "restitution": 1.5}}, ["b"])
    with pytest.raises(ValueError, match="needs 'mass' and 'restitution'"):
        op.validate_props({"b": {"mass": 1}}, ["b"])
    with pytest.raises(ValueError, match="non-empty"):
        op.validate_props({}, [])


def test_load_object_props(tmp_path):
    p = tmp_path / "props.yaml"
    yaml.safe_dump({"objects": {"ball": {"mass": 0.43, "restitution": 0.6}}}, open(p, "w"))
    assert op.load_object_props(str(p), ["ball"])["ball"]["mass"] == 0.43
    with pytest.raises(ValueError, match="no entry"):
        op.load_object_props(str(p), ["ball", "other"])

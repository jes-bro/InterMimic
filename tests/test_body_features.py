"""utils/body_features.py: MJCF bone offsets on a real subject rig, and the
twin-env pairing rules.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_body_features.py -q
"""
import importlib.util
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH = os.path.join(ROOT, "isaacgym", "src", "intermimic", "utils", "body_features.py")
spec = importlib.util.spec_from_file_location("body_features", PATH)
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

SMPLX = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "assets", "smplx")


def test_real_rig_has_52_bodies_in_task_order():
    offs = bf.mjcf_body_offsets(os.path.join(SMPLX, "smplx_omomo_sub100.xml"))
    names = [n for n, _ in offs]
    assert len(names) == 52
    assert names[0] == "Pelvis" and names[17] == "L_Wrist" and names[36] == "R_Wrist"   # the order the task relies on
    assert offs[0][1] == (0.0, 0.0, 0.0)


def test_feature_matrix_differs_between_bodies_and_has_156_columns():
    paths = [os.path.join(SMPLX, f"smplx_omomo_sub{s}.xml") for s in (100, 101, 102)]
    m = bf.body_feature_matrix(paths)
    assert len(m) == 3 and all(len(r) == 156 for r in m)
    assert m[0] != m[1] and m[1] != m[2]           # distinct bodies -> distinct vectors


def test_feature_matrix_refuses_order_mismatch(tmp_path):
    a = tmp_path / "a.xml"; b = tmp_path / "b.xml"
    a.write_text('<mujoco><worldbody><body name="Pelvis" pos="0 0 0"><body name="L_Hip" pos="0 0.1 -0.1"/></body></worldbody></mujoco>')
    b.write_text('<mujoco><worldbody><body name="Pelvis" pos="0 0 0"><body name="R_Hip" pos="0 -0.1 -0.1"/></body></worldbody></mujoco>')
    assert bf.body_feature_matrix([str(a)], expected_bodies=2) == [[0.0, 0.0, 0.0, 0.0, 0.1, -0.1]]
    with pytest.raises(ValueError, match="body order differs"):
        bf.body_feature_matrix([str(a), str(b)], expected_bodies=2)
    with pytest.raises(ValueError, match="expected 52"):
        bf.body_feature_matrix([str(a)])


def test_twin_partners_same_object_different_body():
    n_env, n_obj, n_body = 1024, 13, 43
    partner = bf.twin_partners(n_env, n_obj, n_body)
    pairs = bf.twin_pairs(partner)
    assert len(pairs) > 400                                   # most envs paired
    for a, b in pairs:
        assert a % n_obj == b % n_obj                         # same object bucket
        assert a % n_body != b % n_body                       # different body
        assert partner[a] == b and partner[b] == a
    assert all(partner[e] == -1 or partner[partner[e]] == e for e in range(n_env))
    # each env in at most one pair
    seen = [e for p in pairs for e in p]
    assert len(seen) == len(set(seen))


def test_twin_partners_activity_sizes_and_refusal():
    partner = bf.twin_partners(1024, 132, 43)                 # act student: 132 objects
    pairs = bf.twin_pairs(partner)
    assert len(pairs) >= 300
    for a, b in pairs:
        assert a % 132 == b % 132 and a % 43 != b % 43
    with pytest.raises(ValueError, match="share body"):
        bf.twin_partners(64, 8, 4)                            # 8 objects, 4 bodies: e and e+8 share a body

"""scripts/merge_activity_data.py on a fixture: two activity arms with cfgs,
motion dirs and body-major retarget trees -> one flat motion dir, one merged
tree, one props file with the solved restitutions and the union dataSub.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_merge_activity_data.py -q
"""
import importlib.util
import os

import pytest
import yaml

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "merge_activity_data.py")
spec = importlib.util.spec_from_file_location("merge_activity_data", SCRIPT)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)

BODIES = ["sub1", "sub2"]


def _arm(root, cfg_dir, name, clips, subs, mass, obj_r, plane_r, motion="m", retarget="r"):
    mdir = os.path.join(root, f"{motion}_{name}")
    os.makedirs(mdir)
    for c in clips:
        open(os.path.join(mdir, c), "wb").close()
    open(os.path.join(mdir, "notes.txt"), "w").close()          # non-.pt: ignored
    rdir = os.path.join(root, f"{retarget}_{name}")
    for b in BODIES:
        os.makedirs(os.path.join(rdir, b))
        for c in clips:
            open(os.path.join(rdir, b, c), "wb").close()
    env = {"motion_file": os.path.relpath(mdir, root), "retargetedMotionDir": os.path.relpath(rdir, root),
           "objectMass": mass, "plane": {"restitution": plane_r}, "dataSub": subs}
    if obj_r is not None:
        env["objectShapeProps"] = {"restitution": obj_r}
    os.makedirs(cfg_dir, exist_ok=True)
    yaml.safe_dump({"env": env}, open(os.path.join(cfg_dir, ma.ARM_CFG.format(name=name)), "w"))


@pytest.fixture
def fx(tmp_path):
    root, cfg_dir = str(tmp_path), str(tmp_path / "cfg")
    _arm(root, cfg_dir, "bball7", ["sub401_ballA_000.pt", "sub402_ballB_000.pt", "sub999_ballZ_000.pt"],
         ["sub401", "sub402"], 0.624, 0.85, 0.85)                       # sub999 outside dataSub
    _arm(root, cfg_dir, "cpr13", ["sub509_manikin_000.pt", "sub510_manikin_001.pt"],
         ["sub509", "sub510"], 3.7, None, 0.7)                          # no objectShapeProps -> 0.05
    student = tmp_path / "student.yaml"
    yaml.safe_dump({"env": {"subjectBodies": BODIES}}, open(student, "w"))
    return root, cfg_dir, str(student)


def _args(root, cfg_dir, student, extra=()):
    return ["--arms", "bball7", "cpr13", "--out-motion", os.path.join(root, "act"),
            "--out-retarget", os.path.join(root, "act_tree"), "--props-out", os.path.join(root, "props.yaml"),
            "--bodies-from", student, "--student-plane-restitution", "0.7",
            "--cfg-dir", cfg_dir, "--repo", root, *extra]


def test_plan_props_and_union(fx):
    root, cfg_dir, student = fx
    links, props, data_sub = ma.main(_args(root, cfg_dir, student, ["--dry-run"]))
    assert [f for _, f in links] == ["sub401_ballA_000.pt", "sub402_ballB_000.pt",
                                     "sub509_manikin_000.pt", "sub510_manikin_001.pt"]
    assert props["ballA"] == {"mass": 0.624, "restitution": 1.0, "arm": "bball7"}
    assert props["manikin"] == {"mass": 3.7, "restitution": 0.05, "arm": "cpr13"}
    assert data_sub == ["sub401", "sub402", "sub509", "sub510"]
    assert not os.path.exists(os.path.join(root, "act"))                 # dry run wrote nothing


def test_writes_links_tree_and_props(fx):
    root, cfg_dir, student = fx
    ma.main(_args(root, cfg_dir, student))
    act = os.path.join(root, "act")
    assert sorted(os.listdir(act)) == ["sub401_ballA_000.pt", "sub402_ballB_000.pt",
                                       "sub509_manikin_000.pt", "sub510_manikin_001.pt"]
    assert os.path.islink(os.path.join(act, "sub509_manikin_000.pt"))
    # merge_retarget_trees links each source tree WHOLE (it does not know dataSub),
    # so the tree may hold clips the flat dir filtered out (sub999 here). The task
    # needs motion clips SUBSET-OF tree clips per body; extra tree files are inert.
    for b in BODIES:
        tree = set(os.listdir(os.path.join(root, "act_tree", b)))
        assert set(os.listdir(act)) <= tree
        assert "sub999_ballZ_000.pt" in tree and "sub999_ballZ_000.pt" not in os.listdir(act)
    doc = yaml.safe_load(open(os.path.join(root, "props.yaml")))
    assert doc["student_plane_restitution"] == 0.7
    assert doc["dataSub"] == ["sub401", "sub402", "sub509", "sub510"]
    assert set(doc["objects"]) == {"ballA", "ballB", "manikin"}
    with pytest.raises(SystemExit, match="not empty"):
        ma.main(_args(root, cfg_dir, student))


def test_object_name_collision_refused(tmp_path):
    root, cfg_dir = str(tmp_path), str(tmp_path / "cfg")
    _arm(root, cfg_dir, "bball7", ["sub401_ball_000.pt"], ["sub401"], 0.624, 0.85, 0.85)
    _arm(root, cfg_dir, "soccer15", ["sub480_ball_000.pt"], ["sub480"], 0.43, 0.65, 0.65)
    student = tmp_path / "s.yaml"
    yaml.safe_dump({"env": {"subjectBodies": BODIES}}, open(student, "w"))
    with pytest.raises(SystemExit, match="object name 'ball' is used by both"):
        ma.main(["--arms", "bball7", "soccer15", "--out-motion", os.path.join(root, "a"),
                 "--out-retarget", os.path.join(root, "t"), "--props-out", os.path.join(root, "p.yaml"),
                 "--bodies-from", str(student), "--student-plane-restitution", "0.7",
                 "--cfg-dir", cfg_dir, "--repo", root, "--dry-run"])


def test_infeasible_plane_refused(fx):
    root, cfg_dir, student = fx
    args = _args(root, cfg_dir, student, ["--dry-run"])
    args[args.index("0.7")] = "0.85"           # cpr's 0.05/0.7 pair would need object < 0
    with pytest.raises(ValueError, match="outside"):
        ma.main(args)

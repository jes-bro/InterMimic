"""collect_g3_teachers.py on a fixture checkpoint tree: latest-snapshot rule,
sub2's irregular dir, activity sources read from the arm's cfg, partial-set
refusal, manifest contents.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_collect_g3_teachers.py -q
"""
import importlib.util
import os

import pytest
import yaml

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "collect_g3_teachers.py")
spec = importlib.util.spec_from_file_location("collect_g3_teachers", SCRIPT)
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)


@pytest.fixture(autouse=True)
def _no_torch_epoch(monkeypatch):
    """Fixture files are bytes, not checkpoints: default to 'no epoch inside' so
    the filename rule applies; individual tests override with a real map."""
    monkeypatch.setattr(cg, "epoch_inside", lambda p: None)


def test_epoch_inside_beats_filename_number(tmp_path, monkeypatch):
    """A collaborator's mimic.pth at 67,500 epochs next to mimic_00030000.pth:
    the inside epoch must win (2026-09-16 sub2 export)."""
    _ckpt(tmp_path, "smplx_teacher_g3_omomo_geoall__f0", ["mimic.pth", "mimic_00030000.pth"])
    inside = {"mimic.pth": 67500, "mimic_00030000.pth": 30000}
    monkeypatch.setattr(cg, "epoch_inside", lambda p: inside[os.path.basename(p)])
    p, ep = cg.latest_ckpt(os.path.join(tmp_path, "smplx_teacher_g3_omomo_geoall__f0", "nn"))
    assert p.endswith("/mimic.pth") and ep == 67500


def test_only_mimic_pth_is_loaded(tmp_path, monkeypatch):
    """A week-long run has ~70 snapshots; only mimic.pth needs opening (the
    numbered files carry their epoch in the name). Numbered ones must win when
    they are higher, and mimic.pth when it is."""
    nn = os.path.join(tmp_path, "smplx_teacher_g3_omomo_geoall_srchalf6__f0", "nn")
    _ckpt(tmp_path, "smplx_teacher_g3_omomo_geoall_srchalf6__f0",
          ["mimic.pth"] + [f"mimic_{e:08d}.pth" for e in range(500, 40001, 500)])
    loads = []
    monkeypatch.setattr(cg, "epoch_inside", lambda p: loads.append(p) or 40000)
    p, ep = cg.latest_ckpt(nn)
    assert loads == [os.path.join(nn, "mimic.pth")]          # exactly one load
    assert (p.endswith("mimic_00040000.pth") or p.endswith("mimic.pth")) and ep == 40000
    monkeypatch.setattr(cg, "epoch_inside", lambda p: 40500)   # mimic.pth newer than any snapshot
    assert cg.latest_ckpt(nn) == (os.path.join(nn, "mimic.pth"), 40500)
    monkeypatch.setattr(cg, "epoch_inside", lambda p: 12000)   # mimic.pth stale
    assert cg.latest_ckpt(nn) == (os.path.join(nn, "mimic_00040000.pth"), 40000)


def _ckpt(root, exp, names):
    d = os.path.join(root, exp, "nn")
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(n.encode())


def _cfg(cfg_dir, name, subs):
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, cg.ACT_CFG.format(name=name)), "w") as fh:
        yaml.safe_dump({"env": {"dataSub": subs}}, fh)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "checkpoints"
    _ckpt(root, "smplx_teacher_g3_omomo_geoall__f0", ["mimic.pth", "mimic_00012000.pth", "mimic_00009500.pth"])  # sub2
    _ckpt(root, "smplx_teacher_g3_omomo_geoall_src5__f0", ["mimic.pth"])                                       # no snapshots
    _ckpt(root, "smplx_teacher_g3_bball7_geoall__f0", ["mimic_00008000.pth"])
    cfg_dir = tmp_path / "cfg"
    _cfg(cfg_dir, "bball7", ["sub401", "sub402", "sub458"])
    return str(root), str(cfg_dir), str(tmp_path / "out")


def test_plan_picks_latest_and_maps_sub2(tree):
    root, cfg_dir, _ = tree
    plan = cg.plan_teachers(root, [2, 5], [], cfg_dir)
    by = {f: (srcs, ck, ep) for f, srcs, ck, ep in plan}
    assert by["sub2.pth"][0] == [2] and by["sub2.pth"][1].endswith("geoall__f0/nn/mimic_00012000.pth")
    assert by["sub2.pth"][2] == 12000
    assert by["sub5.pth"][1].endswith("src5__f0/nn/mimic.pth") and by["sub5.pth"][2] is None


def test_activity_sources_come_from_cfg(tree):
    root, cfg_dir, _ = tree
    plan = cg.plan_teachers(root, [], ["bball7"], cfg_dir)
    assert plan == [("bball7.pth", [401, 402, 458],
                     os.path.join(root, "smplx_teacher_g3_bball7_geoall__f0", "nn", "mimic_00008000.pth"), 8000)]


def test_omomo_arms_multi_source_teachers(tree):
    root, cfg_dir, _ = tree
    _ckpt(root, "smplx_teacher_g3_omomo_geoall_srchalf7__f0", ["mimic_00020000.pth"])
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, cg.OMOMO_ARM_CFG.format(name="srchalf7")), "w") as fh:
        yaml.safe_dump({"env": {"dataSub": ["sub2", "sub6", "sub7"]}}, fh)
    plan = cg.plan_teachers(root, [], [], cfg_dir, omomo_arms=["srchalf7"])
    assert plan == [("srchalf7.pth", [2, 6, 7],
                     os.path.join(root, "smplx_teacher_g3_omomo_geoall_srchalf7__f0", "nn", "mimic_00020000.pth"), 20000)]
    with pytest.raises(SystemExit, match="sub2 owned by both"):
        cg.plan_teachers(root, [2], [], cfg_dir, omomo_arms=["srchalf7"])


def test_partial_set_refused(tree):
    root, cfg_dir, _ = tree
    with pytest.raises(SystemExit, match=r"sub9: .*has no mimic"):
        cg.plan_teachers(root, [2, 9], [], cfg_dir)
    with pytest.raises(SystemExit, match="no env cfg for arm 'soccer15'"):
        cg.plan_teachers(root, [], ["soccer15"], cfg_dir)


def test_write_manifest_and_refuse_overwrite(tree):
    root, cfg_dir, out = tree
    cg.main(["--omomo-sources", "2", "5", "--activities", "bball7", "--out", out,
             "--root", root, "--cfg-dir", cfg_dir])
    m = yaml.safe_load(open(os.path.join(out, "teachers.yaml")))["teachers"]
    assert [e["file"] for e in m] == ["sub2.pth", "sub5.pth", "bball7.pth"]
    assert m[2]["sources"] == [401, 402, 458] and m[2]["epoch"] == 8000
    assert open(os.path.join(out, "sub2.pth"), "rb").read() == b"mimic_00012000.pth"
    with pytest.raises(SystemExit, match="never overwrite"):
        cg.main(["--omomo-sources", "2", "--out", out, "--root", root, "--cfg-dir", cfg_dir])


def test_dry_run_writes_nothing(tree):
    root, cfg_dir, out = tree
    cg.main(["--omomo-sources", "2", "--out", out, "--root", root, "--cfg-dir", cfg_dir, "--dry-run"])
    assert not os.path.exists(out)

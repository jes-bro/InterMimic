#!/usr/bin/env python3
"""The bball7 multi-subject pipeline, on fixtures small enough to read.

Covers the pieces that decide what lands in the training set:
  * cari4d_bball7_manifest.py: latest export wins, excludes honoured, the
    sub4xx ids and per-clip object tokens, clip indices per subject, mesh
    disambiguation from meta.json, extraction of exactly the needed files
  * cari4d_subject_betas.py: one mean vector per subject, over all its clips
  * cari4d_to_interact.py --betas-npz: the shared body overrides the clip's own
  * relabel_contact_human.radius_from_mesh: half the largest extent
  * utils/object_mass.density_for_mass: mass / volume on a known solid
  * the task's objectDensity/objectMass exclusivity, pinned in the source

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_cari4d_bball7_pipeline.py -v
"""
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, SCRIPTS)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


manifest = _load("cari4d_bball7_manifest", "scripts/cari4d_bball7_manifest.py")
subject_betas = _load("cari4d_subject_betas", "scripts/cari4d_subject_betas.py")
relabel = _load("relabel_contact_human", "scripts/relabel_contact_human.py")


# ---- fixtures: a fake reconstruction tarball --------------------------------
def make_bundle(T, betas):
    return {"gt": {}, "in": {},
            "pr": {"smpl_pose": torch.randn(T, 72), "smpl_t": torch.randn(T, 3),
                   "betas": torch.tensor(betas, dtype=torch.float32).repeat(T, 1),
                   "pose_abs": torch.eye(4).repeat(T, 1, 1)}}


def make_tarball(dirpath, clip, stamp, T, betas, meshes=("036",), solved=None, gender="male",
                 lo=10):
    """<dir>/<clip>-recon-<stamp>.tar.gz laid out like a real export."""
    name = f"{clip}-recon-{stamp}.tar.gz"
    path = os.path.join(dirpath, name)
    with tarfile.open(path, "w:gz") as tf:
        def add(rel, data):
            info = tarfile.TarInfo(f"{clip}-recon/{rel}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        buf = io.BytesIO(); torch.save(make_bundle(T, betas), buf)
        add(f"output/opt/exp/{clip}.pth", buf.getvalue())
        for m in meshes:
            add(f"work/{clip}/meshes-metric/{clip}_{m}_align.obj",
                b"v 0 0 0\nv 0.24 0 0\nv 0 0.2 0\nv 0 0 0.22\n")
        meta = {"clip": clip, "take": "take_x", "gender": gender, "lo": lo, "hi": lo + T - 1,
                "n_frames": T, "stages": {"solve": {"state": "done",
                                                    "metric_mesh": f"/x/{clip}_{solved or meshes[0]}_align.obj"}}}
        add("meta.json", json.dumps(meta).encode())
        add("gender.txt", (gender + "\n").encode())
    return path


@pytest.fixture
def exports(tmp_path):
    d = tmp_path / "dl"; d.mkdir()
    make_tarball(d, "Date03_Sub01_bball_rev003b", "20260908-050629", 107, [0.8] * 10)
    make_tarball(d, "Date03_Sub01_bball_rev003b", "20260912-013050", 68, [0.9] * 10)   # latest
    make_tarball(d, "Date03_Sub01_bball_rev009c", "20260912-014032", 70, [0.7] * 10)
    make_tarball(d, "Date08_Sub12_bball_t012a", "20260912-054603", 118, [0.1] * 10,
                 meshes=("024", "141"), solved="141")
    make_tarball(d, "Date06_Sub04_bball_t014bt", "20260912-030000", 48, [1.6] * 10)  # excluded
    make_tarball(d, "Date07_Sub05_soccer_t004c", "20260907-200008", 333, [0.6] * 10)  # not bball
    return d


# ---- manifest ---------------------------------------------------------------
def test_object_token_and_subject_id():
    assert manifest.object_token("Date03_Sub01_bball_rev003b") == "bballd03s01rev003b"
    assert manifest.object_token("Date08_Sub09_bball_t022as01s02") == "bballd08s09t022as01s02"
    assert manifest.subject_id("Date13_Sub58_bball_t001e") == 458
    assert "_" not in manifest.object_token("Date06_Sub04_bball_t012a")
    with pytest.raises(ValueError):
        manifest.object_token("weird_name")


def test_manifest_rows(exports, tmp_path):
    out = tmp_path / "bundles"
    rc = manifest.main([str(exports), "--out-dir", str(out),
                        "--exclude", "Date06_Sub04_bball_t014bt"])
    assert rc == 0
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    clips = [r["clip"] for r in rows]
    assert clips == ["Date03_Sub01_bball_rev003b", "Date03_Sub01_bball_rev009c",
                     "Date08_Sub12_bball_t012a"]                       # soccer + excluded gone
    r3 = rows[0]
    assert r3["export"].endswith("20260912-013050.tar.gz")          # latest export
    assert r3["n_frames"] == "68" and r3["subject_id"] == "401" and r3["clip_idx"] == "000"
    assert rows[1]["clip_idx"] == "001"                                # per-subject counter
    assert rows[2]["subject_id"] == "412" and rows[2]["clip_idx"] == "000"
    assert rows[2]["mesh"].endswith("_141_align.obj")                  # the solved mesh, not 024
    assert rows[0]["object"] == "bballd03s01rev003b"
    # extraction: bundle + mesh + meta + gender per clip, and the 68-frame bundle
    for r in rows:
        assert (out / r["bundle"]).is_file() and (out / r["mesh"]).is_file()
        assert (out / r["clip"] / "meta.json").is_file()
    b = torch.load(out / r3["bundle"], weights_only=False)
    assert b["pr"]["smpl_pose"].shape[0] == 68


def test_manifest_refuses_unknown_exclude(exports, tmp_path):
    with pytest.raises(SystemExit):
        manifest.main([str(exports), "--out-dir", str(tmp_path / "b"), "--exclude", "Date99_Sub99_bball_x"])


def test_manifest_two_meshes_without_solve_record(tmp_path):
    d = tmp_path / "dl"; d.mkdir()
    make_tarball(d, "Date03_Sub01_bball_rev003b", "20260912-013050", 10, [0.9] * 10,
                 meshes=("024", "141"), solved="999")     # meta names neither
    with pytest.raises(ValueError):
        manifest.build_rows(manifest.scan(str(d), "bball"), set(), None)


# ---- subject betas ----------------------------------------------------------
def test_subject_betas_mean(exports, tmp_path):
    out = tmp_path / "bundles"
    manifest.main([str(exports), "--out-dir", str(out), "--exclude", "Date06_Sub04_bball_t014bt"])
    stats = subject_betas.subject_means(str(out / "manifest.csv"), str(out))
    assert set(stats) == {"sub401", "sub412"}
    assert stats["sub401"]["n_clips"] == 2
    np.testing.assert_allclose(stats["sub401"]["mean"], [0.8] * 10, atol=1e-6)   # mean(0.9, 0.7)
    assert stats["sub401"]["spread"] == pytest.approx(np.linalg.norm([0.1] * 10), abs=1e-6)
    np.testing.assert_allclose(stats["sub412"]["mean"], [0.1] * 10, atol=1e-6)
    npz = tmp_path / "b.npz"
    subject_betas.main(["--manifest", str(out / "manifest.csv"), "--bundles-root", str(out),
                        "--out", str(npz)])
    store = np.load(npz)
    assert sorted(store.files) == ["sub401", "sub412"] and store["sub401"].shape == (10,)


# ---- adapter override -------------------------------------------------------
def test_adapter_uses_shared_betas(exports, tmp_path):
    out = tmp_path / "bundles"
    manifest.main([str(exports), "--out-dir", str(out), "--exclude", "Date06_Sub04_bball_t014bt"])
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    r = rows[0]
    npz = tmp_path / "shared.npz"
    np.savez(npz, sub401=np.full(10, 0.55, dtype=np.float32))
    interact = tmp_path / "InterAct"; (interact / "simulation").mkdir(parents=True)
    (interact / "simulation" / "interact2mimic.py").write_text("# stub\n")
    cmd = [sys.executable, os.path.join(SCRIPTS, "cari4d_to_interact.py"),
           "--bundle", str(out / r["bundle"]), "--mesh", str(out / r["mesh"]),
           "--interact-root", str(interact), "--dataset-tag", "behave_test",
           "--gender", "male", "--subject-id", "401", "--object-name", r["object"],
           "--clip-idx", "0", "--betas-npz", str(npz)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "betas: SHARED sub401" in res.stdout
    seq = interact / "data" / "behave_test" / "sequences_canonical" / f"sub401_{r['object']}_000"
    h = np.load(seq / "human.npz")
    np.testing.assert_allclose(h["beta"], [0.55] * 10, atol=1e-6)      # not the clip's 0.9
    # and a missing key fails loudly rather than falling back to the clip's own
    res = subprocess.run(cmd[:-1] + [str(npz), "--betas-key", "sub999"], capture_output=True, text=True)
    assert res.returncode != 0 and "no key 'sub999'" in res.stderr


# ---- per-clip ball radius ---------------------------------------------------
def test_radius_from_mesh(tmp_path):
    p = tmp_path / "ball.obj"
    p.write_text("v -0.12 0 0\nv 0.12 0 0\nv 0 -0.1 0\nv 0 0.1 0\nv 0 0 -0.11\nv 0 0 0.11\nf 1 2 3\n")
    assert relabel.radius_from_mesh(p) == pytest.approx(0.12)
    (tmp_path / "empty.obj").write_text("# nothing\n")
    with pytest.raises(ValueError):
        relabel.radius_from_mesh(tmp_path / "empty.obj")


# ---- per-object mass --------------------------------------------------------
def test_density_for_mass_unit_cube(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    object_mass = _load("object_mass", "isaacgym/src/intermimic/utils/object_mass.py")
    p = tmp_path / "cube.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(str(p))
    density, vol, raw = object_mass.density_for_mass(p, 0.624)
    assert vol == pytest.approx(0.008, rel=1e-6)            # hull volume of a cube = its volume
    assert density == pytest.approx(0.624 / 0.008, rel=1e-6)
    assert raw == pytest.approx(0.008, rel=1e-6)
    # A triangle SOUP (every face its own island, like the Hunyuan balls): the raw
    # volume is unreliable but the hull volume is unchanged -- the whole point.
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.12)
    soup = trimesh.Trimesh(vertices=m.triangles.reshape(-1, 3),
                           faces=np.arange(len(m.faces) * 3).reshape(-1, 3), process=False)
    soup.export(str(tmp_path / "soup.obj"))
    d2, v2, _ = object_mass.density_for_mass(tmp_path / "soup.obj", 0.624)
    assert v2 == pytest.approx(float(m.convex_hull.volume), rel=1e-6)


def test_task_requires_exactly_one_of_density_or_mass():
    """intermimic.py cannot be imported without Isaac Gym; pin the contract in source."""
    src = open(os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")).read()
    assert "self.object_mass = cfg['env'].get('objectMass', None)" in src
    assert "(self.object_density is None) == (self.object_mass is None)" in src
    assert "density_for_mass(obj_file, self.object_mass)" in src
    keys = src[src.index("KNOWN_ENV_KEYS"):src.index("def _validate_env_config")]
    assert "'objectMass'" in keys


def test_bball7_cfgs_are_the_bball_recipe_plus_data_keys():
    """Exactly the documented keys differ from the single-clip g3 bball arm."""
    import yaml
    C = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
    base = yaml.safe_load(open(os.path.join(C, "omomo_teacher_g3_bball_geoall__f0.yaml")))
    new = yaml.safe_load(open(os.path.join(C, "omomo_teacher_g3_bball7_geoall__f0.yaml")))
    diff = {k for k in set(base["env"]) | set(new["env"]) if base["env"].get(k) != new["env"].get(k)}
    assert diff == {"motion_file", "dataSub", "retargetedMotionDir", "objectDensity", "objectMass"}
    assert base["sim"] == new["sim"]
    assert new["env"]["subjectBodies"] == base["env"]["subjectBodies"]     # f0's 43 bodies
    assert new["env"]["dataSub"] == ["sub401", "sub402", "sub404", "sub409", "sub411", "sub412", "sub458"]
    assert "objectDensity" not in new["env"] and new["env"]["objectMass"] == 0.624
    ev = yaml.safe_load(open(os.path.join(C, "omomo_eval_g3_bball7_geoall__f0.yaml")))
    assert ev["evalFor"] == ["g3_bball7_geoall__f0"]
    for k in ("motion_file", "retargetedMotionDir", "dataSub", "objectMass", "rewardShape"):
        assert ev["env"][k] == new["env"][k], k

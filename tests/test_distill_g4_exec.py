#!/usr/bin/env python3
"""EXECUTE the g4 routing on real data, not just read its source.

The task module cannot be imported here (rl_games / Isaac Gym are cluster-only),
so the routing methods are lifted out of the real source text with `ast` and run
on a stand-in task object that reproduces InterMimic's object bookkeeping
EXACTLY as intermimic.py does it (lines 297-306 + the body-major repeat at 379):

    object_name  = [filename.split('_')[-2] for each loaded clip]
    sorted set   -> self.object_name ; self.object_id = index into it, per motion
    retarget     -> self.object_id.repeat(n_bodies) ; data_id = body * n_clips + clip

Over the REAL OMOMO_new file list (~/new_one/OMOMO_new, 4421 clips; hard-fails if
absent), every env's chosen teacher must own the OBJECT in its clip's filename --
checked for ALL 3356 x 43 = 144,308 (body, clip) motions, and again under a
chairs-only filter, and again with a manifest the collector wrote against the
real gen4 cfgs. The g3 methods are executed the same way to prove the seam left
source routing intact. The student launcher's teacher-set guard is executed
too (it is a python program inside the shell script).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_distill_g4_exec.py -q
"""
import ast
import importlib.util
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "isaacgym/src/intermimic"
CFG = PKG / "data/cfg"
OMOMO_NEW = Path(os.path.expanduser("~/new_one/OMOMO_new"))
TRAIN_SOURCES = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]
GROUPS = {"boxes": ["largebox", "smallbox", "plasticbox", "suitcase", "trashcan", "monitor"],
          "stands": ["tripod", "clothesstand", "floorlamp"],
          "chairs": ["whitechair", "woodchair"], "tables": ["largetable", "smalltable"]}
N_BODIES = 43


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


dg3 = _load(PKG / "utils/distill_g3.py", "distill_g3")
dg4 = _load(PKG / "utils/distill_g4.py", "distill_g4")
cg = _load(REPO / "scripts/collect_g3_teachers.py", "collect_g3_teachers")


def lift_methods(path, class_name, names):
    """Compile the named methods of a class out of a source file we cannot import,
    into a fresh class whose globals hold torch + the real helper functions."""
    src = path.read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    ns = {"torch": torch, "load_object_manifest": dg4.load_object_manifest,
          "build_object_lookup": dg4.build_object_lookup,
          "load_teacher_manifest": dg3.load_teacher_manifest,
          "build_source_lookup": dg3.build_source_lookup}
    body = {}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in names:
            code = ast.get_source_segment(src, fn)
            code = "\n".join(l[4:] if l.startswith("    ") else l for l in code.splitlines())   # dedent one level
            exec(code, ns, body)
    missing = set(names) - set(body)
    assert not missing, f"{class_name} lacks {missing}"
    return type(class_name + "Lifted", (), body)


G4 = lift_methods(PKG / "env/tasks/intermimic_distill_g4.py", "InterMimicDistillG4",
                  ["_load_teacher_entries", "_describe_teacher", "_build_teacher_routing", "_refresh_teacher_indices"])
G3 = lift_methods(PKG / "env/tasks/intermimic_distill_g3.py", "InterMimicDistillG3",
                  ["_load_teacher_entries", "_describe_teacher", "_build_teacher_routing", "_refresh_teacher_indices"])


class FakeTask:
    """InterMimic's object bookkeeping, reproduced from intermimic.py on a clip list."""

    def __init__(self, filenames, n_bodies=N_BODIES, num_envs=4096, seed=0):
        self.clip_names = list(filenames)
        self.clip_objects = [f.split('_')[-2] for f in self.clip_names]            # intermimic.py:297
        self.clip_sources = [int(f.split('_')[0][3:]) for f in self.clip_names]
        object_name_set = sorted(set(self.clip_objects))                            # :298
        self.object_id = torch.tensor([object_name_set.index(o) for o in self.clip_objects], dtype=torch.long)  # :304
        self.object_name = object_name_set                                          # :306
        self.source_subject_index = torch.tensor(self.clip_sources, dtype=torch.long)
        self.n_clips = len(self.clip_names)
        self.n_bodies = n_bodies
        self.object_id = self.object_id.repeat(n_bodies)                            # :379 body-major
        self.source_subject_index = self.source_subject_index.repeat(n_bodies)
        self.num_motions = self.n_clips * n_bodies
        self.num_envs = num_envs
        self.device = "cpu"
        g = torch.Generator().manual_seed(seed)
        self.data_id = torch.randint(0, self.num_motions, (num_envs,), generator=g)

    def clip_of(self, data_id):
        """data_id = body * n_clips + clip (intermimic.py's expanded layout)."""
        return int(data_id) % self.n_clips


@pytest.fixture(scope="module")
def train_files():
    if not OMOMO_NEW.is_dir():
        pytest.fail(f"real OMOMO_new not found at {OMOMO_NEW}; this test runs on the real data")
    files = sorted(f for f in os.listdir(OMOMO_NEW)
                   if f.endswith(".pt") and int(f.split("_")[0][3:]) in TRAIN_SOURCES)
    assert len(files) == 3356
    return files


def _entries():
    return [dg4.ObjectTeacherEntry(f"{g}.pth", objs, epoch=1) for g, objs in GROUPS.items()]


def _owner_of(obj):
    return next(i for i, (g, objs) in enumerate(GROUPS.items()) if obj in objs)


# ----------------------------------------------------------------- g4 routing, executed
def test_g4_routes_every_env_to_the_teacher_owning_its_clip_object(train_files):
    t = FakeTask(train_files)
    t._teacher_entries = _entries()
    out = io.StringIO()
    with redirect_stdout(out):
        G4._build_teacher_routing(t)
    assert "4 teachers cover 13 objects, routed by the clip's OBJECT" in out.getvalue()
    assert "NOTE" not in out.getvalue()                       # every teacher used
    G4._refresh_teacher_indices(t)
    assert t.model_indices.shape == (t.num_envs,) and t.sample_indices.tolist() == list(range(t.num_envs))
    for e in range(t.num_envs):
        clip = t.clip_of(t.data_id[e])
        assert int(t.model_indices[e]) == _owner_of(t.clip_objects[clip]), (e, t.clip_names[clip])


def test_g4_routing_is_right_for_every_motion_not_just_a_sample(train_files):
    """Exhaustive: data_id = every (body, clip) of the body-major expansion."""
    t = FakeTask(train_files, num_envs=1)
    t._teacher_entries = _entries()
    with redirect_stdout(io.StringIO()):
        G4._build_teacher_routing(t)
    t.data_id = torch.arange(t.num_motions)
    t.num_envs = t.num_motions
    G4._refresh_teacher_indices(t)
    expected = torch.tensor([_owner_of(o) for o in t.clip_objects]).repeat(t.n_bodies)
    assert t.model_indices.shape == (3356 * N_BODIES,)
    assert torch.equal(t.model_indices, expected)
    # and per object, the whole set of motions lands on exactly one teacher
    for obj in t.object_name:
        mask = t.object_id == t.object_name.index(obj)
        assert mask.sum() == t.clip_objects.count(obj) * N_BODIES
        assert t.model_indices[mask].unique().tolist() == [_owner_of(obj)], obj


def test_g4_routing_under_a_chairs_only_filter_reports_unused_teachers(train_files):
    chairs = [f for f in train_files if f.split("_")[-2] in ("whitechair", "woodchair")]
    assert len(chairs) == 566
    t = FakeTask(chairs)
    t._teacher_entries = _entries()
    out = io.StringIO()
    with redirect_stdout(out):
        G4._build_teacher_routing(t)
    assert "NOTE: 3 teacher(s) serve no clip in this data: ['boxes.pth', 'stands.pth', 'tables.pth']" in out.getvalue()
    t.data_id = torch.arange(t.num_motions); t.num_envs = t.num_motions
    G4._refresh_teacher_indices(t)
    assert t.model_indices.unique().tolist() == [list(GROUPS).index("chairs")]


def test_g4_refuses_an_object_with_no_teacher(train_files):
    t = FakeTask(train_files)
    t._teacher_entries = _entries()[:3]                       # no tables teacher
    with pytest.raises(ValueError, match=r"no teacher: \['largetable', 'smalltable'\]"):
        with redirect_stdout(io.StringIO()):
            G4._build_teacher_routing(t)


def test_g4_never_consults_the_source_subject(train_files):
    """A g4 stand-in WITHOUT source_subject_index must still route: the object is the only key."""
    t = FakeTask(train_files)
    del t.source_subject_index
    t._teacher_entries = _entries()
    with redirect_stdout(io.StringIO()):
        G4._build_teacher_routing(t)
    G4._refresh_teacher_indices(t)
    assert int(t.model_indices.min()) >= 0


def test_g4_describe_and_load_use_the_object_manifest(tmp_path):
    for g, objs in GROUPS.items():
        open(tmp_path / f"{g}.pth", "wb").close()
    yaml.safe_dump({"teachers": [{"file": f"{g}.pth", "objects": objs, "epoch": 5} for g, objs in GROUPS.items()]},
                   open(tmp_path / "teachers.yaml", "w"))
    ents = G4._load_teacher_entries(None, str(tmp_path))
    assert [e.file for e in ents] == [f"{g}.pth" for g in GROUPS]
    assert G4._describe_teacher(None, ents[2]) == "objects ['whitechair', 'woodchair']"
    # a g3 manifest through the g4 loader is refused, naming the right task
    yaml.safe_dump({"teachers": [{"file": "boxes.pth", "sources": [2]}]}, open(tmp_path / "teachers.yaml", "w"))
    with pytest.raises(ValueError, match="InterMimicDistillG3"):
        G4._load_teacher_entries(None, str(tmp_path))


# ----------------------------------------------------------------- g3 routing, still by source
def test_g3_seam_still_routes_by_source(train_files):
    t = FakeTask(train_files)
    t._teacher_entries = [dg3.TeacherEntry(f"sub{s}.pth", [s]) for s in TRAIN_SOURCES]
    out = io.StringIO()
    with redirect_stdout(out):
        G3._build_teacher_routing(t)
    assert "13 teachers cover 13 sources" in out.getvalue()
    G3._refresh_teacher_indices(t)
    for e in range(t.num_envs):
        clip = t.clip_of(t.data_id[e])
        assert int(t.model_indices[e]) == TRAIN_SOURCES.index(t.clip_sources[clip])
    assert G3._describe_teacher(None, t._teacher_entries[0]) == "sources [1]"


def test_g3_seam_refuses_a_source_without_a_teacher(train_files):
    t = FakeTask(train_files)
    t._teacher_entries = [dg3.TeacherEntry(f"sub{s}.pth", [s]) for s in TRAIN_SOURCES if s != 9]
    with pytest.raises(ValueError, match=r"no teacher: \['sub9'\]"):
        with redirect_stdout(io.StringIO()):
            G3._build_teacher_routing(t)


# ----------------------------------------------------------------- collector against the REAL cfgs
def test_collector_names_match_the_real_gen4_teacher_cfgs():
    """The collector must look in the dir the teachers actually write: full_experiment_name."""
    for g in GROUPS:
        tr = yaml.safe_load(open(CFG / "train/rlg" / cg.GEN4_OBJ_CFG.format(name=g)))
        assert cg.GEN4_OBJ_EXP.format(name=g) == tr["params"]["config"]["full_experiment_name"], g
        assert cg.arm_objects(g, str(CFG)) == GROUPS[g], g
    listed = [o for g in GROUPS for o in cg.arm_objects(g, str(CFG))]
    assert len(listed) == 13 == len(set(listed))


def test_collector_manifest_from_real_cfgs_routes_every_real_clip(tmp_path, train_files, monkeypatch):
    """End to end: a tree with the four teachers -> collector (real cfgs, latest) -> manifest
    -> g4 routing over the real clip list."""
    root = tmp_path / "checkpoints"
    epochs = {"boxes": 35000, "stands": 36000, "chairs": 44000, "tables": 46000}
    for g, ep in epochs.items():
        d = root / cg.GEN4_OBJ_EXP.format(name=g) / "nn"
        d.mkdir(parents=True)
        for n in ("mimic.pth", f"mimic_{ep:08d}.pth", "mimic_00001000.pth"):
            (d / n).write_bytes(n.encode())
    monkeypatch.setattr(cg, "epoch_inside", lambda p: 1000)
    out = tmp_path / "teachers"
    with redirect_stdout(io.StringIO()):
        cg.main(["--gen4-objects", *GROUPS, "--out", str(out), "--root", str(root), "--cfg-dir", str(CFG)])
    m = yaml.safe_load(open(out / "teachers.yaml"))["teachers"]
    assert [(e["file"], e["epoch"]) for e in m] == [(f"{g}.pth", ep) for g, ep in epochs.items()]
    t = FakeTask(train_files)
    t._teacher_entries = G4._load_teacher_entries(None, str(out))
    with redirect_stdout(io.StringIO()):
        G4._build_teacher_routing(t)
    t.data_id = torch.arange(t.num_motions); t.num_envs = t.num_motions
    G4._refresh_teacher_indices(t)
    for i, (g, objs) in enumerate(GROUPS.items()):
        for o in objs:
            mask = t.object_id == t.object_name.index(o)
            assert t.model_indices[mask].unique().tolist() == [i], (g, o)


# ----------------------------------------------------------------- the student launcher's guards, executed
ARM = "student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0"


def _launcher_python_guard():
    txt = (REPO / f"slurm_{ARM}.sh").read_text()
    m = re.search(r'if ! python3 -c "\n(import yaml,sys\nt=yaml\.safe_load\(open\(\'\$TP/teachers\.yaml\'\)\).*?)"; then', txt, re.S)
    assert m, "teacher-set guard not found"
    return m.group(1)


def _run_guard(tp):
    code = _launcher_python_guard().replace("$TP", str(tp))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_launcher_teacher_guard_accepts_a_complete_object_manifest(tmp_path):
    yaml.safe_dump({"teachers": [{"file": f"{g}.pth", "objects": objs} for g, objs in GROUPS.items()]},
                   open(tmp_path / "teachers.yaml", "w"))
    r = _run_guard(tmp_path)
    assert r.returncode == 0 and "missing objects: none" in r.stdout, r.stdout + r.stderr


def test_launcher_teacher_guard_refuses_a_g3_manifest(tmp_path):
    yaml.safe_dump({"teachers": [{"file": "sub2.pth", "sources": [2]}]}, open(tmp_path / "teachers.yaml", "w"))
    r = _run_guard(tmp_path)
    assert r.returncode == 1 and "keyed by sources" in r.stdout


def test_launcher_teacher_guard_refuses_a_missing_object(tmp_path):
    groups = {g: list(objs) for g, objs in GROUPS.items()}
    groups["tables"] = ["largetable"]                      # smalltable has no teacher
    yaml.safe_dump({"teachers": [{"file": f"{g}.pth", "objects": objs} for g, objs in groups.items()]},
                   open(tmp_path / "teachers.yaml", "w"))
    r = _run_guard(tmp_path)
    assert r.returncode == 1 and "missing objects: ['smalltable']" in r.stdout


def test_launcher_greps_resolve_the_real_cfg_paths():
    """The launcher reads teacherPolicy / retargetedMotionDir out of the cfg with grep+awk;
    run those exact lines against the real cfg."""
    txt = (REPO / f"slurm_{ARM}.sh").read_text()
    cfg = CFG / f"omomo_{ARM}.yaml"
    for var, want in (("TP", "checkpoints/teachers/gen4_omomo_groups_msexp"),
                      ("RT", "InterAct/OMOMO_retarget_contact_srcall13"),
                      ("TC", f"intermimic/data/cfg/train/rlg/omomo_teacher_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0.yaml")):
        line = next(l for l in txt.splitlines() if l.startswith(f"{var}=$(grep"))
        r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"\n{line}\necho "${var}"'], capture_output=True, text=True)
        assert r.stdout.strip() == want, (var, r.stdout, r.stderr)
    assert (PKG.parent / "intermimic/data/cfg/train/rlg/omomo_teacher_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0.yaml").is_file()


def test_launcher_fold_guard_executes_and_passes_on_the_real_cfg():
    txt = (REPO / f"slurm_{ARM}.sh").read_text()
    block = re.search(r"(for b in sub4 sub10 sub13 sub16; do\n.*?\ndone)", txt, re.S).group(1)
    cfg = CFG / f"omomo_{ARM}.yaml"
    r = subprocess.run(["bash", "-c", f'CFG_ENV="{cfg}"\n{block}\necho FOLD_OK'], capture_output=True, text=True)
    assert r.returncode == 0 and "FOLD_OK" in r.stdout, r.stderr
    # and it would FAIL on a cfg that leaks a test body
    leaked = yaml.safe_load(open(cfg)); leaked["env"]["subjectBodies"].append("sub4")
    bad = REPO / "tests" / "_tmp_leaked_cfg.yaml"
    try:
        yaml.safe_dump(leaked, open(bad, "w"))
        r = subprocess.run(["bash", "-c", f'CFG_ENV="{bad}"\n{block}\necho FOLD_OK'], capture_output=True, text=True)
        assert r.returncode == 1 and "test body sub4 found" in r.stderr and "FOLD_OK" not in r.stdout
    finally:
        bad.unlink(missing_ok=True)


def test_student_obs_width_rule_for_the_gen4_student():
    e = yaml.safe_load(open(CFG / f"omomo_{ARM}.yaml"))["env"]
    assert dg3.student_obs_width(e["numObs"], e["obsHorizons"], e["studentObsHorizons"], uses_betas=False) == e["numObsRetarget"] == 9594

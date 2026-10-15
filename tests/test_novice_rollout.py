#!/usr/bin/env python3
"""scripts/novice_rollout.py (expert motion on the novice's body): the plan is
validated and the derived files written before any GPU time; the stage commands
carry the right bodies, source, flags and anchoring. Pure helpers on fixtures; a
full --dry-run on a REAL OMOMO clip (~/new_one/OMOMO_new, hard-fails if absent)
through a real arm's eval cfg. Plus sim_to_cari4d_bundle.anchor_rollout numerically.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_novice_rollout.py -q
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "isaacgym/src/intermimic/data/assets"
def _omomo_new():
    """The FULL OMOMO_new (4421 clips): $OMOMO_NEW, else the laptop copy, else the repo's
    InterAct/OMOMO_new when it is the full set (the cluster clone). Tests hard-fail otherwise."""
    for cand in (os.environ.get("OMOMO_NEW"), "~/new_one/OMOMO_new", str(REPO / "InterAct/OMOMO_new")):
        if cand and Path(os.path.expanduser(cand)).is_dir():
            n = sum(1 for x in os.listdir(os.path.expanduser(cand)) if x.endswith(".pt"))
            if n == 4421:
                return Path(os.path.expanduser(cand))
    return Path(os.path.expanduser(os.environ.get("OMOMO_NEW") or "~/new_one/OMOMO_new"))


OMOMO_NEW = _omomo_new()
TRACKED_MJCF = ASSETS / "smplx/smplx_variant_sub4_inertial.xml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


nr = _load("nr", "scripts/novice_rollout.py")


# ----------------------------------------------------------------- pure helpers
def test_parse_clip_name():
    assert nr.parse_clip_name("x/sub204_bball_000.pt") == ("sub204", "bball", "000")
    assert nr.parse_clip_name("sub488_soccerd63s88t002j_001.pt") == ("sub488", "soccerd63s88t002j", "001")
    for bad in ("sub204_bball_000.npz", "bball_000.pt", "sub204.pt"):
        with pytest.raises(SystemExit):
            nr.parse_clip_name(bad)


def test_object_props_entry_exact_then_family_never_a_guess(tmp_path):
    f = tmp_path / "props.yaml"
    yaml.safe_dump({"objects": {"bballd03s01rev003b": {"mass": 0.624, "restitution": 1.0},
                                "bballd05s02x": {"mass": 0.624, "restitution": 1.0},
                                "manikin": {"mass": 3.7, "restitution": 0.05}}}, open(f, "w"))
    assert nr.object_props_entry(f, "manikin") == ({"mass": 3.7, "restitution": 0.05}, "manikin")
    assert nr.object_props_entry(f, "bballd99newtake") == ({"mass": 0.624, "restitution": 1.0}, "bballd03s01rev003b")
    with pytest.raises(SystemExit, match="no entry of the family 'soccer'"):
        nr.object_props_entry(f, "soccerd71s96t005c")
    yaml.safe_dump({"objects": {"bballd01s01a": {"mass": 0.6, "restitution": 1.0}, "bballd02s01b": {"mass": 0.7, "restitution": 1.0}}}, open(f, "w"))
    with pytest.raises(SystemExit, match="entries disagree"):
        nr.object_props_entry(f, "bballd03s01c")
    assert nr.object_props_entry.__globals__["re"].match(r"^([a-z]+?)(?=d\d|\d|$)", "soccerd71s96t005c").group(1) == "soccer"
    with pytest.raises(SystemExit, match="objectPropsFile not found"):
        nr.object_props_entry(tmp_path / "missing.yaml", "bballA")


def test_height_for_file_then_table_then_mjcf(tmp_path):
    hj = tmp_path / "h.json"
    json.dump({"204": 1.81}, open(hj, "w"))
    assert nr.height_for("sub204", hj, TRACKED_MJCF) == (1.81, str(hj))
    assert nr.height_for("sub2", hj, TRACKED_MJCF) == (1.601, "intermimic.py SUBJECT_HEIGHTS")
    h, src = nr.height_for("sub999", hj, TRACKED_MJCF)
    assert 1.2 < h < 2.2 and "estimated from smplx_variant_sub4_inertial.xml" in src


def test_mjcf_for_prefers_the_cari4d_body_and_refuses_outside_asset_root(tmp_path):
    (tmp_path / "x.xml").write_text("<mujoco/>")
    with pytest.raises(SystemExit, match="must live under the asset root"):
        nr.mjcf_for("sub1", str(tmp_path / "x.xml"))
    with pytest.raises(SystemExit, match="no MJCF for sub99999"):
        nr.mjcf_for("sub99999")
    probe = ASSETS / "smplx/smplh_behave_sub99998.xml"
    try:
        probe.write_text("<mujoco/>")
        assert nr.mjcf_for("sub99998") == "smplx/smplh_behave_sub99998.xml"
    finally:
        probe.unlink()


def test_arm_cfgs_resolves_student_and_teacher_entries():
    env_yaml, train, entry, task = nr.arm_cfgs("student_g3_act_xf_ret_nvadlr__f0")
    assert env_yaml.name == "omomo_eval_student_g3_act_xf_ret_nvadlr__f0.yaml" and train.is_file()
    assert (entry, task) == ("intermimic.run_distill", "InterMimicDistillG3")
    env_yaml, train, entry, task = nr.arm_cfgs("gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0")
    assert (entry, task) == ("intermimic.run", "InterMimic") and train.name.startswith("omomo_teacher_")
    with pytest.raises(SystemExit, match="no eval cfg for arm"):
        nr.arm_cfgs("no_such_arm__f0")


def test_derived_env_cfg_redirects_only_the_given_keys(tmp_path):
    base = REPO / "isaacgym/src/intermimic/data/cfg/omomo_eval_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0.yaml"
    out = tmp_path / "env.yaml"
    changed = nr.derived_env_cfg(base, {"motion_file": tmp_path / "clip", "retargetedMotionDir": tmp_path / "rt",
                                        "subjectHeightsFile": tmp_path / "h.json", "objectPropsFile": tmp_path / "p.yaml"},
                                 out, "test")
    assert set(changed) == {"motion_file", "retargetedMotionDir", "subjectHeightsFile"}     # chairs arm has no props file
    a, b = yaml.safe_load(open(base)), yaml.safe_load(open(out))
    for k in a["env"]:
        assert (b["env"][k] == a["env"][k]) == (k not in changed), k
    assert "objectPropsFile" not in b["env"] and b["sim"] == a["sim"]
    assert open(out).read().startswith("# DERIVED by scripts/novice_rollout.py")


# ----------------------------------------------------------------- the anchor (sim_to_cari4d_bundle)
def test_anchor_rollout_moves_first_pelvis_and_heading_only():
    s2c = _load("s2c", "scripts/sim_to_cari4d_bundle.py")
    from scipy.spatial.transform import Rotation as R
    T = 6
    body_pos = np.zeros((T, 52, 3)); body_pos[:, 0] = [1, 2, 0.9]; body_pos[:, 1] = [1.3, 2, 0.9]
    body_pos[:, 0, 0] += np.arange(T) * 0.1                       # walks along +x
    body_rot = np.tile(R.from_euler("z", 30, degrees=True).as_quat(), (T, 52, 1))
    obj_pos = np.tile([1.5, 2.0, 0.5], (T, 1)); obj_rot = np.tile([0, 0, 0, 1.0], (T, 1))
    p0, r0 = np.array([-3.0, 4.0, 0.95]), R.from_euler("z", 120, degrees=True).as_quat()
    bp, br, op, orr, yaw, shift = s2c.anchor_rollout(body_pos, body_rot, obj_pos, obj_rot, p0, r0)
    assert abs(yaw - 90) < 1e-6
    np.testing.assert_allclose(bp[0, 0], [-3, 4, 0.9], atol=1e-9)                # xy anchored, z kept
    assert abs(np.degrees(s2c.yaw_of(s2c.quat_to_mat(br[0, 0]))) - 120) < 1e-6   # heading anchored
    np.testing.assert_allclose(bp[0, 1] - bp[0, 0], [0, 0.3, 0], atol=1e-9)       # rigid: offsets rotate with it
    np.testing.assert_allclose(bp[5, 0] - bp[0, 0], [0, 0.5, 0], atol=1e-9)       # the walk now goes along +y
    np.testing.assert_allclose(op[0] - bp[0, 0], [0, 0.5, -0.4], atol=1e-9)        # object moves with the body
    assert abs(np.degrees(s2c.yaw_of(s2c.quat_to_mat(orr[0]))) - 90) < 1e-6
    assert body_pos[0, 0, 0] == 1.0                                                # inputs untouched


# ----------------------------------------------------------------- full dry run on real data
@pytest.fixture
def real_clip():
    src = OMOMO_NEW / "sub6_woodchair_000.pt"
    if not src.is_file():
        pytest.fail(f"real clip not found: {src}")
    return src


def test_dry_run_expert_on_novice(tmp_path, real_clip):
    """Expert sub6 (a chairs source) performed by novice sub9998 through the chairs
    msexp teacher's eval cfg: every input validated, derived files written, stage
    commands correct, nothing executed."""
    novice_clip = tmp_path / "sub9998_woodchair_000.pt"; novice_clip.symlink_to(real_clip)
    models = tmp_path / "models"; models.mkdir(); (models / "SMPLH_MALE.pkl").write_bytes(b"x")
    bundle = tmp_path / "hy3d" / "Date170_Sub9998_woodchair_t000.pth"; bundle.parent.mkdir(); bundle.write_bytes(b"x")
    mesh = tmp_path / "hy3d" / "Date170_Sub9998_woodchair_t000_align" / "chair_align.obj"; mesh.parent.mkdir(); mesh.write_text("v 0 0 0\n")
    ckpt = tmp_path / "mimic_00027000.pth"; ckpt.write_bytes(b"x")
    expert_mjcf = ASSETS / "smplx/_test_expert_sub6.xml"
    novice_mjcf = ASSETS / "smplx/smplh_behave_sub9998.xml"
    out = tmp_path / "out"
    try:
        expert_mjcf.write_text(TRACKED_MJCF.read_text()); novice_mjcf.write_text(TRACKED_MJCF.read_text())
        argv = [sys.executable, str(REPO / "scripts/novice_rollout.py"), "--dry-run",
                "--expert-clip", str(real_clip), "--expert-mjcf", str(expert_mjcf),
                "--novice-clip", str(novice_clip), "--novice-bundle", str(bundle), "--object-mesh", str(mesh),
                "--arm", "gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0", "--checkpoint", str(ckpt),
                "--gender", "male", "--models", str(models), "--out", str(out), "--num-envs", "16",
                "--cari4d-root", str(tmp_path), "--video", str(tmp_path / "v.mp4")]
        r = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, r.stdout + r.stderr
        o = r.stdout
        assert "sub9998 performs sub6_woodchair_000.pt (expert sub6, object woodchair)" in o
        assert "DRY RUN: nothing executed" in o and "DISTILL_SKIP_TEACHERS" not in o
        assert "expert smplx/_test_expert_sub6.xml  ->  novice smplx/smplh_behave_sub9998.xml (REPLAY_ASSET_FILE)" in o
        assert "estimated from smplh_behave_sub9998.xml" in o                        # unseen body: no table entry
        # stage commands
        assert f"--source sub6 --source-mjcf {expert_mjcf} --target sub9998 --target-mjcf {novice_mjcf}" in o
        assert f"--out-dir {out}/retarget/sub9998" in o
        assert "DUMP_PICK=best" in o and "REPLAY_ASSET_FILE=smplx/smplh_behave_sub9998.xml" in o
        assert "--subject_bodies sub9998 --data_sub sub6 --data_objects woodchair --num_envs 16" in o
        assert "[novice_npz]" in o and f"--out {out}/novice_original.npz" in o
        assert "[novice_skin]" in o and "[expert_skin]" in o and "--model-type smplh" in o
        assert f"--out {out}/novice_original.mp4" in o and f"--out {out}/expert_on_novice.mp4" in o
        assert f"DUMP_TRAJ={out}/expert_on_novice.npz" in o
        assert "[differencing]" in o and f"--novice {out}/novice_original.npz --expert {out}/expert_on_novice.npz" in o
        assert "--task woodchair" in o and "--dry_run" in o and f"--out {out}/feedback/joint_differences.json" in o
        assert f"--pt {novice_clip} --anchor-rollout" in o and f"--out {out}/hy3d/Date170_Sub9998_woodchair_t000_sim.pth" in o
        assert "viz_pred.py" in o and "--flat_color 255,140,0" in o and "--no_side" in o
        # derived files
        assert (out / "clip" / "sub6_woodchair_000.pt").is_symlink() and (out / "retarget" / "sub9998").is_dir()
        h = json.load(open(out / "heights.json")); assert "9998" in h and "600" in h
        e = yaml.safe_load(open(out / "env_cfg.yaml"))["env"]
        assert e["motion_file"] == str(out / "clip") and e["retargetedMotionDir"] == str(out / "retarget")
        assert e["subjectHeightsFile"] == str(out / "heights.json") and e["humanoidAssetSuffix"] == "_inertial"
        assert e["stateInit"] == "Start" and e["rolloutLength"] == 1000 and not (out / "expert_on_novice.npz").exists()
        # refusals: expert == novice, expert not a training source, novice a training body
        same = subprocess.run(argv[:argv.index("--novice-clip") + 1] + [str(real_clip)] + argv[argv.index("--novice-clip") + 2:],
                              capture_output=True, text=True, cwd=str(REPO))
        assert same.returncode != 0 and "the same subject" in same.stderr
        seen = subprocess.run([x if x != str(novice_clip) else str(tmp_path / "sub2_woodchair_000.pt") for x in argv],
                              capture_output=True, text=True, cwd=str(REPO))
        assert seen.returncode != 0
    finally:
        expert_mjcf.unlink(missing_ok=True); novice_mjcf.unlink(missing_ok=True)


def test_student_arm_commands_use_run_distill_and_the_teacher_skip(tmp_path):
    env_yaml, train, entry, task = nr.arm_cfgs("student_g3_act_xf_ret_nvadlr__f0")
    p = {"out": tmp_path, "frames": 500, "expert_mjcf": "smplx/smplh_behave_sub401.xml",
         "novice_mjcf": "smplx/smplh_behave_sub204.xml", "is_student": True, "entry": entry, "task": task,
         "train_yaml": train, "ckpt": tmp_path / "c.pth", "expert": "sub401", "novice": "sub204", "object": "bballx",
         "num_envs": 32, "models": tmp_path, "mesh": tmp_path / "m" / "m.obj", "bundle": tmp_path / "b.pth",
         "novice_clip": tmp_path / "sub204_bball_000.pt", "expert_clip": tmp_path / "sub401_bballx_000.pt", "seq": "b",
         "cari4d_root": None, "video": None, "flat_color": "255,140,0", "retarget_iters": 300,
         "drill_name": "Reverse Layup", "feedback_model": None}
    stages, files = nr.commands(p, tmp_path / "env.yaml")
    names = [s[0] for s in stages]
    assert names == ["retarget", "novice_npz", "rollout", "novice_skin", "expert_skin", "differencing", "bundle", "overlay"]
    by = {n: (e, a) for n, e, a in stages}
    envv, argv = by["rollout"]
    assert envv["DISTILL_SKIP_TEACHERS"] == "1" and envv["DUMP_PICK"] == "best" and envv["DUMP_FRAMES"] == "500"
    assert argv[3] == "intermimic.run_distill" and argv[5] == "InterMimicDistillG3"
    assert argv[argv.index("--subject_bodies") + 1] == "sub204" and argv[argv.index("--data_sub") + 1] == "sub401"
    assert files["rt_file"] == tmp_path / "retarget" / "sub204" / "sub401_bballx_000.pt"
    assert files["dump"].name == "expert_on_novice.npz" and files["novice_npz"].name == "novice_original.npz"
    d = by["differencing"][1]
    assert d[d.index("--task") + 1] == "Reverse Layup" and "--dry_run" in d and "--model" not in d
    assert "--anchor-rollout" in by["bundle"][1] and by["overlay"][1] is None
    p["feedback_model"] = "Qwen/Qwen2.5-3B-Instruct"
    d = {n: a for n, _, a in nr.commands(p, tmp_path / "env.yaml")[0]}["differencing"]
    assert "--dry_run" not in d and d[d.index("--model") + 1] == "Qwen/Qwen2.5-3B-Instruct"


# ----------------------------------------------------------------- the short form
def _novice_export(root, seq="Date170_Sub9997_bball_t004a", gender="male"):
    d = root / seq; d.mkdir(parents=True)
    (d / f"{seq}.pth").write_bytes(b"x")
    (d / f"{seq}_067_align.obj").write_text("v 0 0 0\n")
    json.dump({"clip": seq, "gender": gender, "n_frames": 135}, open(d / "meta.json", "w"))
    return d


def test_resolve_novice_dir_from_a_cari4d_export_folder(tmp_path, real_clip):
    d = _novice_export(tmp_path)
    repo = tmp_path / "repo"
    inst = repo / "InterAct" / "behave_cari4d_date170_sub9997_bball_t004a"; inst.mkdir(parents=True)
    (inst / "sub9997_bball_000.pt").symlink_to(real_clip)
    bundle, mesh, gender, clip, n = nr.resolve_novice_dir(d, repo=repo)
    assert bundle.name == "Date170_Sub9997_bball_t004a.pth" and mesh.name.endswith("_067_align.obj")
    assert gender == "male" and clip.name == "sub9997_bball_000.pt" and n == 135
    # refusals: no installed clip, two bundles, bad gender
    with pytest.raises(SystemExit, match="exactly one installed clip"):
        nr.resolve_novice_dir(d, repo=tmp_path / "empty")
    (d / "other.pth").write_bytes(b"x")
    with pytest.raises(SystemExit, match="exactly one bundle"):
        nr.resolve_novice_dir(d, repo=repo)
    (d / "other.pth").unlink()
    json.dump({"clip": "Date170_Sub9997_bball_t004a", "gender": None}, open(d / "meta.json", "w"))
    with pytest.raises(SystemExit, match="gender is None"):
        nr.resolve_novice_dir(d, repo=repo)


def test_resolve_expert_clip_by_bare_name_in_the_arms_motion_dir(tmp_path, real_clip):
    repo = tmp_path / "repo"; md = repo / "InterAct" / "act"; md.mkdir(parents=True)
    (md / "sub401_bballd03_000.pt").symlink_to(real_clip)
    assert nr.resolve_expert_clip("sub401_bballd03_000", "InterAct/act", repo=repo).name == "sub401_bballd03_000.pt"
    assert nr.resolve_expert_clip("sub401_bballd03_000.pt", "InterAct/act", repo=repo).name == "sub401_bballd03_000.pt"
    assert nr.resolve_expert_clip(str(real_clip), "InterAct/act", repo=repo) == real_clip.absolute()
    with pytest.raises(SystemExit, match="not found as a path nor as"):
        nr.resolve_expert_clip("sub402_x_000", "InterAct/act", repo=repo)


def test_latest_checkpoint_rule(tmp_path):
    tr = tmp_path / "t.yaml"; tr.write_text("params:\n  config:\n    full_experiment_name: smplx_student_x\n")
    nn = tmp_path / "checkpoints" / "smplx_student_x" / "nn"; nn.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no checkpoint"):
        nr.latest_checkpoint("x", tr, repo=tmp_path)
    (nn / "mimic.pth").write_bytes(b"x")
    assert nr.latest_checkpoint("x", tr, repo=tmp_path).name == "mimic.pth"
    for n in ("mimic_00018000.pth", "mimic_00100000.pth", "mimic_00099000.pth"):
        (nn / n).write_bytes(b"x")
    assert nr.latest_checkpoint("x", tr, repo=tmp_path).name == "mimic_00100000.pth"


def test_short_form_dry_run(tmp_path, real_clip):
    """--expert-clip <bare name> --novice-dir <folder>, everything else defaulted or from
    the environment, through the chairs teacher arm (its motion dir is OMOMO_new)."""
    seq = "Date170_Sub9997_woodchair_t004a"
    d = _novice_export(tmp_path, seq)
    inst = REPO / "InterAct" / f"behave_cari4d_{seq.lower()}"
    novice_mjcf = ASSETS / "smplx/smplh_behave_sub9997.xml"
    expert_mjcf = ASSETS / "smplx/smplx_omomo_sub6.xml"
    made_expert = not expert_mjcf.exists()
    models = tmp_path / "models"; models.mkdir(); (models / "SMPLH_MALE.pkl").write_bytes(b"x")
    ckpt_dir = REPO / "checkpoints" / "smplx_teacher_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0" / "nn"
    made_ckpt = not ckpt_dir.exists()
    try:
        inst.mkdir(parents=True); (inst / "sub9997_woodchair_000.pt").symlink_to(real_clip)
        novice_mjcf.write_text(TRACKED_MJCF.read_text())
        if made_expert:
            expert_mjcf.write_text(TRACKED_MJCF.read_text())
        if made_ckpt:
            ckpt_dir.mkdir(parents=True); (ckpt_dir / "mimic_00027000.pth").write_bytes(b"x")
        r = subprocess.run([sys.executable, str(REPO / "scripts/novice_rollout.py"), "--dry-run",
                            "--expert-clip", str(real_clip), "--novice-dir", str(d),      # repo's OMOMO_new is sub2-only here
                            "--arm", "gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0", "--out", str(tmp_path / "out")],
                           capture_output=True, text=True, cwd=str(REPO),
                           env={**os.environ, "SMPLX_MODELS": str(models), "CARI4D_ROOT": str(tmp_path)})
        assert r.returncode == 0, r.stdout + r.stderr
        o = r.stdout
        assert "sub9997 performs sub6_woodchair_000.pt (expert sub6, object woodchair)" in o
        assert f"novice recon : sub9997_woodchair_000.pt + bundle {seq} (camera, betas, male)" in o
        assert "mimic_00027000.pth" in o or "mimic_0" in o                              # latest snapshot picked
        assert "--flat_color 255,140,0" in o and "viz_pred.py" in o                      # CARI4D_ROOT honoured
        assert "DRY RUN" in o
    finally:
        for f in (inst / "sub9997_woodchair_000.pt",):
            f.unlink(missing_ok=True)
        inst.rmdir()
        novice_mjcf.unlink(missing_ok=True)
        if made_expert:
            expert_mjcf.unlink(missing_ok=True)
        if made_ckpt:
            (ckpt_dir / "mimic_00027000.pth").unlink(missing_ok=True); ckpt_dir.rmdir(); ckpt_dir.parent.rmdir()


# ----------------------------------------------------------------- retrieval -> expert clip
def _matches(tmp_path):
    doc = {"query": {"clip": "Date170_Sub204_bball_t004a"}, "hits": [
        {"rank": 1, "clip": "Date08_Sub09_bball_t022a", "participant": 385, "participant_tier": "unlabelled",
         "take_label": "unlabelled", "motion_sim": 0.37, "body_dist": 0.73},
        {"rank": 2, "clip": "Date03_Sub01_bball_rev009c", "participant": 387, "participant_tier": "Late Expert",
         "take_label": "Late Expert", "motion_sim": 0.32, "body_dist": 0.23},
        {"rank": 3, "clip": "Date06_Sub04_bball_t012a", "participant": 382, "participant_tier": "Late Expert",
         "take_label": "Late Expert", "motion_sim": 0.30, "body_dist": 0.40},
    ]}
    m = tmp_path / "matches.json"; json.dump(doc, open(m, "w"))
    return m


def test_installed_name_for_uses_the_manifest_naming_rule(tmp_path):
    md = tmp_path / "InterAct" / "act"; md.mkdir(parents=True)
    (md / "sub401_bballd03s01rev009c_001.pt").write_bytes(b"x")
    assert nr.installed_name_for("Date03_Sub01_bball_rev009c", "InterAct/act", repo=tmp_path).name == "sub401_bballd03s01rev009c_001.pt"
    with pytest.raises(SystemExit, match="expected exactly one installed file"):
        nr.installed_name_for("Date06_Sub04_bball_t012a", "InterAct/act", repo=tmp_path)
    (md / "sub401_bballd03s01rev009c_002.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="expected exactly one installed file"):
        nr.installed_name_for("Date03_Sub01_bball_rev009c", "InterAct/act", repo=tmp_path)
    with pytest.raises(SystemExit, match="is not Date<N>_Sub<N>"):
        nr.installed_name_for("garbage", "InterAct/act", repo=tmp_path)


def test_pick_expert_skips_unrated_and_uninstalled_hits(tmp_path, capsys):
    m = _matches(tmp_path)
    md = tmp_path / "InterAct" / "act"; md.mkdir(parents=True)
    (md / "sub409_bballd08s09t022a_000.pt").write_bytes(b"x")       # rank 1 IS installed but unrated
    (md / "sub401_bballd03s01rev009c_001.pt").write_bytes(b"x")     # rank 2: rated + installed
    hit, path = nr.pick_expert(m, "InterAct/act", repo=tmp_path)
    assert hit["rank"] == 2 and path.name == "sub401_bballd03s01rev009c_001.pt"
    out = capsys.readouterr().out
    assert "#1 Date08_Sub09_bball_t022a" in out and "skipped: not a rated expert" in out
    assert "#2 Date03_Sub01_bball_rev009c" in out and "<== chosen" in out
    assert "#3 Date06_Sub04_bball_t012a" in out and "NOT INSTALLED" in out
    # rank 2 gone: rank 3 is rated but not installed -> refuse, never fall back to the unrated rank 1
    (md / "sub401_bballd03s01rev009c_001.pt").unlink()
    with pytest.raises(SystemExit, match="no retrieval hit is both a RATED expert and installed"):
        nr.pick_expert(m, "InterAct/act", repo=tmp_path)


def test_dry_run_with_matches_json_picks_the_rated_expert(tmp_path, real_clip):
    """Short form with a prior matches.json: the expert comes from retrieval, not a flag."""
    seq = "Date170_Sub9996_woodchair_t004a"
    d = _novice_export(tmp_path, seq)
    inst = REPO / "InterAct" / f"behave_cari4d_{seq.lower()}"
    novice_mjcf = ASSETS / "smplx/smplh_behave_sub9996.xml"
    expert_mjcf = ASSETS / "smplx/smplh_behave_sub402.xml"          # 400 + Sub02
    expert_file = REPO / "InterAct/OMOMO_new" / "sub402_woodchaird03s02rev009c_000.pt"   # chairs arm's motion dir
    made_expert_mjcf = not expert_mjcf.exists()
    models = tmp_path / "models"; models.mkdir(); (models / "SMPLH_MALE.pkl").write_bytes(b"x")
    ckpt = tmp_path / "c.pth"; ckpt.write_bytes(b"x")
    matches = tmp_path / "m.json"
    json.dump({"query": {"clip": seq}, "hits": [
        {"rank": 1, "clip": "Date03_Sub02_woodchair_rev009c", "participant": 388, "participant_tier": "Late Expert",
         "take_label": "Late Expert", "motion_sim": 0.3, "body_dist": 0.2}]}, open(matches, "w"))
    try:
        inst.mkdir(parents=True); (inst / "sub9996_woodchair_000.pt").symlink_to(real_clip)
        expert_file.symlink_to(real_clip)
        novice_mjcf.write_text(TRACKED_MJCF.read_text())
        if made_expert_mjcf:
            expert_mjcf.write_text(TRACKED_MJCF.read_text())
        base = [sys.executable, str(REPO / "scripts/novice_rollout.py"), "--dry-run", "--novice-dir", str(d),
                "--matches", str(matches), "--checkpoint", str(ckpt), "--out", str(tmp_path / "out")]
        # the chairs arm lists sub2, not sub402, as a source: the plan must refuse that (expert not a training source)
        r = subprocess.run(base + ["--arm", "gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0"],
                           capture_output=True, text=True, cwd=str(REPO), env={**os.environ, "SMPLX_MODELS": str(models)})
        assert r.returncode != 0 and "is not in the arm's dataSub" in r.stderr, r.stdout + r.stderr
        assert "<== chosen" in r.stdout and "retrieval hits for" in r.stdout      # the pick itself happened first
    finally:
        (inst / "sub9996_woodchair_000.pt").unlink(missing_ok=True); inst.rmdir()
        expert_file.unlink(missing_ok=True); novice_mjcf.unlink(missing_ok=True)
        if made_expert_mjcf:
            expert_mjcf.unlink(missing_ok=True)


def test_retrieval_copy_carries_the_three_documented_changes():
    s = (REPO / "scripts/recon_retrieval.py").read_text()
    assert 'DEFAULT_DB = osp.join(HERE, "recon_retrieval_db")' in s
    assert (REPO / "scripts/recon_retrieval_db").is_dir() and any((REPO / "scripts/recon_retrieval_db").iterdir())
    assert "note: no takes.json" in s and '"--rated_only"' in s
    assert 'key="participant_tier", match=models.MatchAny(any=list(RATED_TIERS))' in s
    assert nr.RATED_TIERS == ("Early Expert", "Intermediate Expert", "Late Expert")
    assert "PULLED INTO InterMimic" in s and "57fa48f" in s


# ----------------------------------------------------------------- clip_to_dump (the novice's npz)
def test_clip_to_dump_on_a_real_clip(tmp_path, real_clip):
    cd = _load("cd", "scripts/clip_to_dump.py")
    import torch
    x = torch.load(real_clip, map_location="cpu", weights_only=False).detach().numpy()
    d = cd.clip_to_dump(x, "sub6")
    T = x.shape[0]
    assert d["body_pos"].shape == (T, 52, 3) and d["body_rot"].shape == (T, 52, 4)
    assert d["obj_pos"].shape == (T, 3) and d["obj_rot"].shape == (T, 4) and d["subject"] == "sub6"
    np.testing.assert_allclose(d["body_pos"][:, 0], x[:, 0:3], atol=1e-5)       # pelvis == root_pos
    assert cd.subject_of("x/sub204_bball_000.pt") == "sub204"
    with pytest.raises(ValueError, match=r"expected \(T, 591\)"):
        cd.clip_to_dump(x[:, :500], "sub6")
    bad = x.copy(); bad[:, 383:387] *= 3
    with pytest.raises(ValueError, match="not unit-norm"):
        cd.clip_to_dump(bad, "sub6")
    # the CLI writes the same thing, and the feedback script reads it back as a rollout
    out = tmp_path / "n.npz"
    r = subprocess.run([sys.executable, str(REPO / "scripts/clip_to_dump.py"), str(real_clip), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "153 frames, subject sub6" in r.stdout, r.stdout + r.stderr
    mf = _load("mf", "scripts/motion_feedback.py")
    m = mf.load_motion(str(out))
    assert m["body_pos"].shape == (T, 52, 3) and m["subject"] == "sub6"


def test_differencing_dry_run_writes_the_three_tables(tmp_path, real_clip):
    other = OMOMO_NEW / "sub7_woodchair_003.pt"
    if not other.is_file():
        pytest.fail(f"real clip not found: {other}")
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    for src, dst in ((real_clip, a), (other, b)):
        subprocess.run([sys.executable, str(REPO / "scripts/clip_to_dump.py"), str(src), "--out", str(dst)], check=True, capture_output=True)
    out = tmp_path / "feedback" / "joint_differences.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts/motion_feedback.py"), "--novice", str(a), "--expert", str(b),
                        "--task", "sit on a chair", "--dry_run", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.load(open(out))
    assert set(d["tables"]) == {"joint_angles", "object", "motion"} and d["feedback"] is None and d["model"] is None
    assert d["alignment"]["novice_frames"][0] == 0 and len(d["time"]["rows"]) == 3
    rows = d["tables"]["joint_angles"]
    assert {r["joint"] for r in rows} >= {"pelvis", "lower back", "left knee", "right elbow"}
    assert all("novice_minus_expert_deg" in r for r in rows)


def test_resolve_novice_dir_falls_back_to_the_sub_number_under_any_tag(tmp_path, real_clip):
    d = _novice_export(tmp_path, "Date170_Sub9995_bball_t004a")
    repo = tmp_path / "repo"
    inst = repo / "InterAct" / "behave_cari4d_rectinj_hy3d3_optj3d"; inst.mkdir(parents=True)   # the release tag, not the seq
    (inst / "sub9995_bball_000.pt").symlink_to(real_clip)
    bundle, mesh, gender, clip, n = nr.resolve_novice_dir(d, repo=repo)
    assert clip.name == "sub9995_bball_000.pt"
    (inst / "sub9995_bball_001.pt").symlink_to(real_clip)
    with pytest.raises(SystemExit, match="exactly one installed clip"):
        nr.resolve_novice_dir(d, repo=repo)

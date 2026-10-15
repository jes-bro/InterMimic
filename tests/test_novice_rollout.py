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
OMOMO_NEW = Path(os.path.expanduser("~/new_one/OMOMO_new"))
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
        assert "[skinned]" in o and "--model-type smplh" in o
        assert f"--pt {novice_clip} --anchor-rollout" in o and f"--out {out}/hy3d/Date170_Sub9998_woodchair_t000_sim.pth" in o
        assert "viz_pred.py" in o and "--flat_color 255,140,0" in o and "--no_side" in o
        # derived files
        assert (out / "clip" / "sub6_woodchair_000.pt").is_symlink() and (out / "retarget" / "sub9998").is_dir()
        h = json.load(open(out / "heights.json")); assert "9998" in h and "600" in h
        e = yaml.safe_load(open(out / "env_cfg.yaml"))["env"]
        assert e["motion_file"] == str(out / "clip") and e["retargetedMotionDir"] == str(out / "retarget")
        assert e["subjectHeightsFile"] == str(out / "heights.json") and e["humanoidAssetSuffix"] == "_inertial"
        assert e["stateInit"] == "Start" and e["rolloutLength"] == 1000 and not (out / "rollout.npz").exists()
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
         "cari4d_root": None, "video": None, "flat_color": "255,140,0", "retarget_iters": 300}
    stages, rt_file, dump, betas = nr.commands(p, tmp_path / "env.yaml")
    names = [s[0] for s in stages]
    assert names == ["retarget", "rollout", "skinned", "bundle", "overlay"]
    envv, argv = stages[1][1], stages[1][2]
    assert envv["DISTILL_SKIP_TEACHERS"] == "1" and envv["DUMP_PICK"] == "best" and envv["DUMP_FRAMES"] == "500"
    assert argv[3] == "intermimic.run_distill" and argv[5] == "InterMimicDistillG3"
    assert argv[argv.index("--subject_bodies") + 1] == "sub204" and argv[argv.index("--data_sub") + 1] == "sub401"
    assert rt_file == tmp_path / "retarget" / "sub204" / "sub401_bballx_000.pt"
    assert "--anchor-rollout" in stages[3][2] and stages[4][2] is None

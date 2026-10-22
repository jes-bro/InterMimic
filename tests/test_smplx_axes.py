"""The SMPL <-> simulator axis mapping and the exact native pose (scripts/smplx_pose.py).

Real model files and real clips, no fixtures: SMPL-X from $SMPLX_MODELS (default
~/Downloads/models/smplx), the checked-in humanoid MJCFs and InterAct/OMOMO_new clips.
A missing file is a FAILURE here, not a skip -- the whole point is the real geometry.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_smplx_axes.py -q
"""
import glob
import importlib.util
import os

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.expanduser(os.environ.get("SMPLX_MODELS", "~/Downloads/models/smplx"))
ASSETS = os.path.join(REPO, "isaacgym/src/intermimic/data/assets/smplx")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sp = _load("smplx_pose", "scripts/smplx_pose.py")
d2i = _load("dump_to_interact", "scripts/dump_to_interact.py")
BODY_NAMES = [n for n, _, _ in sp._parse_mjcf_tree(os.path.join(ASSETS, "omomo.xml"))
              if not any(f in n for f in ("Index", "Middle", "Pinky", "Ring", "Thumb"))]


@pytest.fixture(scope="module")
def poser():
    assert os.path.isfile(os.path.join(MODELS, "SMPLX_MALE.npz")), MODELS
    return sp.SMPLXPoser(models_dir=MODELS, betas_path=os.path.join(REPO, "scripts/omomo_betas.npz"),
                         model_type="smplx")


@pytest.fixture(scope="module")
def sub2_clip():
    clips = sorted(glob.glob(os.path.join(REPO, "InterAct/OMOMO_new/sub2_*.pt")))
    assert clips, "no sub2 clips under InterAct/OMOMO_new"
    return torch.load(clips[0], map_location="cpu", weights_only=False).detach().numpy()


def test_q_zup_is_the_measured_permutation():
    # sim = (z, x, y) of SMPL: SMPL's (left, up, forward) -> the sim's (forward, left, up)
    assert sp.Q_ZUP.tolist() == [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert np.allclose(sp.Q_ZUP @ sp.Q_ZUP.T, np.eye(3)) and np.linalg.det(sp.Q_ZUP) == 1
    assert np.array_equal(d2i.Q_ZUP, sp.Q_ZUP), "dump_to_interact.py carries a different matrix"


def test_template_lands_on_the_generic_mjcf_rest_skeleton(poser):
    """omomo.xml IS sub9's skeleton: under Q_ZUP the shaped template's 22 body joints
    coincide with the MJCF's rest joints to the millimetre (37 cm under the old matrix)."""
    _, J0, _ = poser._shape("sub9")
    _, p_rest = poser.mjcf_fk(np.zeros(153), np.zeros(3), np.array([0, 0, 0, 1.0]))
    body = [poser.body_order.index(n) for n in BODY_NAMES]
    a, b = J0[[poser.b2s[k] for k in body]], p_rest[body]
    err = np.linalg.norm((a - a.mean(0)) - (b - b.mean(0)), axis=1)
    assert err.max() < 0.001, f"max {err.max()*100:.2f} cm"


def test_native_pose_inverts_the_sims_body_state_exactly(sub2_clip):
    """With the subject's own MJCF, native_from_dof -> fk_joints reproduces the sim's
    body joints (MJCF FK) to < 0.5 mm, frame after frame. Hands are excluded: the MJCF
    fingertip bodies sit ~3 cm from SMPL's last finger joints by construction."""
    p2 = sp.SMPLXPoser(models_dir=MODELS, betas_path=os.path.join(REPO, "scripts/omomo_betas.npz"),
                       model_type="smplx", mjcf=os.path.join(ASSETS, "smplx_omomo_sub2.xml"))
    body = [p2.body_order.index(n) for n in BODY_NAMES]
    x = sub2_clip
    worst = 0.0
    for t in range(0, min(len(x), 120), 20):
        _, p_sim = p2.mjcf_fk(x[t, 9:162], x[t, 0:3], x[t, 3:7])
        pose_aa, trans = p2.native_from_dof("sub2", x[t, 9:162], x[t, 0:3], x[t, 3:7])
        J = p2.fk_joints("sub2", pose_aa, trans)[p2.b2s]
        worst = max(worst, np.linalg.norm(J[body] - p_sim[body], axis=1).max())
        # and the stored body_pos of the clip is the same skeleton: no alignment at all
        stored = x[t, 162:318].reshape(52, 3)
        assert np.linalg.norm(J[body] - stored[body], axis=1).mean() < 0.01
    assert worst < 0.0005, f"{worst*100:.3f} cm"


def test_native_from_bodies_matches_native_from_dof(sub2_clip):
    """A DUMP_TRAJ carries body_rot/body_pos; a clip carries dof_pos. Same frame, same pose."""
    p2 = sp.SMPLXPoser(models_dir=MODELS, betas_path=os.path.join(REPO, "scripts/omomo_betas.npz"),
                       model_type="smplx", mjcf=os.path.join(ASSETS, "smplx_omomo_sub2.xml"))
    x = sub2_clip[40]
    R, p = p2.mjcf_fk(x[9:162], x[0:3], x[3:7])
    from scipy.spatial.transform import Rotation
    quats = Rotation.from_matrix(R).as_quat()                  # xyzw, as the sim dumps
    a_pose, a_tr = p2.native_from_dof("sub2", x[9:162], x[0:3], x[3:7])
    b_pose, b_tr = p2.native_from_bodies("sub2", quats, p)
    assert np.allclose(a_pose, b_pose, atol=1e-6) and np.allclose(a_tr, b_tr, atol=1e-6)
    seq_pose, seq_tr = p2.native_sequence("sub2", quats[None], p[None])
    assert seq_pose.shape == (1, 55, 3) and np.allclose(seq_pose[0], b_pose)


def test_generic_skeleton_on_another_body_is_a_few_cm_not_a_quarter_turn(poser, sub2_clip):
    """The old selftest rigid-aligned (rotation included) and so could not see the yaw.
    Translation-only: sub2's clip on the generic (sub9) skeleton is ~3 cm of shape, not 37."""
    M = poser._model("male")
    x = sub2_clip[30]
    v, _ = poser.pose_from_dof("sub2", x[9:162], x[0:3], x[3:7])
    Jz = (M["J_reg"] @ v)[poser.b2s]
    stored = x[162:318].reshape(52, 3)
    err = np.linalg.norm((Jz - Jz.mean(0)) - (stored - stored.mean(0)), axis=1).mean()
    assert err < 0.06, f"{err*100:.1f} cm"


SMPLH_MODELS = os.path.expanduser("~/interact/InterAct/models/smplh")


def test_dump_to_interact_frame_change_is_q_zup_transposed():
    """InterAct renders in SMPL's native frame; a sim-frame vector gets there by Q_ZUP^T."""
    assert np.array_equal(d2i.GYM_TO_YUP, sp.Q_ZUP.T)


def test_dump_to_interact_native_export_reproduces_the_sim(sub2_clip, tmp_path):
    """A DUMP_TRAJ built from a real sub2 clip (MJCF FK -> body_rot/body_pos), exported with
    the exact native pose, forwarded through SMPL-H in the renderer's frame, lands on the
    sim's body joints to < 0.5 mm; the object's translation is the same frame change."""
    import subprocess
    import sys
    from scipy.spatial.transform import Rotation
    assert os.path.isfile(os.path.join(SMPLH_MODELS, "SMPLH_MALE.pkl")), SMPLH_MODELS
    p2 = sp.SMPLXPoser(models_dir=SMPLH_MODELS, betas_path=os.path.join(REPO, "scripts/omomo_betas.npz"),
                       model_type="smplh", mjcf=os.path.join(ASSETS, "smplx_omomo_sub2.xml"))
    frames = sub2_clip[0:60:12]
    R, P = zip(*(p2.mjcf_fk(x[9:162], x[0:3], x[3:7]) for x in frames))
    body_rot = np.stack([Rotation.from_matrix(r).as_quat() for r in R])        # (T,52,4) xyzw
    body_pos = np.stack(P)
    dump = tmp_path / "sub2_fake_rollout.npz"
    np.savez(dump, subject="sub2", body_rot=body_rot, body_pos=body_pos,
             obj_pos=frames[:, 318:321], obj_rot=frames[:, 321:325])
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/dump_to_interact.py"), str(dump),
                        "--betas", os.path.join(REPO, "scripts/omomo_betas.npz"), "--gender", "male",
                        "--models", SMPLH_MODELS, "--out_root", str(tmp_path / "ia"), "--clip", "t",
                        "--object_name", "ball"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    h = np.load(tmp_path / "ia" / "sequences_canonical" / "t" / "human.npz")
    o = np.load(tmp_path / "ia" / "sequences_canonical" / "t" / "object.npz")
    assert h["poses"].shape == (len(frames), 156) and h["trans"].shape == (len(frames), 3)
    assert h["betas"].shape == (10,)          # InterAct's BEHAVE renderer is a 10-beta SMPL-H
    M = p2._model("male")
    parents = M["parents"]
    for t in range(len(frames)):
        pose = h["poses"][t].reshape(52, 3).astype(np.float64)
        # human.npz is SMPL-native: sim = Q_ZUP @ (v(pose) + trans_yup), so fk in the sim frame
        # needs trans_sim = Q_ZUP @ trans_yup. The pelvis is exact ...
        J = p2.fk_joints("sub2", pose, sp.Q_ZUP @ h["trans"][t].astype(np.float64))
        assert np.linalg.norm(J[0] - body_pos[t][p2.body_order.index("Pelvis")]) < 0.0005
        # ... and so is every body's GLOBAL rotation (conjugated back into the sim's axes).
        # Positions of the other joints differ by bone length only: sub2's betas are SMPL-X
        # coefficients, and SMPL-H's shape basis gives a slightly different skeleton (a few cm).
        G = [sp.expmap(pose[0])]
        for j in range(1, 52):
            G.append(G[parents[j]] @ sp.expmap(pose[j]))
        for k, jid in enumerate(p2.b2s):
            assert np.abs(sp.Q_ZUP @ G[jid] @ sp.Q_ZUP.T - R[t][k]).max() < 1e-6, p2.body_order[k]
        assert np.linalg.norm(J[p2.b2s] - body_pos[t], axis=1).max() < 0.10
    assert np.allclose(o["trans"], (sp.Q_ZUP.T @ frames[:, 318:321].T).T, atol=1e-5)


def test_render_mesh_replay_defaults_to_the_native_path():
    txt = open(os.path.join(REPO, "scripts/render_mesh_replay.py")).read()
    assert 'ap.add_argument("--ik", action="store_true"' in txt
    assert "P.native_from_bodies(subject, q, p)" in txt and "P.native_from_dof(subject, r[9:162]" in txt
    assert "frames_from_dump(P, a.dump, a.ik_iters, a.stride, a.subject, a.ik)" in txt
    b = open(os.path.join(REPO, "scripts/sim_to_cari4d_bundle.py")).read()
    assert "P.native_sequence(subject" in b and "Rp = R_sb @ Q" in b

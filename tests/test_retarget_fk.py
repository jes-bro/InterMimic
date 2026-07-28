"""Pin the clip layout and FK conventions that retarget_contact.py depends on.

These are the checks whose absence let a root-local write ship: the existing
--selftest compares FK against FK, so any convention error shared by both sides
cancels out. Everything here compares against the DATA instead.

Needs real OMOMO clips + the per-subject MJCFs; skips cleanly without them.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_retarget_fk.py -q
"""

import glob
import importlib.util
import os

import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..")
MOTION_DIRS = [os.path.expanduser("~/new_one/OMOMO_new"),
               os.path.join(REPO, "InterAct/OMOMO_new")]


def _load_rc():
    path = os.path.join(REPO, "scripts", "retarget_contact.py")
    spec = importlib.util.spec_from_file_location("retarget_contact", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_rc()


def _clip_for(subject):
    """First clip of `subject`, or None if the dataset isn't on this machine."""
    for d in MOTION_DIRS:
        hits = sorted(glob.glob(os.path.join(d, f"{subject}_*.pt")))
        if hits:
            return hits[0]
    return None


def _have_mjcf(subject):
    return os.path.exists(os.path.join(REPO, rc.MJCF % subject))


SUBJECTS = ["sub2", "sub6", "sub9"]
CASES = [s for s in SUBJECTS if _clip_for(s) and _have_mjcf(s)]
needs_data = pytest.mark.skipif(not CASES, reason="OMOMO_new clips / MJCFs not present")


@pytest.fixture(scope="module")
def cwd_repo():
    """rc.MJCF is a repo-relative path, so the chain must be built from the root."""
    old = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(old)


def _load(subject):
    clip = torch.load(_clip_for(subject), map_location="cpu", weights_only=False).double()
    return clip, clip.shape[0]


@needs_data
@pytest.mark.parametrize("subject", CASES)
def test_root_rot_is_unit_quaternion_with_zero_pad(subject, cwd_repo):
    """Cols 3:7 are a unit quaternion and 7:9 are zero -- NOT a 6D rotation.

    The old header called 3:9 a 6D rotation; building a matrix from those six
    numbers gives a non-orthonormal frame, which is how the bug hid.
    """
    clip, _ = _load(subject)
    q = clip[:, rc.I_ROOTQ]
    assert torch.allclose(q.norm(dim=-1), torch.ones(len(q), dtype=torch.float64), atol=1e-6)
    assert clip[:, 7:9].abs().max() == 0.0
    # the discredited 6D reading is not even a rotation
    r0, r1 = clip[:, 3:6], clip[:, 6:9]
    fake = torch.stack([r0, r1, torch.cross(r0, r1, dim=-1)], dim=-2)
    off = (fake @ fake.transpose(-1, -2) - torch.eye(3, dtype=torch.float64)).abs().max()
    assert off > 0.1, "6D reading looks orthonormal; revisit the layout assumption"


@needs_data
@pytest.mark.parametrize("subject", CASES)
def test_fk_with_root_reproduces_stored_body_pos(subject, cwd_repo):
    """THE gate: FK(dof, root) must equal the clip's own world-frame body_pos."""
    clip, T = _load(subject)
    chain = rc.MJCFChain(subject)
    p = chain.fk(clip[:, rc.I_DOF],
                 root_pos=clip[:, rc.I_ROOTP],
                 root_rot=rc.quat_xyzw_to_mat(clip[:, rc.I_ROOTQ]))
    stored = clip[:, rc.I_BODY].reshape(T, rc.NB, 3)
    err = (p - stored).norm(dim=-1)
    assert err.mean() < 0.001, f"{subject}: FK off by {err.mean()*1000:.3f} mm mean"
    assert err.max() < 0.005, f"{subject}: FK off by {err.max()*1000:.3f} mm max"


@needs_data
@pytest.mark.parametrize("subject", CASES)
def test_rootless_fk_is_wildly_wrong(subject, cwd_repo):
    """Root-local FK must NOT be mistaken for world -- this is the shipped bug.

    Guards against someone 'simplifying' the root back out of the solve.
    """
    clip, T = _load(subject)
    p_local = rc.MJCFChain(subject).fk(clip[:, rc.I_DOF])
    stored = clip[:, rc.I_BODY].reshape(T, rc.NB, 3)
    assert (p_local - stored).norm(dim=-1).mean() > 0.5, \
        "root-local FK unexpectedly close to world; the frames may have changed"


@needs_data
def test_retarget_output_is_world_frame_and_identity_is_a_noop(cwd_repo):
    """sub2 -> sub2 must return the clip essentially unchanged, IN WORLD FRAME.

    The old code passed an identity selftest while writing root-local body_pos,
    because it only compared FK to FK. Here the output is checked against the
    input clip, which is world-frame by definition.
    """
    subject = CASES[0]
    clip, T = _load(subject)
    clip = clip[:20]                        # a few frames keeps the solve quick
    out, stats = rc.retarget(clip, subject, subject, iters=5, device="cpu", verbose=False)
    body_err = (out[:, rc.I_BODY].double() - clip[:, rc.I_BODY]).reshape(-1, rc.NB, 3).norm(dim=-1)
    assert body_err.mean() < 0.005, f"identity retarget moved body_pos {body_err.mean()*1000:.2f} mm"
    dof_err = (out[:, rc.I_DOF].double() - clip[:, rc.I_DOF]).abs().max()
    assert dof_err < 0.05, f"identity retarget changed dof by {dof_err:.4f} rad"
    # fields the solve does not own must be byte-identical
    for name, sl in [("root_pos", rc.I_ROOTP), ("root_rot", rc.I_ROOTQ),
                     ("obj_pos", rc.I_OBJP), ("contact_h", rc.I_CONTACT_H)]:
        assert torch.equal(out[:, sl].double(), clip[:, sl]), f"{name} was modified"


@needs_data
def test_fk_gate_rejects_mismatched_subject(cwd_repo):
    """Feeding a clip of the wrong subject must raise, not silently retarget."""
    if len(CASES) < 2:
        pytest.skip("need two subjects with data to cross them")
    clip, _ = _load(CASES[0])
    with pytest.raises(RuntimeError, match="does not reproduce"):
        rc.retarget(clip[:10], CASES[1], CASES[1], iters=1, device="cpu", verbose=False)

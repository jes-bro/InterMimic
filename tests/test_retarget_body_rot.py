#!/usr/bin/env python3
"""body_rot must follow the re-solved dof, not stay at the source's values.

Before this fix `retarget()` wrote only dof (9:162) and body_pos (162:318); the
`out = clip.clone()` carried body_rot (383:591) through untouched. The loader reads
body_rot from those columns (intermimic.py:706) and the reward's `rr` factor scores
the simulated rotations against them -- so the reference described one pose with its
positions and a different pose with its rotations.

It is specifically a regression rather than a pre-existing wart because global body
rotations are SHAPE-INVARIANT: Rg[b] = Rg[parent] @ expmap(dof_b), and bone offsets
enter only the position recursion. Un-retargeted, every body could hit the reference
rotations exactly (test_rotations_are_shape_invariant). Re-solving dof breaks that
unless the field is rewritten.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_retarget_body_rot.py -v
"""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from retarget_contact import (  # noqa: E402
    I_BODY, I_BODYROT, I_DOF, I_ROOTP, I_ROOTQ, NB,
    MJCFChain, _geodesic_deg, mat_to_quat_xyzw, quat_xyzw_to_mat, retarget,
)

CLIP = os.path.join(REPO, "InterAct/OMOMO_new/sub2_largetable_000.pt")
SOURCE = "sub2"
ITERS = 120          # enough to move dof meaningfully; keeps the suite ~1 min


@pytest.fixture(scope="module")
def clip():
    if not os.path.exists(CLIP):
        pytest.skip(f"SKIP: {CLIP} not present")
    return torch.load(CLIP, map_location="cpu", weights_only=False).detach().double()


@pytest.fixture(scope="module")
def root(clip):
    return clip[:, I_ROOTP], quat_xyzw_to_mat(clip[:, I_ROOTQ])


@pytest.fixture(scope="module")
def retargeted(clip):
    """One real cross-body solve, shared by the tests that inspect its output."""
    out, stats = retarget(clip, SOURCE, "sub9", iters=ITERS, verbose=False)
    return out.double(), stats


def _stored_rot(c):
    return quat_xyzw_to_mat(c[:, I_BODYROT].reshape(-1, 4)).reshape(-1, NB, 3, 3)


# ------------------------------------------------------------------ conventions
def test_quat_roundtrip_is_exact():
    """mat_to_quat_xyzw must invert quat_xyzw_to_mat, including the 180-degree branch."""
    torch.manual_seed(0)
    q = torch.randn(4096, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    back = mat_to_quat_xyzw(quat_xyzw_to_mat(q), like=q)
    assert torch.allclose(back, q, atol=1e-9), \
        f"round-trip max err {(back - q).abs().max():.2e}"


def test_stored_body_rot_matches_fk_of_stored_dof(clip, root):
    """The gate's premise: for an untouched clip, FK rotations == stored body_rot."""
    root_p, root_R = root
    _, R = MJCFChain(SOURCE).fk(clip[:, I_DOF], root_pos=root_p, root_rot=root_R,
                                return_rot=True)
    err = _geodesic_deg(_stored_rot(clip), R)
    assert err.mean() < 0.05, f"stored body_rot vs FK: {err.mean():.4f} deg"


def test_rotations_are_shape_invariant(clip, root):
    """Same dof on a different body gives the same global rotations -- which is why
    the un-retargeted reference was exactly reachable and a stale one is not.

    Asserted elementwise, not as an angle: the arccos in _geodesic_deg loses precision
    near identity (it reports ~0.035 deg even for a tensor against itself), so an
    angular threshold could not express "exactly equal".
    """
    root_p, root_R = root
    _, Ra = MJCFChain("sub2").fk(clip[:, I_DOF], root_pos=root_p, root_rot=root_R,
                                 return_rot=True)
    _, Rb = MJCFChain("sub16").fk(clip[:, I_DOF], root_pos=root_p, root_rot=root_R,
                                  return_rot=True)
    assert (Ra - Rb).abs().max() == 0.0, "rotations must not depend on bone length at all"


# --------------------------------------------------------------- the actual fix
def test_written_body_rot_is_consistent_with_written_dof(retargeted, root):
    """THE fix: the emitted body_rot is the FK of the emitted dof, not the source's."""
    out, _ = retargeted
    root_p, root_R = root
    _, R_expected = MJCFChain("sub9").fk(out[:, I_DOF], root_pos=root_p, root_rot=root_R,
                                         return_rot=True)
    err = _geodesic_deg(_stored_rot(out), R_expected)
    assert err.mean() < 0.05, \
        f"written body_rot disagrees with FK of written dof by {err.mean():.4f} deg"


def test_body_rot_actually_changed(clip, retargeted):
    """Guard against a silent no-op: the field must differ from the source's."""
    out, stats = retargeted
    moved = _geodesic_deg(_stored_rot(clip), _stored_rot(out))
    assert moved.mean() > 0.2, (
        f"body_rot moved only {moved.mean():.4f} deg -- if the solve changed dof, the "
        "rotations must have changed too; suspect the write was dropped")
    assert stats["body_rot_shift_deg"] == pytest.approx(moved.mean().item(), abs=0.05)


def test_written_quaternions_are_unit_and_sign_matched(clip, retargeted):
    """Unit norm, and on the same hemisphere as the values they replace -- the reward
    decodes raw components, so a flipped sign would read as a large rotation error."""
    out, _ = retargeted
    q = out[:, I_BODYROT].reshape(-1, 4)
    assert torch.allclose(q.norm(dim=-1), torch.ones_like(q[:, 0]), atol=1e-6)
    dot = (q * clip[:, I_BODYROT].reshape(-1, 4)).sum(-1)
    assert (dot >= 0).all(), f"{(dot < 0).sum()} quaternions flipped hemisphere"


def test_identity_retarget_leaves_body_rot_alone(clip):
    """sub2 -> sub2 changes nothing, so body_rot must come back essentially unchanged."""
    out, _ = retarget(clip, SOURCE, SOURCE, iters=40, verbose=False)
    err = _geodesic_deg(_stored_rot(clip), _stored_rot(out.double()))
    assert err.mean() < 0.5, f"identity retarget moved body_rot by {err.mean():.4f} deg"


def test_other_columns_are_untouched(clip, retargeted):
    """Only dof, body_pos and body_rot may change -- contacts and object data carry through."""
    out, _ = retargeted
    for name, sl in [("contact/obj block", slice(318, 331)),
                     ("contact_human", slice(331, 383)),
                     ("root", slice(0, 9))]:
        assert torch.allclose(out[:, sl], clip[:, sl].float().double(), atol=1e-6), \
            f"{name} was modified"

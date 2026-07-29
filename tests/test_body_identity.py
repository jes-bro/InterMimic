#!/usr/bin/env python3
"""Proof that our source motions and target bodies are NOT retargeted onto sub9.

The claim under test has two halves, and each is checked against the artifacts
themselves rather than against config intent:

  A. TARGET BODIES   -- the MJCFs we simulate are distinct per subject.
  B. SOURCE MOTIONS  -- the reference in OMOMO_new is authored per subject; each
                        subject's body_pos matches its OWN skeleton, not sub9's.
  C. POSITIVE CONTROL-- OMOMO_retarget (upstream's teacher-corrected drop) IS on a
                        single body, and this same test detects that. Without C, a
                        passing A/B would be worthless: a test that cannot detect
                        the failure it rules out has proven nothing.
  D. CONFIGS         -- the cfgs we train read OMOMO_new, never the sub9-body drop.

Everything is offline (no Isaac Gym). Run:
    python3 -m pytest tests/test_body_identity.py -v
"""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from body_identity import (  # noqa: E402
    FULL_OMOMO_NEW, REPO, REPO_OMOMO_RETARGET,
    available_mjcf_subjects, best_match, identity_table, match_errors,
    mjcf_bone_lengths, subject_bone_lengths, subjects_in,
)

CFG_DIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

# A rigid skeleton's bone lengths do not move at all between frames. 0.01 mm is
# far above float32 storage noise and far below any real body difference.
RIGID_TOL_MM = 0.01
# How precisely we can read a body out of a clip: measured own-body fits land at
# 0.02-0.03 mm, so 0.05 mm is the noise floor of the whole method.
MEASURE_PRECISION_MM = 0.05
# Two bodies count as resolvably different once they are 10x that floor apart.
# The closest real pair is sub12/sub13 at 0.863 mm -- genuinely similar builds,
# but still 30x the precision floor, so identity stays recoverable (see A1b).
RESOLVABLE_FLOOR_MM = 10 * MEASURE_PRECISION_MM
# For "is this reference on sub9's body?", 1 mm is a conservative separation.
DISTINCT_FLOOR_MM = 1.0


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def real_subjects():
    return available_mjcf_subjects(real_only=True)


@pytest.fixture(scope="module")
def omomo_new():
    """The dataset training actually reads. Skips LOUDLY if the full copy is absent."""
    if not glob.glob(os.path.join(FULL_OMOMO_NEW, "*.pt")):
        pytest.skip(f"SKIP: full OMOMO_new not found at {FULL_OMOMO_NEW} -- "
                    "this test needs all 17 subjects; the repo's bundled copy is sub2-only")
    return identity_table(FULL_OMOMO_NEW)


@pytest.fixture(scope="module")
def omomo_retarget():
    if not glob.glob(os.path.join(REPO_OMOMO_RETARGET, "*.pt")):
        pytest.skip(f"SKIP: {REPO_OMOMO_RETARGET} not present")
    return identity_table(REPO_OMOMO_RETARGET)


def _cfgs():
    return sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml")))


def _cfg_text(path):
    with open(path) as fh:
        return fh.read()


# ============================================================ A. TARGET BODIES
def _closest_pair(names, vec_of):
    """(distance_mm, (a, b)) for the two most similar bodies in a set."""
    return min((float(np.abs(vec_of(a) - vec_of(b)).mean() * 1000), (a, b))
               for i, a in enumerate(names) for b in names[i + 1:])


def test_A1_real_subject_mjcfs_are_pairwise_distinct(real_subjects):
    """The 17 per-subject MJCFs are 17 different bodies, not 17 copies of one."""
    d, pair = _closest_pair(real_subjects, mjcf_bone_lengths)
    assert d > RESOLVABLE_FLOOR_MM, (
        f"closest pair {pair} differs by only {d:.3f} mm -- below the "
        f"{RESOLVABLE_FLOOR_MM} mm resolvability floor, bodies not distinct")


def test_A1b_closest_real_pair_is_sub12_sub13(real_subjects):
    """Pin the closest real pair, so a future data change that collapses bodies shows up.

    sub12 and sub13 are genuinely similar builds (0.863 mm apart) -- still 30x the
    measurement floor, which is why test_B2 can tell even those two apart.
    """
    d, pair = _closest_pair(real_subjects, mjcf_bone_lengths)
    assert set(pair) == {"sub12", "sub13"}, f"closest pair is now {pair} at {d:.3f} mm"
    assert 0.8 < d < 0.95, f"sub12/sub13 separation moved to {d:.3f} mm (was 0.863)"


def test_A2_canonical_omomo_xml_is_sub9(real_subjects):
    """omomo.xml -- upstream's single canonical humanoid -- IS sub9's body, exactly.

    This is the fact that makes the whole question worth asking, so it is pinned.
    """
    errs = match_errors(mjcf_bone_lengths("omomo"), real_subjects)
    best = min(errs, key=errs.get)
    assert best == "sub9", f"omomo.xml best-matches {best}, not sub9"
    assert errs["sub9"] < 1e-6, f"omomo.xml vs sub9 = {errs['sub9']:.6f} mm, expected exact"
    runner_up = sorted(e for s, e in errs.items() if s != "sub9")[0]
    assert runner_up > DISTINCT_FLOOR_MM, "sub9 is not uniquely the canonical body"


def test_A3_training_cfg_subject_bodies_resolve_to_distinct_mjcfs():
    """Every body listed in a cfg's subjectBodies exists on disk and is distinct.

    If the sim were secretly running one body, these would collapse to one file.
    """
    import re
    checked = 0
    for cfg in _cfgs():
        m = re.search(r"^\s*subjectBodies:\s*\[(.*?)\]", _cfg_text(cfg), re.M | re.S)
        if not m:
            continue
        bodies = re.findall(r"'(sub\d+)'", m.group(1))
        if len(bodies) < 2:
            continue
        checked += 1
        assert len(set(bodies)) == len(bodies), f"{os.path.basename(cfg)}: duplicate bodies"
        vecs = {}
        for b in bodies:
            path = os.path.join(REPO, "isaacgym/src/intermimic/data/assets/smplx",
                                f"smplx_omomo_{b}.xml")
            assert os.path.exists(path), f"{os.path.basename(cfg)}: missing MJCF for {b}"
            vecs[b] = mjcf_bone_lengths(b)
        stacked = np.array([vecs[b] for b in bodies])
        spread = float(np.abs(stacked - stacked.mean(0)).mean() * 1000)
        assert spread > DISTINCT_FLOOR_MM, (
            f"{os.path.basename(cfg)}: {len(bodies)} bodies but mean spread only "
            f"{spread:.3f} mm -- they are effectively one body")
    assert checked > 0, "no cfg with subjectBodies found -- fixture is not exercising anything"


# =========================================================== B. SOURCE MOTIONS
def test_B1_omomo_new_body_pos_is_a_rigid_skeleton(omomo_new):
    """Precondition: body_pos must BE a rigid body for the identity test to mean anything."""
    for sub, r in omomo_new.items():
        assert r["max_std_mm"] < RIGID_TOL_MM, (
            f"{sub}: bone lengths drift {r['max_std_mm']:.4f} mm within a clip -- "
            "body_pos is not a rigid skeleton, identity claims below would be unsound")


def test_B2_every_subject_matches_its_OWN_body(omomo_new):
    """The decisive test: argmin over all 17 candidate bodies is the subject itself."""
    for sub, r in omomo_new.items():
        assert r["best"] == sub, (
            f"{sub}'s reference best-matches {r['best']} ({r['best_err']:.3f} mm), not itself")
        assert r["best_err"] < 0.1, f"{sub}: own-body error {r['best_err']:.3f} mm is too loose"


def test_B3_source_motions_are_NOT_on_sub9(omomo_new):
    """The claim in the title: for every subject except sub9, sub9 is a much worse fit."""
    for sub, r in omomo_new.items():
        if sub == "sub9":
            continue
        own, to9 = r["errors"][sub], r["errors"]["sub9"]
        assert to9 > DISTINCT_FLOOR_MM, (
            f"{sub}: error against sub9's body is only {to9:.3f} mm -- indistinguishable "
            "from being retargeted onto sub9")
        assert to9 > 20 * own, (
            f"{sub}: sub9 fits nearly as well as its own body (own {own:.3f} mm vs "
            f"sub9 {to9:.3f} mm)")


def test_B4_subjects_do_not_share_one_skeleton(omomo_new):
    """If everything had been retargeted to one body, all subjects would measure alike."""
    d, pair = _closest_pair(list(omomo_new), lambda s: omomo_new[s]["measured"])
    assert d > RESOLVABLE_FLOOR_MM, (
        f"closest two subjects' references ({pair}) differ by {d:.3f} mm -- consistent "
        "with a single shared body")
    # And the spread is body-sized, not noise-sized: the widest pair is ~24 mm.
    widest = max(
        float(np.abs(omomo_new[a]["measured"] - omomo_new[b]["measured"]).mean() * 1000)
        for a in omomo_new for b in omomo_new)
    assert widest > 20, f"widest pair only {widest:.1f} mm apart -- expected ~24 mm"


# ========================================================= C. POSITIVE CONTROL
def test_C1_control_omomo_retarget_IS_a_single_body(omomo_retarget):
    """Upstream's teacher-corrected drop IS retargeted onto one body -- and this
    test detects it. Establishes the tests above can fail when the claim is false."""
    subs = list(omomo_retarget)
    assert len(subs) >= 2, f"control needs >=2 subjects, got {subs}"
    spread = max(
        float(np.abs(omomo_retarget[a]["measured"] - omomo_retarget[b]["measured"]).mean() * 1000)
        for i, a in enumerate(subs) for b in subs[i + 1:])
    assert spread < DISTINCT_FLOOR_MM, (
        f"control: subjects differ by {spread:.3f} mm -- expected them to share one body")


def test_C2_control_that_single_body_is_sub9(omomo_retarget, real_subjects):
    """And the shared body is sub9 -- i.e. upstream really does retarget to one subject."""
    for sub, r in omomo_retarget.items():
        best, err = best_match(r["measured"], real_subjects)
        assert best == "sub9", f"control {sub}: shared body is {best}, expected sub9"
        assert err < 1.0, f"control {sub}: sub9 fit {err:.3f} mm is looser than expected"


def test_C3_control_and_training_data_disagree(omomo_new, omomo_retarget, real_subjects):
    """Same subject, two datasets, opposite answers -- the discriminator works."""
    for sub in set(omomo_new) & set(omomo_retarget):
        train_best, _ = best_match(omomo_new[sub]["measured"], real_subjects)
        ctrl_best, _ = best_match(omomo_retarget[sub]["measured"], real_subjects)
        assert train_best == sub and ctrl_best == "sub9", (
            f"{sub}: OMOMO_new->{train_best}, OMOMO_retarget->{ctrl_best} "
            "(expected own-body and sub9 respectively)")


# ================================================================== D. CONFIGS
def test_D1_training_cfgs_never_read_the_sub9_body_dataset():
    """No cfg that sets subjectBodies reads OMOMO_retarget as its motion_file."""
    import re
    offenders = []
    for cfg in _cfgs():
        txt = _cfg_text(cfg)
        if not re.search(r"^\s*subjectBodies:", txt, re.M):
            continue
        mf = re.search(r"^\s*motion_file:\s*(\S+)", txt, re.M)
        assert mf, f"{os.path.basename(cfg)}: sets subjectBodies but no motion_file"
        if "OMOMO_retarget" in mf.group(1):
            offenders.append(os.path.basename(cfg))
    assert not offenders, f"per-body cfgs reading the sub9-body dataset: {offenders}"


def test_D2_per_body_cfgs_never_point_EITHER_motion_key_at_the_sub9_dataset():
    """motion_file_retarget also feeds the reference (intermimic_all.py:103,116), so
    its VALUE matters, not merely its presence. Our distill cfgs set the key but aim
    it at OMOMO_new -- the per-subject data -- which is the thing to lock in."""
    import re
    offenders = []
    for cfg in _cfgs():
        txt = _cfg_text(cfg)
        if not re.search(r"^\s*subjectBodies:", txt, re.M):
            continue
        for key in ("motion_file", "motion_file_retarget"):
            m = re.search(rf"^\s*{key}:\s*(\S+)", txt, re.M)
            if m and "OMOMO_retarget" in m.group(1):
                offenders.append(f"{os.path.basename(cfg)}:{key}")
    assert not offenders, f"per-body cfgs reading the sub9-body dataset: {offenders}"


def test_D3_only_upstream_cfgs_read_the_sub9_body_dataset():
    """Pin who uses OMOMO_retarget at all: upstream's three omomo_all* cfgs, none of
    which set subjectBodies. Any new cfg reaching for it will fail here."""
    import re
    users = set()
    for cfg in _cfgs():
        txt = _cfg_text(cfg)
        for key in ("motion_file", "motion_file_retarget"):
            m = re.search(rf"^\s*{key}:\s*(\S+)", txt, re.M)
            if m and "OMOMO_retarget" in m.group(1):
                users.add(os.path.basename(cfg))
    assert users == {"omomo_all.yaml", "omomo_all_test.yaml", "omomo_all_transformer.yaml",
                     "omomo_g1_29dof_with_hand.yaml"}, f"unexpected OMOMO_retarget users: {sorted(users)}"

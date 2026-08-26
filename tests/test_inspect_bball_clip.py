#!/usr/bin/env python3
"""Tests for scripts/inspect_bball_clip.py's logic on a fabricated clip --
the data is cluster-only, so pin the math here: span grouping, wrist-index
resolution by name, and the hand-distance/lowest-z extraction.

Run:  python tests/test_inspect_bball_clip.py   (exit 0 = all green)
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import inspect_bball_clip as ib  # noqa: E402


def test_contact_spans():
    flags = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
    assert ib.contact_spans(flags) == [(0, 1, True), (2, 4, False),
                                       (5, 7, True), (8, 8, False)]
    assert ib.contact_spans(torch.zeros(3, dtype=torch.bool)) == [(0, 2, False)]
    print("ok: contact span grouping")


def test_wrist_indices_from_real_mjcf():
    # any per-subject MJCF present locally works; skip loudly if none rcloned
    import glob
    mjcfs = glob.glob(os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml"))
    if not mjcfs:
        print("SKIP: no per-subject MJCFs present locally")
        return
    idx, names = ib.wrist_indices(mjcfs[0])
    assert names[idx[0]] == "L_Wrist" and names[idx[1]] == "R_Wrist"
    assert len(names) == 52
    print(f"ok: wrist indices resolved by name ({idx})")


def test_extraction_math():
    # fabricate a 2-frame clip: known ball + body positions
    T = 2
    c = torch.zeros(T, 591)
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0            # all bodies at z=1
    bp[0, 5, 2] = 0.23           # frame 0: lowest body at the floor-offset height
    bp[1, 5, 2] = 0.0
    bp[0, 16] = torch.tensor([1.0, 0.0, 1.0])   # a "wrist" at x=1
    c[:, ib.I_BODY] = bp.view(T, -1)
    c[0, ib.I_OBJP] = torch.tensor([1.0, 0.0, 1.3])  # ball 0.3 above that wrist
    c[0, ib.I_CONTACT_OBJ] = 1.0
    bpv = c[:, ib.I_BODY].view(T, 52, 3)
    lowest = bpv[:, :, 2].min(dim=1).values
    assert abs(lowest[0] - 0.23) < 1e-6 and abs(lowest[1] - 0.0) < 1e-6
    d = (bpv[0, [16], :] - c[0, ib.I_OBJP][None, :]).norm(dim=-1).min()
    assert abs(d - 0.3) < 1e-6
    assert bool(c[0, ib.I_CONTACT_OBJ] > 0.5) and not bool(c[1, ib.I_CONTACT_OBJ] > 0.5)
    print("ok: lowest-z / hand-distance / flag extraction")


def test_runs_where():
    assert ib.runs_where([0, 1, 1, 0, 0, 1]) == [(1, 2), (5, 5)]
    assert ib.runs_where([0, 0]) == []
    assert ib.runs_where([1, 1]) == [(0, 1)]
    print("ok: contiguous run extraction")


def test_penetration_spans():
    # Two dips below the floor, separated by a grounded stretch. Depth is
    # reported as a POSITIVE distance below z=0, at the frame where it is worst.
    lowest = torch.tensor([0.05, -0.02, -0.07, -0.03, 0.01, 0.04, -0.01, 0.02])
    spans = ib.penetration_spans(lowest)
    assert len(spans) == 2, spans
    s, e, depth, jf = spans[0]
    assert (s, e, jf) == (1, 3, 2), spans[0]
    assert abs(depth - 0.07) < 1e-6, depth
    assert spans[1][:2] == (6, 6)
    # A clip that never penetrates must report nothing at all -- this is the
    # branch that says "no correction needed", so a false positive here would
    # send someone editing data that is fine.
    assert ib.penetration_spans(torch.tensor([0.0, 0.05, 0.2])) == []
    print("ok: penetration spans (depth, deepest frame, empty case)")


def test_flight_spans():
    # A jump: grounded, climb through the 0.10 threshold, apex, land. The span
    # must START at the takeoff (frame 2), not partway up the climb -- that is
    # why the entry threshold stays low while the APEX filter does the rejecting.
    lowest = torch.tensor([0.02, 0.03, 0.12, 0.30, 0.54, 0.31, 0.09, 0.02])
    assert ib.flight_spans(lowest) == [(2, 5)]
    # Foot-lift jitter (a 2-frame blip) must NOT register as a jump.
    assert ib.flight_spans(torch.tensor([0.0, 0.15, 0.15, 0.0])) == []
    # A 3-frame excursion that never gets high is ALSO not a jump: the bball
    # clip's unreliable tail produced two of these (apex 0.200 and 0.187) and
    # they reported nonsense crouches, including a 0.572 m "pelvis dip".
    assert ib.flight_spans(torch.tensor([0.0, 0.15, 0.15, 0.15, 0.0])) == []
    assert ib.flight_spans(torch.tensor([0.0, 0.20, 0.19, 0.20, 0.0])) == []
    # ...but a 3-frame excursion that DOES clear the apex bar is kept.
    assert ib.flight_spans(torch.tensor([0.0, 0.15, 0.30, 0.15, 0.0])) == [(1, 3)]
    # And the apex bar is tunable rather than baked in.
    assert ib.flight_spans(torch.tensor([0.0, 0.15, 0.15, 0.15, 0.0]),
                           min_apex=0.10) == [(1, 3)]
    print("ok: flight spans (low entry threshold, apex filter, min length)")


def test_named_indices():
    names = ["Pelvis", "L_Knee", "R_Knee", "L_Wrist"]
    idx, missing = ib.named_indices(names, ["L_Knee", "R_Knee"])
    assert idx == [1, 2] and missing == []
    # A missing body must degrade the column, not raise -- the report still has
    # value without knees.
    idx, missing = ib.named_indices(names, ["L_Knee", "L_Ankle"])
    assert idx == [1] and missing == ["L_Ankle"]
    print("ok: named index resolution with graceful misses")


def test_crouch_discrimination():
    """The measurement that decides whether a coverage knob can help at all:
    a real countermovement must read differently from a rigid takeoff."""
    # Same flight in both; only the pelvis run-up differs.
    lowest = torch.tensor([0.02] * 5 + [0.12, 0.30, 0.54, 0.30, 0.05])
    (s, e), = ib.flight_spans(lowest)
    assert (s, e) == (5, 8)

    crouched = torch.tensor([0.95, 0.92, 0.84, 0.78, 0.88] + [1.0] * 5)
    rigid = torch.tensor([0.95, 0.95, 0.94, 0.95, 0.95] + [1.0] * 5)
    w0 = max(0, s - 15)
    dip_c = float(crouched[w0:s].max()) - float(crouched[w0:s].min())
    dip_r = float(rigid[w0:s].max()) - float(rigid[w0:s].min())
    assert dip_c >= 0.10, dip_c      # crosses the "crouch present" bar
    assert dip_r < 0.05, dip_r       # falls in the "shallow/absent" bucket
    print(f"ok: crouch vs rigid separate ({dip_c:.2f} m vs {dip_r:.2f} m)")


def test_lowest_body_identification():
    """Whole-body sink vs one limb dipping want different corrections, so the
    report must name the offending body and track the pelvis independently."""
    T = 3
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0
    bp[:, 0, 2] = 0.9            # pelvis
    bp[0, 7, 2] = -0.05          # frame 0: one body dips below the floor
    bp[1, 9, 2] = -0.02          # frame 1: a DIFFERENT body is lowest
    bp[2, :, 2] = 0.3            # frame 2: whole body low, nothing penetrating
    bp[2, 0, 2] = 0.3
    lowest = bp[:, :, 2].min(dim=1).values
    which = bp[:, :, 2].argmin(dim=1)
    assert int(which[0]) == 7 and int(which[1]) == 9
    spans = ib.penetration_spans(lowest)
    assert len(spans) == 1 and spans[0][:2] == (0, 1)
    # pelvis held steady across the penetration -> limb-local, not a sink
    assert abs(float(bp[0:2, 0, 2].min()) - 0.9) < 1e-6
    print("ok: lowest-body identity and pelvis-independent tracking")


def test_hand_contact_geometry():
    """The check that separates a data problem from a policy problem.

    rcg_hand grades the sim's per-body contact against the reference's
    contact_human channel. If the reference flags hand contact on a frame where
    no hand body is near the ball surface, rcg_hand is unearnable there -- the
    policy would have to be where the reference says AND touching the ball, and
    those are different places. That is a relabel/data fix, not a training knob.
    """
    R = 0.13                                   # ball radius
    T = 4
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0
    obj = torch.zeros(T, 3)
    # frame 0: a finger body ON the surface while flagged -> earnable
    bp[0, 20] = torch.tensor([R, 0.0, 0.0])
    # frame 1: nearest hand body 0.10 m OUTSIDE the surface while flagged
    bp[1, 20] = torch.tensor([R + 0.10, 0.0, 0.0])
    # frame 2: hand inside the ball (interpenetrating) while flagged
    bp[2, 20] = torch.tensor([R - 0.04, 0.0, 0.0])
    # frame 3: hand far away, NOT flagged -> must not count against the data
    bp[3, 20] = torch.tensor([2.0, 0.0, 0.0])
    ch = torch.zeros(T, 52)
    ch[0:3, 20] = 1.0                          # frames 0-2 claim contact

    hand_flags = ch[:, ib.HAND_BODY_IDS] > ib.CONTACT_THRES
    says = hand_flags.any(dim=1)
    gap = ((bp[:, ib.HAND_BODY_IDS, :] - obj[:, None, :]).norm(dim=-1) - R).min(dim=1).values

    assert says.tolist() == [True, True, True, False]
    g = gap[says]
    assert abs(g[0]) < 1e-5, g[0]              # on the surface
    assert abs(g[1] - 0.10) < 1e-5, g[1]       # 10 cm short -> unearnable
    assert abs(g[2] + 0.04) < 1e-5, g[2]       # interpenetrating -> fine
    # The bar is PhysX's contact_offset (0.02), not zero: a hand 1.2 cm off the
    # surface still generates a contact in sim, so rcg_hand IS earnable there.
    # Testing gap > 0 made a correctly relabelled clip read 8/40 instead of 0/40.
    assert int((g > 0.02).sum()) == 1, g
    assert abs(g[1] - 0.10) < 1e-5          # 10 cm out: beyond any tolerance
    near = torch.tensor([0.012])            # inside contact_offset
    assert int((near > 0.02).sum()) == 0, "a 1.2 cm gap must NOT count as unearnable"
    # The unflagged far frame must be excluded, or every free-flight frame would
    # be counted as a data defect and the verdict would always say DATA PROBLEM.
    # (gap is the MIN over all 32 hand bodies, so the untouched ones at 1.0 m
    # set the floor here -- the assertion is "far and unflagged", not a value.)
    assert gap[3] > 0.5 and not bool(says[3])
    print("ok: hand-contact geometry separates earnable from unearnable frames")


def test_hand_body_ids_match_the_reward():
    """HAND_BODY_IDS must be compute_cg_reward's own lists, or the diagnostic
    grades different bodies than the reward does."""
    src = open(os.path.join(
        REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")).read()
    assert "left_contact_hand_ids = list(range(17, 33))" in src
    assert "right_contact_hand_ids = list(range(36, 52))" in src
    assert ib.HAND_BODY_IDS == list(range(17, 33)) + list(range(36, 52))
    # ...and body 17/36 really are the wrists, per the MJCF itself.
    import glob
    mjcfs = glob.glob(os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml"))
    if mjcfs:
        idx, _ = ib.wrist_indices(mjcfs[0])
        assert idx == [17, 36], idx
    # contact_human's channel slice must match the task's own loader.
    assert "loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52]" in src
    assert ib.I_CONTACT_HUMAN == slice(331, 383)
    print("ok: hand body ids and contact_human slice agree with the task")


def test_end_to_end_report():
    """Run the real script on a fabricated clip. The unit tests above cover the
    math; this covers the PRINT path, where a formatting slip (or a None knee
    column) crashes the whole report on the cluster and wastes a round trip."""
    import glob
    import subprocess
    import tempfile

    mjcfs = sorted(glob.glob(os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub*.xml")))
    if not mjcfs:
        print("SKIP: no per-subject MJCFs present locally")
        return

    # A clip with all the branches live: a penetration span, a real jump with a
    # crouch, contact spans on both sides of a free-flight stretch.
    T = 40
    c = torch.zeros(T, 591)
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0
    bp[:, 0, 2] = 0.95                                   # pelvis
    feet = 7
    bp[:, feet, 2] = 0.02                                # feet on the floor
    bp[3:7, feet, 2] = -0.06                             # penetration span
    bp[20:25, 0, 2] = torch.linspace(0.95, 0.80, 5)      # crouch (pelvis dips)
    bp[25:31, feet, 2] = torch.tensor([0.15, 0.35, 0.54, 0.36, 0.14, 0.03])  # flight
    c[:, ib.I_BODY] = bp.view(T, -1)
    c[:, ib.I_OBJP] = torch.tensor([0.0, 0.0, 1.2])
    c[5:20, ib.I_CONTACT_OBJ] = 1.0
    # section 8 needs contact_human: claim hand contact on the same frames, with
    # a hand body placed well OFF the ball surface -> the DATA PROBLEM verdict.
    ch = torch.zeros(T, 52)
    ch[5:20, 20] = 1.0
    c[:, ib.I_CONTACT_HUMAN] = ch
    bp[:, 20] = torch.tensor([0.0, 0.0, 1.6])       # 0.27 m from a 0.13 m ball
    c[:, ib.I_BODY] = bp.view(T, -1)

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "sub100_bball_000.pt")
        torch.save(c, clip)
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts/inspect_bball_clip.py"),
             "--clip", clip, "--mjcf", mjcfs[0], "--every", "5"],
            capture_output=True, text=True)

    assert proc.returncode == 0, f"script exited {proc.returncode}:\n{proc.stderr}"
    out = proc.stdout
    for expected in ("== 6. floor penetration", "== 7. crouch before takeoff",
                     "== 8. reference hand-contact vs geometry",
                     "max depth 0.060 m", "lowest body", "pelvis z(m)",
                     "DATA PROBLEM"):
        assert expected in out, f"missing {expected!r} in report:\n{out}"
    # The fabricated pelvis dips 0.15 m over the run-up, so the verdict must be
    # the affirmative one -- if this flips, the threshold moved.
    assert "CROUCH PRESENT" in out, f"crouch not detected:\n{out}"
    # And the limb-vs-sink discriminator must say limb: pelvis held at 0.95.
    assert "LIMB-local artifact" in out, f"wrong penetration verdict:\n{out}"
    print("ok: end-to-end report renders, both verdicts fire correctly")


if __name__ == "__main__":
    test_contact_spans()
    test_wrist_indices_from_real_mjcf()
    test_extraction_math()
    test_runs_where()
    test_penetration_spans()
    test_flight_spans()
    test_named_indices()
    test_crouch_discrimination()
    test_lowest_body_identification()
    test_hand_contact_geometry()
    test_hand_body_ids_match_the_reward()
    test_end_to_end_report()
    print("ALL GREEN")

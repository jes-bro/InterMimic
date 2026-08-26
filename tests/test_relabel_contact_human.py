#!/usr/bin/env python3
"""Tests for scripts/relabel_contact_human.py on fabricated clips.

The real data is cluster-only, and this script REWRITES a dataset, so the
properties that matter are the ones that keep it from quietly corrupting one:

  1. it relabels the right 32 channels and nothing else  -> test_scope
  2. it preserves the clip's own tri-state convention    -> test_levels
  3. the flag it writes matches the geometry             -> test_geometry
  4. it REFUSES a clip the recon never makes contact in  -> test_guard
  5. positions are never touched                          -> test_scope
  6. both contact channels agree afterwards               -> test_channels_agree

(4) is Jess's own criterion: relabelling presumes contact exists somewhere. On a
clip where it does not, an all-free relabel would delete the contact supervision
and look like a successful fix.

Run:  python tests/test_relabel_contact_human.py   (exit 0 = all green)
"""
import os
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import relabel_contact_human as rc  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts/relabel_contact_human.py")
R = 0.13

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def make_clip(T=8, touch_frames=(2, 3, 4), contact_v=1.0, free_v=-1.0,
              stale_claim=(6,)):
    """A clip where the hand genuinely touches on `touch_frames`, but the stored
    flags ALSO claim contact on `stale_claim` (the real defect: a relabel that
    rewrote contact_obj and left contact_human carrying the old story)."""
    t = torch.zeros(T, 591)
    bp = torch.zeros(T, 52, 3)
    bp[:, :, 2] = 1.0
    t[:, rc.I_OBJP] = torch.tensor([0.0, 0.0, 0.0])
    # Park every hand body far away, then bring body 20 in on the touch frames.
    for b in rc.HAND_BODY_IDS:
        bp[:, b] = torch.tensor([3.0, 0.0, 0.0])
    for f in touch_frames:
        bp[f, 20] = torch.tensor([R - 0.01, 0.0, 0.0])      # 1 cm inside
    t[:, rc.I_BODY] = bp.view(T, -1)

    ch = torch.full((T, 52), free_v)
    for f in list(touch_frames) + list(stale_claim):
        ch[f, 20] = contact_v
    # Non-hand bodies carry floor/self contact this script must not touch.
    ch[:, 5] = contact_v
    ch[:, 6] = 0.0
    t[:, rc.I_CONTACT_HUMAN] = ch
    return t


def test_geometry():
    print("1. the written flag follows the geometry:")
    t = make_clip()
    out, touching = rc.relabel(t, R, threshold=0.02, smooth=1,
                               keep_contact_obj=False)
    got = touching.any(dim=1).nonzero().flatten().tolist()
    check("contact exactly on the frames a hand body is within threshold",
          got == [2, 3, 4], f"(got {got})")
    # The stale claim at frame 6 must be GONE -- that is the whole point.
    ch = out[:, rc.I_CONTACT_HUMAN]
    check("the stale unearnable claim is cleared", float(ch[6, 20]) < 0,
          f"(frame 6 body 20 = {float(ch[6, 20])})")
    check("a genuine contact frame is kept", float(ch[3, 20]) > 0)

    # A body that never touches must never be flagged, even on contact frames --
    # rcg_hand grades per body, so flagging all 32 would be as wrong as none.
    check("non-touching hand bodies stay free on contact frames",
          float(ch[3, 21]) < 0, f"(body 21 = {float(ch[3, 21])})")


def test_levels():
    print("\n2. the clip's own tri-state convention is preserved:")
    # compute_cg_reward's ecg_all keys on `< -contact_thres`, so a naive 0/1
    # rewrite would silently change what rcg_all measures.
    t = make_clip(contact_v=1.0, free_v=-1.0)
    cv, fv, vals = rc.observed_levels(t)
    check("reads contact=+1 / free=-1 from the data", (cv, fv) == (1.0, -1.0),
          f"(got {cv}, {fv}; values {vals})")
    out, _ = rc.relabel(t, R, 0.02, 1, False)
    written = sorted({round(float(v), 4)
                      for v in torch.unique(out[:, rc.I_CONTACT_HUMAN][:, rc.HAND_BODY_IDS])})
    check("writes only values the source already used", set(written) <= {1.0, -1.0},
          f"(wrote {written})")

    # A clip using a different convention must be followed, not overridden.
    t2 = make_clip(contact_v=1.0, free_v=0.0)
    cv2, fv2, _ = rc.observed_levels(t2)
    check("follows a 1/0 clip instead of forcing -1", (cv2, fv2) == (1.0, 0.0),
          f"(got {cv2}, {fv2})")


def test_scope():
    print("\n3. nothing outside the 32 hand channels is touched:")
    t = make_clip()
    out, _ = rc.relabel(t, R, 0.02, 1, False)
    check("body positions byte-identical",
          torch.equal(out[:, rc.I_BODY], t[:, rc.I_BODY]))
    check("object positions byte-identical",
          torch.equal(out[:, rc.I_OBJP], t[:, rc.I_OBJP]))
    others = [i for i in range(52) if i not in rc.HAND_BODY_IDS]
    check("non-hand contact_human channels untouched",
          torch.equal(out[:, rc.I_CONTACT_HUMAN][:, others],
                      t[:, rc.I_CONTACT_HUMAN][:, others]))
    # Everything outside the two contact regions must survive verbatim.
    check("channels 0..329 untouched", torch.equal(out[:, :330], t[:, :330]))
    check("channels 383.. untouched", torch.equal(out[:, 383:], t[:, 383:]))


def test_channels_agree():
    print("\n4. the two contact channels agree afterwards:")
    t = make_clip()
    out, touching = rc.relabel(t, R, 0.02, 1, keep_contact_obj=False)
    obj_flag = (out[:, rc.I_CONTACT_OBJ] > 0.5)
    hand_any = (out[:, rc.I_CONTACT_HUMAN][:, rc.HAND_BODY_IDS] > 0.1).any(dim=1)
    check("contact_obj == any(hand body in contact)",
          torch.equal(obj_flag, hand_any))
    # The source disagreed (stale claim at frame 6); the fix must remove that.
    src_obj = (t[:, rc.I_CONTACT_OBJ] > 0.5)
    src_hand = (t[:, rc.I_CONTACT_HUMAN][:, rc.HAND_BODY_IDS] > 0.1).any(dim=1)
    check("the source DID disagree (fixture is exercising the real defect)",
          not torch.equal(src_obj, src_hand))

    # --keep-contact-obj must leave channel 330 exactly as found.
    out2, _ = rc.relabel(t, R, 0.02, 1, keep_contact_obj=True)
    check("--keep-contact-obj leaves channel 330 alone",
          torch.equal(out2[:, rc.I_CONTACT_OBJ], t[:, rc.I_CONTACT_OBJ]))


def test_smoothing():
    print("\n5. majority smoothing kills flicker without inventing contact:")
    x = torch.zeros(7, 2, dtype=torch.bool)
    x[3, 0] = True                                  # a lone 1-frame blip
    check("a single-frame blip is removed",
          not bool(rc.majority_smooth(x, 3)[3, 0]))
    y = torch.zeros(7, 2, dtype=torch.bool)
    y[2:5, 0] = True                                # a real 3-frame span
    check("a genuine 3-frame span survives",
          rc.majority_smooth(y, 3)[2:5, 0].all().item())
    check("smooth=1 is a no-op", torch.equal(rc.majority_smooth(x, 1), x))


def test_guard():
    """Jess's criterion, enforced: relabelling presumes contact exists."""
    print("\n6. a clip the recon never makes contact in is REFUSED:")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        # touch_frames empty -> no hand body ever reaches the ball
        torch.save(make_clip(touch_frames=(), stale_claim=(2, 3)),
                   os.path.join(src, "sub100_bball_000.pt"))
        mjcf = os.path.join(
            REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub100.xml")
        if not os.path.exists(mjcf):
            print("  SKIP: no per-subject MJCF present locally")
            return
        dst = os.path.join(tmp, "dst")
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--src-dir", src, "--dst-dir", dst,
             "--mjcf", mjcf, "--threshold", "0.02"],
            capture_output=True, text=True)
        check("exits non-zero", proc.returncode != 0, f"(exit {proc.returncode})")
        check("says it is a reconstruction problem, not a labelling one",
              "RECONSTRUCTION problem" in (proc.stdout + proc.stderr))
        check("writes NOTHING (no half-built dataset left behind)",
              not os.path.exists(dst))

    print("\n7. a healthy clip round-trips through the CLI:")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        torch.save(make_clip(), os.path.join(src, "sub100_bball_000.pt"))
        open(os.path.join(src, "notes.txt"), "w").write("carried along")
        mjcf = os.path.join(
            REPO, "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub100.xml")
        if not os.path.exists(mjcf):
            print("  SKIP: no per-subject MJCF present locally")
            return
        dst = os.path.join(tmp, "dst")
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--src-dir", src, "--dst-dir", dst,
             "--mjcf", mjcf, "--threshold", "0.02"],
            capture_output=True, text=True)
        check("exits 0", proc.returncode == 0, f"({proc.stderr[-400:]})")
        check("non-.pt files carried along",
              os.path.exists(os.path.join(dst, "notes.txt")))
        out = torch.load(os.path.join(dst, "sub100_bball_000.pt"),
                         map_location="cpu", weights_only=False)
        src_t = torch.load(os.path.join(src, "sub100_bball_000.pt"),
                           map_location="cpu", weights_only=False)
        check("written clip has the stale claim cleared",
              float(out[:, rc.I_CONTACT_HUMAN][6, 20]) < 0)
        check("written clip kept positions", torch.equal(out[:, rc.I_BODY],
                                                         src_t[:, rc.I_BODY]))

        # Re-running into an existing dir must refuse, per the never-overwrite rule.
        proc2 = subprocess.run(
            [sys.executable, SCRIPT, "--src-dir", src, "--dst-dir", dst,
             "--mjcf", mjcf], capture_output=True, text=True)
        check("refuses to overwrite an existing dst",
              proc2.returncode != 0
              and "refusing to overwrite" in (proc2.stdout + proc2.stderr))

        # --census must write nothing at all.
        dst2 = os.path.join(tmp, "dst2")
        proc3 = subprocess.run(
            [sys.executable, SCRIPT, "--src-dir", src, "--mjcf", mjcf, "--census"],
            capture_output=True, text=True)
        check("--census exits 0 and writes nothing",
              proc3.returncode == 0 and not os.path.exists(dst2))
        check("--census reports the convention it found",
              "contact_human hand values present" in proc3.stdout)


def main():
    test_geometry()
    test_levels()
    test_scope()
    test_channels_agree()
    test_smoothing()
    test_guard()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Diagnose the CARI4D basketball clip's reference quality in one pass:

  1. contact timeline   -- the recon's contact_obj flag as spans (does contact
                           ever RESUME after the dribble bounce?)
  2. hand-ball distance -- min wrist-to-ball distance per frame (is the recon
                           ball ever back NEAR the hand, even if the flag says
                           no? flags-broken vs trajectory-broken)
  3. floor offset       -- lowest body point per frame (grounded frames at
                           z ~= 0.23 = the known monocular floor offset is live)
  4. ball height        -- ball z per frame (bounce profile vs the sim floor)
  6. floor penetration  -- WHICH body goes below z=0 and how deep. A single limb
                           dipping is a local IK artifact; the pelvis dropping
                           with it means the whole body sank, and the two want
                           different corrections. Frames below the floor are
                           unreachable in sim (PhysX pushes the humanoid out),
                           so they are a reward desert wherever they land inside
                           the training start range.
  7. crouch / takeoff   -- pelvis and knee height around each flight span. The
                           feet (= "lowest body") sit at z~0 whether the subject
                           loads the knees or stays rigid, so lowest-z CANNOT
                           answer "does the reference actually crouch before the
                           jump?". Pelvis dip in the frames before takeoff can.
                           If the recon flattened the crouch, no amount of
                           start-frame coverage teaches a knee-load -- there is
                           nothing to imitate.

Everything prints as aligned per-frame/per-span tables; no plots, no GPU, no
Isaac Gym -- runs anywhere the clip file and the subject MJCF exist (cluster).

  python3 scripts/inspect_bball_clip.py
  python3 scripts/inspect_bball_clip.py --clip InterAct/behave_cari4d/sub100_bball_000.pt --every 2
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree  # noqa: E402  (the validated MJCF parser)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Clip layout (see retarget_contact.py header; verified against OMOMO_new):
I_ROOT = slice(0, 3)          # root position -- what the SIM actually drives FK from
I_BODY = slice(162, 318)      # 52 bodies x 3, world frame (stored kinematics)
I_OBJP = slice(318, 321)      # object position
I_CONTACT_OBJ = 330           # 1.0 where the recon says object is in contact


def contact_spans(flags):
    """[(start, end, bool)] runs of the contact flag."""
    out, i = [], 0
    for k, g in itertools.groupby(flags.tolist()):
        n = len(list(g))
        out.append((i, i + n - 1, bool(k)))
        i += n
    return out


def wrist_indices(mjcf_path):
    """L_Wrist / R_Wrist indices in the clip's 52-body order, from the MJCF
    itself -- no hardcoded indices (they'd silently rot if the order changed)."""
    names = [n for n, _, _ in _parse_mjcf_tree(mjcf_path)]
    try:
        return [names.index("L_Wrist"), names.index("R_Wrist")], names
    except ValueError:
        raise SystemExit(f"FATAL: L_Wrist/R_Wrist not in {mjcf_path} body names; "
                         f"got {names[:8]}...")


def named_indices(names, wanted):
    """Indices of `wanted` body names, skipping any this MJCF doesn't have.
    Returns (indices, missing) -- callers decide whether a miss is fatal, so a
    body-set difference degrades a column rather than killing the whole report."""
    idx = [names.index(w) for w in wanted if w in names]
    return idx, [w for w in wanted if w not in names]


def runs_where(mask):
    """[(start, end)] for each contiguous True run in a boolean sequence."""
    out = []
    for k, g in itertools.groupby(range(len(mask)), key=lambda i: bool(mask[i])):
        idx = list(g)
        if k:
            out.append((idx[0], idx[-1]))
    return out


def penetration_spans(lowest, floor=0.0):
    """Runs where the lowest body point is BELOW the floor, with the depth and
    the frame at which it is deepest.

    Sim cannot reproduce these poses -- PhysX pushes the humanoid out -- so the
    tracking terms penalize the policy for something physically unreachable.
    Returns [(start, end, depth, deepest_frame)] with depth > 0 = metres below.
    """
    spans = []
    for s, e in runs_where([float(z) < floor for z in lowest]):
        seg = lowest[s:e + 1]
        j = int(seg.argmin())
        spans.append((s, e, float(floor - seg[j]), s + j))
    return spans


def flight_spans(lowest, thr=0.10, min_len=3):
    """Runs where the lowest body point is clearly airborne.

    `thr` is well above the recon's grounded noise (median ~0.05) and well below
    a real jump apex (~0.54), so it separates flight from foot-lift jitter.
    `min_len` drops 1-2 frame blips that are IK noise, not a jump.
    """
    return [(s, e) for s, e in runs_where([float(z) > thr for z in lowest])
            if e - s + 1 >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(
        REPO, "InterAct/behave_cari4d/sub100_bball_000.pt"))
    ap.add_argument("--mjcf", default=os.path.join(
        REPO, "isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub100.xml"),
        help="the clip subject's MJCF (for wrist indices by NAME)")
    ap.add_argument("--every", type=int, default=4,
                    help="print every Nth frame in the per-frame tables")
    args = ap.parse_args()

    if not os.path.exists(args.clip):
        raise SystemExit(f"FATAL: clip not found: {args.clip} (cluster-only data?)")
    c = torch.load(args.clip, map_location="cpu")
    c = c.detach() if hasattr(c, "detach") else c
    T = c.shape[0]
    bp = c[:, I_BODY].view(T, 52, 3)
    obj = c[:, I_OBJP]
    flags = c[:, I_CONTACT_OBJ] > 0.5
    wrists, names = wrist_indices(args.mjcf)
    hand_d = (bp[:, wrists, :] - obj[:, None, :]).norm(dim=-1).min(dim=1).values
    lowest = bp[:, :, 2].min(dim=1).values
    lowest_body = bp[:, :, 2].argmin(dim=1)          # WHICH body is on the floor
    pelvis_z = bp[:, 0, 2]                           # body 0 = pelvis (root)
    knees, missing_knees = named_indices(names, ["L_Knee", "R_Knee"])
    knee_z = bp[:, knees, 2].min(dim=1).values if knees else None

    print(f"clip: {args.clip}  ({T} frames)")

    print(f"\n== 1. contact_obj flag timeline ==")
    for s, e, k in contact_spans(flags):
        print(f"  frames {s:3d}-{e:3d} ({e-s+1:3d}f): {'CONTACT' if k else 'free'}")
    n_resume = sum(1 for i, (s, e, k) in enumerate(contact_spans(flags)) if k and i > 0)
    print(f"  -> contact spans after the first: {n_resume} "
          f"({'contact RESUMES' if n_resume else 'contact NEVER resumes -- no catch in the supervision'})")

    print(f"\n== 2/3/4. per-frame: hand-ball dist | lowest body z | ball z | flag ==")
    print(f"  {'frame':>5s} {'hand-ball(m)':>12s} {'lowest z(m)':>11s} "
          f"{'lowest body':>12s} {'pelvis z(m)':>11s} {'ball z(m)':>9s}  flag")
    for i in range(0, T, args.every):
        print(f"  {i:5d} {hand_d[i]:12.3f} {lowest[i]:11.3f} "
              f"{names[int(lowest_body[i])]:>12s} {pelvis_z[i]:11.3f} "
              f"{obj[i,2]:9.3f}  {'CONTACT' if flags[i] else 'free'}")

    # 5. FK-vs-stored consistency. The sim drives the humanoid by FK from
    # root_pos+root_rot+dof_pos; hand-ball above is measured from the STORED
    # body_pos channels. If a conversion transform (rotate / drop-to-floor)
    # touched one set and not the other, body_pos can say "ball in hand" while
    # the FK-driven reference stands somewhere else. root_pos vs body_pos[0]
    # (the pelvis) catches the translation/rotation forms of that mismatch.
    root_delta = (c[:, I_ROOT] - bp[:, 0, :]).norm(dim=-1)
    print(f"\n== 5. root_pos vs stored pelvis (FK/body_pos consistency) ==")
    print(f"  |root_pos - body_pos[0]|: mean {root_delta.mean():.3f} m  "
          f"max {root_delta.max():.3f} m at frame {int(root_delta.argmax())}")
    print(f"  read: ~0.0x m (fixed pelvis offset) = consistent; growing or")
    print(f"        decimeter+ deltas = conversion transformed the channel sets")
    print(f"        differently -- the sim's reference does NOT match these tables.")

    # 6. Floor penetration. Which body, how deep, and did the pelvis go with it?
    # A lone limb below the floor is a local IK artifact -- correcting it means
    # touching that limb. The pelvis dropping too means the whole body sank, and
    # the correction is a vertical offset on the root. These are different fixes,
    # so the report must distinguish them rather than just flag "below floor".
    print(f"\n== 6. floor penetration (frames the sim physically cannot reproduce) ==")
    pens = penetration_spans(lowest)
    if not pens:
        print("  none -- no body point goes below z=0. Nothing to correct here.")
    else:
        pelvis_ref = float(pelvis_z.median())
        for s, e, depth, jf in pens:
            culprits = sorted({names[int(lowest_body[i])] for i in range(s, e + 1)})
            pelvis_drop = pelvis_ref - float(pelvis_z[s:e + 1].min())
            print(f"  frames {s:3d}-{e:3d} ({e-s+1:3d}f): max depth {depth:.3f} m "
                  f"at frame {jf}, body '{names[int(lowest_body[jf])]}'")
            print(f"      bodies on the floor during the span: {', '.join(culprits)}")
            print(f"      pelvis dips {pelvis_drop:+.3f} m below its own median "
                  f"({pelvis_ref:.3f} m) -> "
                  f"{'WHOLE BODY sank (root-offset fix)' if pelvis_drop > depth * 0.5 else 'LIMB-local artifact (pelvis stayed put)'}")
        print(f"  read: these frames are a reward desert wherever they fall inside")
        print(f"        the training start range -- the policy is graded against a")
        print(f"        pose it cannot reach. Compare the span to rolloutLength's")
        print(f"        start range (randint(0, clip_len - rolloutLength)).")

    # 7. Crouch. The whole "it won't bend its knees" question lives here: the
    # feet are on the floor either way, so lowest-z is blind to it. If the recon
    # flattened the crouch there is nothing to imitate and no coverage knob helps.
    print(f"\n== 7. crouch before takeoff (is a knee-load present to imitate?) ==")
    if missing_knees:
        print(f"  WARNING: MJCF lacks {missing_knees} -- knee column unavailable")
    flights = flight_spans(lowest)
    if not flights:
        print("  no flight span found (lowest body never exceeds 0.10 m) -- this")
        print("  clip contains no jump, so there is no takeoff to prepare for.")
    for s, e in flights:
        w0 = max(0, s - 15)                      # the run-up: 0.5 s at 30 fps
        stand = float(pelvis_z[w0:s].max()) if s > w0 else float(pelvis_z[s])
        dip = float(pelvis_z[w0:s].min()) if s > w0 else float(pelvis_z[s])
        print(f"  flight frames {s:3d}-{e:3d} (apex lowest-z "
              f"{float(lowest[s:e+1].max()):.3f} m at frame {s + int(lowest[s:e+1].argmax())})")
        print(f"      pelvis over the {s-w0} run-up frames {w0}-{s-1}: "
              f"max {stand:.3f} -> min {dip:.3f} m  (dip {stand - dip:.3f} m)")
        if knee_z is not None:
            print(f"      knee    over the same frames: "
                  f"max {float(knee_z[w0:s].max()):.3f} -> "
                  f"min {float(knee_z[w0:s].min()):.3f} m")
        # A real countermovement jump drops the pelvis ~0.10-0.20 m. Under ~0.05
        # there is effectively no crouch in the reference to learn from.
        verdict = ("CROUCH PRESENT -- a knee-load exists to imitate"
                   if stand - dip >= 0.10 else
                   "SHALLOW/ABSENT -- the reference barely dips; the policy has "
                   "nothing to copy, so coverage knobs will not teach a jump"
                   if stand - dip < 0.05 else
                   "MARGINAL -- some dip, but shallower than a real countermovement")
        print(f"      read: {verdict}")

    grounded = lowest[lowest < lowest.median() + 0.05]
    print(f"\n== summary ==")
    print(f"  grounded-frame lowest-body z: median {grounded.median():.3f} m "
          f"(~0.00 = floor ok, ~0.23 = the known monocular floor offset is LIVE)")
    # "post-bounce" was hardcoded as frame 50, which crashes on any clip shorter
    # than 51 frames (the trimmed t4/t10/t32 exports) and is simply the wrong
    # frame for any clip whose bounce lands elsewhere. Derive it from the clip's
    # own free-flight stretch instead: the ball leaves the hand, and the question
    # is whether it ever comes back.
    free_spans = [(s, e) for s, e, k in contact_spans(flags) if not k and s > 0]
    post = free_spans[0][1] + 1 if free_spans else T // 2
    if post < T:
        tail = hand_d[post:]
        print(f"  hand-ball distance: min {hand_d.min():.3f} m at frame {int(hand_d.argmin())}; "
              f"post-frame-{post} min {tail.min():.3f} m at frame {post + int(tail.argmin())}")
    else:
        print(f"  hand-ball distance: min {hand_d.min():.3f} m at frame {int(hand_d.argmin())}; "
              f"(no frames after the first free-flight span -- no post-bounce window)")
    print(f"  ball z: min {obj[:,2].min():.3f}  max {obj[:,2].max():.3f} m")
    print(f"  read: flags say no resume + hand-ball gets small again -> FLAGS broken (relabel);")
    print(f"        hand-ball never gets small post-bounce -> ball TRAJECTORY broken (rebuild).")


if __name__ == "__main__":
    main()

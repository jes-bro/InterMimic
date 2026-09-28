#!/usr/bin/env python3
"""Numeric verdict for the decoupled kinematic replay (slurm_replay_xbody.sh):
is a BODY's MJCF driven correctly by a good SOURCE motion?

Run the same pinned clip through two bodies with DUMP_TRAJ set, then compare:

  CLIP=sub2_largetable_005 BODY=sub2 SOURCE=sub2 DUMP_TRAJ=renders/rxb_sub2_005.npz sbatch slurm_replay_xbody.sh
  CLIP=sub2_largetable_005 BODY=sub4 SOURCE=sub2 DUMP_TRAJ=renders/rxb_sub4_005.npz sbatch slurm_replay_xbody.sh
  python3 scripts/compare_replay_dumps.py renders/rxb_sub4_005.npz renders/rxb_sub2_005.npz

WHY THIS IS SHARP. play_dataset poses the humanoid by FK from the SOURCE's
root + dof stream through the BODY's MJCF. The per-subject MJCFs are
structurally identical (audit_mjcf.py --xmldiff: same tree, joints, axes,
ranges) and differ only in bone lengths / capsule sizes. Therefore, frame by
frame, for the identical dof stream:
  (1) every rigid body's GLOBAL ROTATION must be identical between the two
      bodies (rotations do not depend on bone lengths), and
  (2) every child-minus-parent position vector, expressed in the PARENT's
      frame, must be CONSTANT over time within each body (it is that body's
      fixed rest offset, i.e. the bone). Across the two bodies those rest
      offsets differ in length (and by a degree or so in direction, since a
      real subject's joints are not a pure scaling of another's), which is
      reported as information, not failure.
A body that violates (1) or (2), or that produces NaN/inf, is genuinely broken
in the simulator; a body that passes is fine and its problems lie elsewhere
(policy, conditioning, contact). Foot heights are printed as information
(a shorter body on the taller body's root trajectory floats; that is expected
and is what retargeting fixes, not a defect).

Dump format (intermimic_players.py DUMP_TRAJ): body_pos (T,52,3), body_rot
(T,52,4) as (x,y,z,w), obj_pos, obj_rot, subject. Body order = MJCF document
order, read from --mjcf so the report names bodies.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_MJCF = Path(__file__).resolve().parents[1] / \
    "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_sub2.xml"
FEET = ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")


def mjcf_tree(path):
    """(names in document order, parent index per body; -1 for the root)."""
    root = ET.parse(path).getroot()
    names, parents = [], []

    def walk(el, parent_idx):
        for child in el:
            if child.tag == "body":
                idx = len(names)
                names.append(child.get("name"))
                parents.append(parent_idx)
                walk(child, idx)
            else:
                walk(child, parent_idx)          # e.g. <worldbody> wrapper

    walk(root, -1)
    return names, np.array(parents)


def quat_angle_deg(qa, qb):
    """Angle between two unit quaternions (any sign), (..., 4) -> (...) degrees."""
    d = np.clip(np.abs((qa * qb).sum(-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def vec_angle_deg(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def local_bones(pos, rot, child, parents):
    """Child-minus-parent vectors rotated into the parent's frame: (T, len(child), 3).
    rot is (T, B, 4) as (x,y,z,w); the parent's inverse rotation is applied."""
    from scipy.spatial.transform import Rotation as R
    T = pos.shape[0]
    v = pos[:, child] - pos[:, parents[child]]
    out = np.empty_like(v)
    for t in range(T):
        out[t] = R.from_quat(rot[t, parents[child]]).inv().apply(v[t])
    return out


def load(path):
    d = np.load(path, allow_pickle=True)
    return (d["body_pos"].astype(np.float64), d["body_rot"].astype(np.float64),
            str(d["subject"]) if "subject" in d.files else Path(path).stem)


def compare(sus_pos, sus_rot, ctl_pos, ctl_rot, names, parents,
            rot_tol_deg=1.0, dir_tol_deg=1.0, len_cv_tol=0.02):
    """Returns (verdict_ok, report_lines). Pure function, tested directly."""
    lines, ok = [], True
    T = min(len(sus_pos), len(ctl_pos))
    if len(sus_pos) != len(ctl_pos):
        lines.append(f"  note: frame counts differ ({len(sus_pos)} vs {len(ctl_pos)}); comparing the first {T}")
    sus_pos, sus_rot, ctl_pos, ctl_rot = sus_pos[:T], sus_rot[:T], ctl_pos[:T], ctl_rot[:T]

    # (0) finite?
    for tag, arr in (("suspect pos", sus_pos), ("suspect rot", sus_rot),
                     ("control pos", ctl_pos), ("control rot", ctl_rot)):
        bad = ~np.isfinite(arr)
        if bad.any():
            ok = False
            frames = np.unique(np.nonzero(bad)[0])
            bodies = sorted({names[i] for i in np.unique(np.nonzero(bad)[1])})
            lines.append(f"  FAIL {tag}: NaN/inf in {len(frames)} frame(s) (first {frames[0]}), bodies {bodies[:8]}")
    if not ok:
        return ok, lines

    # (1) global rotations identical
    ang = quat_angle_deg(sus_rot, ctl_rot)                 # (T, B)
    worst = ang.max(0)
    bad_rot = [(names[i], worst[i]) for i in np.argsort(-worst) if worst[i] > rot_tol_deg]
    lines.append(f"  global rotation: max deviation {worst.max():.3f} deg over all bodies/frames "
                 f"(tol {rot_tol_deg})")
    if bad_rot:
        ok = False
        lines.append("  FAIL rotation mismatch: " + ", ".join(f"{n} {a:.2f}deg" for n, a in bad_rot[:10]))

    # (2) each bone in its PARENT's frame must be a constant vector over time
    #     (the rest offset). A bone that wanders = FK through a broken body.
    child = np.nonzero(parents >= 0)[0]
    sv = local_bones(sus_pos, sus_rot, child, parents)      # (T, B-1, 3)
    cv = local_bones(ctl_pos, ctl_rot, child, parents)
    for tag, v in (("suspect", sv), ("control", cv)):
        mean = v.mean(0)                                     # (B-1, 3) rest offset estimate
        L = np.linalg.norm(mean, axis=-1)
        wander = np.linalg.norm(v - mean, axis=-1).max(0)    # metres, per bone
        wander_deg = vec_angle_deg(v, np.broadcast_to(mean, v.shape)).max(0)
        wander_deg[L < 1e-6] = 0.0                           # zero-length stubs have no direction
        bad = [(names[child[j]], wander_deg[j], wander[j]) for j in np.argsort(-wander_deg)
               if wander_deg[j] > dir_tol_deg or wander[j] > len_cv_tol * max(L[j], 0.05)]
        lines.append(f"  {tag} bones in parent frame: max wander {wander_deg.max():.3f} deg / "
                     f"{wander.max()*1000:.2f} mm over time (tol {dir_tol_deg} deg / "
                     f"{len_cv_tol*100:.0f}% of length)")
        if bad:
            ok = False
            lines.append(f"  FAIL {tag} bone not rigid: " +
                         ", ".join(f"{n} {a:.2f}deg/{d*1000:.1f}mm" for n, a, d in bad[:10]))

    # how the two skeletons differ (information): rest-offset length ratio and
    # direction difference per bone, segments > 5 cm only (skips finger stubs)
    ms, mc = sv.mean(0), cv.mean(0)
    Ls, Lc = np.linalg.norm(ms, axis=-1), np.linalg.norm(mc, axis=-1)
    big = np.nonzero(Lc > 0.05)[0]
    ratio = Ls[big] / Lc[big]
    dang = vec_angle_deg(ms[big], mc[big])
    j = int(np.argmax(dang))
    lines.append(f"  skeleton difference (segments > 5 cm): length ratio suspect/control "
                 f"min {ratio.min():.3f} max {ratio.max():.3f}; rest-offset direction differs "
                 f"by at most {dang.max():.2f} deg ({names[child[big[j]]]})")

    # root and feet (information)
    root_d = np.linalg.norm(sus_pos[:, 0] - ctl_pos[:, 0], axis=-1).max()
    lines.append(f"  root position: max difference {root_d*100:.2f} cm (same reference root => ~0)")
    for tag, P in (("suspect", sus_pos), ("control", ctl_pos)):
        idx = [names.index(f) for f in FEET if f in names]
        lines.append(f"  {tag} feet: min z {P[:, idx, 2].min()*100:+.1f} cm, "
                     f"median z {np.median(P[:, idx, 2])*100:+.1f} cm "
                     f"(negative = below ground, large positive = floating)")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suspect", help="DUMP_TRAJ npz of the body under test")
    ap.add_argument("control", help="DUMP_TRAJ npz of a known-good body, SAME pinned clip")
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF,
                    help="any per-subject MJCF: supplies the 52 body names + parents")
    ap.add_argument("--rot-tol-deg", type=float, default=1.0)
    ap.add_argument("--dir-tol-deg", type=float, default=1.0)
    ap.add_argument("--len-cv-tol", type=float, default=0.02)
    a = ap.parse_args()

    names, parents = mjcf_tree(a.mjcf)
    sp, sr, ssub = load(a.suspect)
    cp, cr, csub = load(a.control)
    if sp.shape[1] != len(names) or cp.shape[1] != len(names):
        sys.exit(f"ERROR: dumps have {sp.shape[1]} / {cp.shape[1]} bodies but the MJCF "
                 f"has {len(names)}")
    print(f"[replay-compare] suspect {ssub} ({len(sp)} frames) vs control {csub} ({len(cp)} frames)")
    ok, lines = compare(sp, sr, cp, cr, names, parents,
                        a.rot_tol_deg, a.dir_tol_deg, a.len_cv_tol)
    print("\n".join(lines))
    if ok:
        print(f"VERDICT: PASS -- {ssub}'s body is driven exactly like {csub}'s; the MJCF is not broken")
    else:
        print(f"VERDICT: FAIL -- {ssub}'s body deviates from {csub}'s on the same motion (see FAIL lines)")
        sys.exit(1)


if __name__ == "__main__":
    main()

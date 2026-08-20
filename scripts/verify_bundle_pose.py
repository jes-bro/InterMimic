#!/usr/bin/env python3
"""Does the installed clip carry the bundle's ARTICULATED pose? The missing
half of verify_bundle_vs_clip.py (which checks only the ball + root
translation): compare per-joint local rotations, which no world transform can
change -- a correct conversion preserves them exactly.

Bundle side: pr['smpl_pose'] (T,72) = SMPL axis-angle, root + 23 joints.
Clip side:   dof_pos (channels 9:162) = 51 joints, axis-angle, MuJoCo order,
             mapped back to SMPL order with the same tables interact2mimic
             used (borrowed verbatim from retarget_omomo_cross_body.py).

Per body joint (SMPL 1..21, hips..wrists) we report the geodesic angle between
the bundle's rotation and the clip's, per frame. ~0 deg everywhere = converter
preserved the pose and the motion difference must be upstream/rendering;
tens of degrees on shoulders/elbows = the converter mangled the limbs (e.g. a
behind-the-back move disappearing), and the frame window localizes it.

  PYTHONPATH=/simurgh2/projects/ret-hoi/CARI4D python3 scripts/verify_bundle_pose.py \
      --bundle .../Date03_Sub01_bball_dribble.pth \
      --clip InterAct/behave_cari4d_optj3d/sub100_bball_000.pt
"""
import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

# interact2mimic.py:741-747 joint order tables (via retarget_omomo_cross_body.py)
SMPL_2_MUJOCO_NEW = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    14, 17, 19, 21,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
]
# sized by max target index: the table maps into a 55-slot space (SMPL-X jaw/eye
# slots 22-24 are skipped), so len() under-allocates
MUJOCO_2_SMPL_NEW = [0] * (max(SMPL_2_MUJOCO_NEW) + 1)
for _s, _m in enumerate(SMPL_2_MUJOCO_NEW):
    MUJOCO_2_SMPL_NEW[_m] = _s

SMPL_BODY_NAMES = {
    1: "L_Hip", 2: "R_Hip", 3: "Spine1", 4: "L_Knee", 5: "R_Knee", 6: "Spine2",
    7: "L_Ankle", 8: "R_Ankle", 9: "Spine3", 10: "L_Foot", 11: "R_Foot",
    12: "Neck", 13: "L_Collar", 14: "R_Collar", 15: "Head", 16: "L_Shoulder",
    17: "R_Shoulder", 18: "L_Elbow", 19: "R_Elbow", 20: "L_Wrist", 21: "R_Wrist",
}


def clip_dof_in_smpl_order(clip):
    """(T,51,3) local axis-angle, SMPL non-root order, from the 591-ch tensor."""
    T = clip.shape[0]
    dof_mujoco = clip[:, 9:9 + 153].double().numpy().reshape(T, 51, 3)
    dof_smpl = np.zeros_like(dof_mujoco)
    for mj in range(1, 52):
        dof_smpl[:, MUJOCO_2_SMPL_NEW[mj] - 1, :] = dof_mujoco[:, mj - 1, :]
    return dof_smpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="CARI4D export .pth to compare against")
    ap.add_argument("--ref-clip", help="OR: another installed .pt to compare against "
                    "(stale-intermediate hunt: ~0 deg vs an OLD clip = the conversion "
                    "reused that clip's human)")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()
    if bool(args.bundle) == bool(args.ref_clip):
        raise SystemExit("pass exactly one of --bundle / --ref-clip")

    clip = torch.load(args.clip, map_location="cpu")
    dof_smpl = clip_dof_in_smpl_order(clip)
    if args.bundle:
        b = torch.load(args.bundle, map_location="cpu", weights_only=False)
        pr = b["pr"] if "pr" in b else b
        smpl_pose = np.asarray(pr["smpl_pose"]).astype(np.float64)   # (T,72)
    else:
        ref = torch.load(args.ref_clip, map_location="cpu")
        ref_dof = clip_dof_in_smpl_order(ref)
        # synthesize a (T,72)-style array from the ref clip's SMPL-order dof
        smpl_pose = np.zeros((len(ref_dof), 72))
        for j in range(1, 22):
            smpl_pose[:, 3 * j:3 * j + 3] = ref_dof[:, j - 1, :]
        ref_ball = ref[:, 318:321].double().numpy()

    T = min(len(smpl_pose), len(dof_smpl))
    print(f"frames: bundle {len(smpl_pose)} | clip {len(dof_smpl)} (comparing {T})")

    # geodesic error per joint per frame, degrees
    errs = {}
    for j, name in SMPL_BODY_NAMES.items():
        Rb = sRot.from_rotvec(smpl_pose[:T, 3 * j:3 * j + 3])
        Rc = sRot.from_rotvec(dof_smpl[:T, j - 1, :])
        errs[name] = np.degrees((Rb.inv() * Rc).magnitude())

    E = np.stack(list(errs.values()))                              # (J,T)
    print(f"\n== per-joint geodesic error (deg) ==")
    print(f"  {'joint':12s} {'mean':>7s} {'max':>7s}  @frame")
    for name, e in sorted(errs.items(), key=lambda kv: -kv[1].mean()):
        print(f"  {name:12s} {e.mean():7.2f} {e.max():7.2f}  {int(e.argmax()):5d}")

    print(f"\n== per-frame mean error over all body joints (deg) ==")
    m = E.mean(0)
    for i in range(0, T, args.every):
        bar = "#" * int(min(m[i], 60) / 2)
        print(f"  {i:5d} {m[i]:7.2f}  {bar}")

    v = E.mean()
    print(f"\noverall mean {v:.2f} deg | overall max {E.max():.2f} deg")
    print("read: <1 deg = pose preserved, converter clean (look upstream/at the render);")
    print("      big errors on arms/shoulders or in a frame window = converter mangled the pose THERE.")

    # Ball trajectory: rigid-invariant speed profile (same check as
    # verify_bundle_vs_clip.py, repeated here so one run = full verdict).
    if args.bundle:
        ball_b = np.asarray(pr["pose_abs"])[:T, :3, 3].astype(np.float64)
    else:
        ball_b = ref_ball[:T]
    ball_c = clip[:T, 318:321].double().numpy()
    sp_b = np.linalg.norm(np.diff(ball_b, axis=0), axis=1)
    sp_c = np.linalg.norm(np.diff(ball_c, axis=0), axis=1)
    d = np.abs(sp_b - sp_c)
    print(f"\n== ball trajectory (rigid-invariant speed profile) ==")
    print(f"  max |speed diff| {d.max():.4f} m/frame at frame {int(d.argmax())} | "
          f"path length ref {sp_b.sum():.2f} m vs clip {sp_c.sum():.2f} m")
    print(f"  {'frame':>5s} {'ref v':>7s} {'clip v':>7s} {'|diff|':>7s}   (m/frame; ~0 diff = same ball motion THERE)")
    for i in range(0, len(d), args.every):
        print(f"  {i:5d} {sp_b[i]:7.3f} {sp_c[i]:7.3f} {d[i]:7.3f}")
    print("  read: <0.01 everywhere = same ball; a window of ~0 diffs vs an OLD clip = "
          "that stretch of the ball is the old trajectory")


if __name__ == "__main__":
    main()

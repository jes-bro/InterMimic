#!/usr/bin/env python3
"""Cross-body kinematic retargeter for OMOMO motion clips.

Takes a source subject's processed .pt file and produces a new .pt file as
if the same motion had been captured on the target subject's body shape.

What gets retargeted:
  - body_pos  (slice 162:318): recomputed via SMPL-X FK with target betas
  - body_rot  (slice 383:591): recomputed via target's skeleton tree FK
  - root_pos  (slice 0:3):     z-shifted to keep target body off the ground
  - root_rot  (slice 3:7):     re-derived from target's global pose
  - obj_pos   (slice 318:321): adjusted by per-frame hand-object offset so
                               the grasp is preserved (option-2 trick we
                               discussed: cheap approx to full IK retarget)

What is preserved unchanged from source:
  - dof_pos   (slice 9:162):   joint angles are body-agnostic
  - obj_rot   (slice 321:325): object rotation doesn't change with the body
  - contact_obj / contact_human channels
  - Object trajectory shape (only translated by hand offset, not rotated)

Why this matters: lets the policy train with body_pos tracking reward intact
(since the reference body_pos now lives in the controlled body's coordinate
frame). See [[feedback_cari4d_rotation_lesson]] and the cross-body project
plan in tasks #3 of the current session.

CLUSTER-ONLY: needs SMPL-X model files + the InterAct vendored phc/poselib.
Run from /simurgh2/projects/ret-hoi/InterMimic.

Usage:
    conda activate intermimic-gym
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cd /simurgh2/projects/ret-hoi/InterMimic
    python -u scripts/retarget_omomo_cross_body.py \\
        --source-sub sub2 --target-sub sub8 \\
        --source-dir /simurgh2/projects/ret-hoi/InterMimic/InterAct/OMOMO \\
        --output-dir /simurgh2/projects/ret-hoi/InterMimic/InterAct/OMOMO_cross
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# --- Cluster paths (hardcoded per reference_cluster_paths memory) ---
INTERACT_ROOT = "/simurgh2/projects/ret-hoi/InterAct"
INTERMIMIC_ROOT = "/simurgh2/projects/ret-hoi/InterMimic"

# Joint reordering tables borrowed verbatim from interact2mimic.py:741-747.
# SMPL has its own joint order (depth-first or whatever order the SMPL authors
# chose); MuJoCo wants its own order (matching the kinematic tree definition
# in the MJCF). interact2mimic.py uses `smpl_2_mujoco_new` (the 52-joint
# variant) for SMPL-X data. We need both directions.
SMPL_2_MUJOCO_NEW = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    14, 17, 19, 21,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
]
# Inverse permutation: smpl_idx = MUJOCO_2_SMPL_NEW[mujoco_idx]. Computed once
# at module load so we can reorder either direction cheaply.
MUJOCO_2_SMPL_NEW = [0] * len(SMPL_2_MUJOCO_NEW)
for smpl_idx, mujoco_idx in enumerate(SMPL_2_MUJOCO_NEW):
    MUJOCO_2_SMPL_NEW[mujoco_idx] = smpl_idx


def setup_interact_paths(interact_root: Path) -> None:
    """chdir + sys.path so smpl_local_robot's module-level imports resolve.

    Mirrors generate_per_subject_mjcfs.py — see that file for full rationale.
    Briefly: smpl_local_robot.py has `MODEL_PATH = "../models"` at module scope
    and uses it in module-level smplx.create() calls, so importing it from
    anywhere other than InterAct/simulation breaks.
    """
    sim_dir = (interact_root / "simulation").resolve()
    if not sim_dir.is_dir():
        raise FileNotFoundError(
            f"InterAct simulation dir not found: {sim_dir}. "
            f"This script is cluster-only — SMPL-X models aren't local."
        )
    os.chdir(sim_dir)
    sys.path.insert(0, str(sim_dir))


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) to quaternion (..., 4) in (x, y, z, w) order.

    Uses scipy.spatial.transform.Rotation. Returns wxyz convention? No: scipy
    returns (x, y, z, w). interact2mimic.py uses the same scipy convention
    throughout, so we match it.
    """
    from scipy.spatial.transform import Rotation as sRot
    return sRot.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(*aa.shape[:-1], 4)


def _quat_to_aa(quat: np.ndarray) -> np.ndarray:
    """Quaternion (..., 4) (x, y, z, w order) to axis-angle (..., 3)."""
    from scipy.spatial.transform import Rotation as sRot
    return sRot.from_quat(quat.reshape(-1, 4)).as_rotvec().reshape(*quat.shape[:-1], 3)


def extract_pose_aa_from_pt(data: "torch.Tensor") -> np.ndarray:
    """Reverse-engineer SMPL-X pose_aa from a processed .pt file.

    The .pt file (591 channels) stores:
      - slice 3:7      = root global rotation as quaternion (4 dims)
      - slice 9:162    = local joint rotations as axis-angle, but reordered
                         to MuJoCo joint order (153 dims = 51 non-root joints)

    To run SMPL-X FK we need pose_aa in SMPL joint order, shape (T, 52*3=156):
      pose_aa[:, 0:3]  = root orientation as axis-angle
      pose_aa[:, 3:]   = local rotations of non-root joints, SMPL order

    Note: interact2mimic.py applies an `upright_start` correction (line 795)
    that left-multiplies global rotations by inv(quat([0.5,0.5,0.5,0.5])).
    The 153-dim dof_pos at slice 9:162 was computed AFTER this correction
    via _local_rotation_to_dof_smpl, so the local axis-angle stored there is
    already "MuJoCo-friendly". For our purpose (running FK to get body
    positions in target body), the local rotations describe the kinematic
    chain shape regardless of frame convention — they're chain-local and
    only depend on relative joint angles, not the world frame.

    HOWEVER, when we feed pose_aa into smplx.create()'s FK, the root orient
    must be in SMPL's global frame, not the MuJoCo frame. We undo the
    upright_start correction on the root rotation only.
    """
    import torch
    from scipy.spatial.transform import Rotation as sRot

    T = data.shape[0]

    # Root rotation: stored as global quat (post-upright_start correction).
    # Undo the correction to get back to SMPL's native global frame:
    #   pose_quat_global = global_rotation * inv(upright_q)
    # so: global_rotation = pose_quat_global * upright_q
    upright_q = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    root_quat_corrected = data[:, 3:7].cpu().numpy()                  # (T, 4)
    root_quat_smpl = (sRot.from_quat(root_quat_corrected) * upright_q).as_quat()
    root_aa_smpl = sRot.from_quat(root_quat_smpl).as_rotvec()         # (T, 3)

    # Local non-root rotations: stored in MuJoCo joint order (51 joints, 3
    # dims each, axis-angle). Reorder to SMPL order.
    # MuJoCo dof index k corresponds to MuJoCo joint index k+1 (since joint 0
    # is the root pelvis which has no DOF). SMPL joint index for the same
    # rotation is MUJOCO_2_SMPL_NEW[k+1].
    dof_mujoco = data[:, 9:9 + 153].cpu().numpy().reshape(T, 51, 3)   # (T, 51, 3)
    dof_smpl = np.zeros_like(dof_mujoco)                              # (T, 51, 3)
    for mujoco_joint_idx in range(1, 52):  # skip pelvis (joint 0)
        smpl_joint_idx = MUJOCO_2_SMPL_NEW[mujoco_joint_idx]
        # MuJoCo dof index = mujoco_joint_idx - 1 (because pelvis is dof-less)
        # SMPL non-root index = smpl_joint_idx - 1
        dof_smpl[:, smpl_joint_idx - 1, :] = dof_mujoco[:, mujoco_joint_idx - 1, :]

    # Assemble pose_aa: (T, 52*3)
    pose_aa = np.concatenate([root_aa_smpl, dof_smpl.reshape(T, 51 * 3)], axis=1)
    return pose_aa


def run_smplx_fk(pose_aa: np.ndarray, betas: np.ndarray, trans: np.ndarray,
                 gender: str) -> tuple[np.ndarray, np.ndarray]:
    """Run SMPL-X forward kinematics for a sequence.

    Returns:
      joints: (T, 52, 3) world-frame joint positions in SMPL joint order
      verts:  (T, V, 3)  world-frame vertex positions (we'll use these for
                         ground-clearance correction, mirroring interact2mimic.py)

    Uses the smplx package directly (interact2mimic.py wraps this in
    `forward_smpl` but it's simple enough to reproduce inline).
    """
    import torch
    import smplx

    T = pose_aa.shape[0]
    # SMPL-X expects num_betas to match the betas length (16 for OMOMO).
    # MODEL_PATH was set by smpl_local_robot at import time relative to
    # InterAct/simulation; we use the same convention here.
    smplx_model = smplx.create(
        model_path="../models",
        model_type="smplx",
        gender=gender,
        num_betas=betas.shape[0],
        flat_hand_mean=False,    # OMOMO setting (per interact2mimic.py:581)
        batch_size=T,
        use_pca=False,
    ).cuda()

    # SMPL-X full pose layout (when use_pca=False):
    #   pose_aa[:, 0:3]   = global_orient
    #   pose_aa[:, 3:66]  = body_pose (21 joints)
    #   pose_aa[:, 66:69] = jaw_pose
    #   pose_aa[:, 69:72] = leye_pose
    #   pose_aa[:, 72:75] = reye_pose
    #   pose_aa[:, 75:120]= left_hand_pose (15 joints)
    #   pose_aa[:, 120:165] = right_hand_pose (15 joints)
    # Our pose_aa is (T, 156) covering the 52 SMPL-X body+hand joints.
    # The 4 face joints (jaw + 2 eyes) aren't in our SMPL/MuJoCo ordering, so
    # we pad with zeros.
    pose_aa_t = torch.from_numpy(pose_aa).float().cuda()
    global_orient = pose_aa_t[:, 0:3]
    body_pose = pose_aa_t[:, 3:66]
    # SMPL-X has 21 body joints (after pelvis) which is exactly pose_aa[3:66].
    # The remaining joints in our (T, 156) are hands. Indices 66:156 → hands.
    # But SMPL-X expects hands at slots after jaw+eyes. So we have to shift:
    left_hand_pose = pose_aa_t[:, 66:66+45]    # joints 22..36 in SMPL → 15 hand joints
    right_hand_pose = pose_aa_t[:, 66+45:66+90]  # joints 37..51 in SMPL → 15 hand joints
    jaw_pose = torch.zeros(T, 3, device="cuda")
    leye_pose = torch.zeros(T, 3, device="cuda")
    reye_pose = torch.zeros(T, 3, device="cuda")

    betas_t = torch.from_numpy(betas[None, :]).float().expand(T, -1).cuda()
    trans_t = torch.from_numpy(trans).float().cuda()

    out = smplx_model(
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        betas=betas_t,
        transl=trans_t,
        return_full_pose=False,
        return_verts=True,
    )

    # joints has shape (T, J, 3) where J is SMPL-X's full joint set (more
    # than 52). We take the first 52 to match the body+hands subset.
    joints_full = out.joints.detach().cpu().numpy()
    joints_52 = joints_full[:, :52, :]
    verts = out.vertices.detach().cpu().numpy()
    return joints_52, verts


def retarget_clip(source_pt_path: Path, target_betas: np.ndarray,
                  target_gender: str, target_mjcf_path: Path,
                  output_pt_path: Path) -> None:
    """Retarget one .pt file from source body to target body.

    The high-level recipe is documented in the module docstring. This
    function implements it in one place so we can iterate on the algorithm
    without restructuring the whole pipeline.
    """
    import torch
    from scipy.spatial.transform import Rotation as sRot
    from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

    print(f"  [retarget] {source_pt_path.name} -> {output_pt_path.name}")

    # 1. Load source data + extract source SMPL-X pose
    data = torch.load(source_pt_path)
    T = data.shape[0]
    source_root_pos = data[:, 0:3].cpu().numpy()           # (T, 3)
    pose_aa = extract_pose_aa_from_pt(data)                # (T, 156)

    # 2. Run SMPL-X FK with target betas → joints in target body shape
    joints_smpl, verts = run_smplx_fk(pose_aa, target_betas, source_root_pos, target_gender)

    # 3. Apply the same ground-clearance correction interact2mimic.py uses:
    #    if any vertex is below z=0 in the FIRST frame, shift everything up.
    #    (We only check frame 0 because that's what interact2mimic.py does —
    #    not perfect for sequences where the lowest point is mid-sequence, but
    #    matches the convention so downstream behavior is consistent.)
    diff_fix = verts[0, :, 2].min()
    if diff_fix < 0:
        joints_smpl[..., 2] -= diff_fix
        verts[..., 2] -= diff_fix
        source_root_pos[..., 2] -= diff_fix
        print(f"  [retarget]   z-shifted by {-diff_fix:.4f} to clear ground")

    # 4. Build local rotations in MuJoCo order for the target skeleton tree.
    #    These are the SAME local rotations as source (joint angles are
    #    body-agnostic), just reordered.
    pose_quat_smpl = _aa_to_quat(pose_aa.reshape(T, 52, 3))     # (T, 52, 4)
    pose_quat_mujoco = pose_quat_smpl[:, SMPL_2_MUJOCO_NEW, :]  # (T, 52, 4)

    # 5. Use the TARGET MJCF's skeleton tree to compute global rotations.
    #    The skeleton tree differs from source's by its local_translations
    #    (link offsets are body-shape-specific).
    target_skeleton = SkeletonTree.from_mjcf(str(target_mjcf_path))
    root_trans_offset = torch.from_numpy(source_root_pos) + target_skeleton.local_translation[0]
    sk_state = SkeletonState.from_rotation_and_root_translation(
        target_skeleton,
        torch.from_numpy(pose_quat_mujoco),
        root_trans_offset,
        is_local=True,
    )
    # interact2mimic.py applies an upright_start correction here (line 795).
    # We do the same so the resulting body_rot matches the convention the
    # InterMimic env expects.
    upright_q = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    global_rot_raw = sk_state.global_rotation.cpu().numpy().reshape(-1, 4)
    pose_quat_global = (sRot.from_quat(global_rot_raw) * upright_q.inv()).as_quat().reshape(T, 52, 4)

    # 6. Build new body_pos in MuJoCo order (joints) with target body shape.
    #    For SMPL-X interact2mimic uses smpl_2_mujoco_new (line 877).
    new_body_pos_mujoco = joints_smpl[:, SMPL_2_MUJOCO_NEW, :]    # (T, 52, 3)

    # 7. Hand-object offset adjustment: preserve per-frame hand-to-object
    #    offset across the retargeting. Strategy: for each frame, compute the
    #    "carry hand position" as the mean of L_Wrist + R_Wrist on the SOURCE
    #    body (read from source's body_pos slice). Compute hand-obj offset.
    #    Then re-attach the object to the TARGET body's mean-wrist position.
    source_body_pos_mujoco = data[:, 162:318].cpu().numpy().reshape(T, 52, 3)
    source_obj_pos = data[:, 318:321].cpu().numpy()              # (T, 3)
    # MuJoCo joint indices for left and right wrists. In SMPL these are
    # joints 20 (L_Wrist) and 21 (R_Wrist). Convert via smpl_2_mujoco_new.
    L_WRIST_MUJOCO = SMPL_2_MUJOCO_NEW[20]
    R_WRIST_MUJOCO = SMPL_2_MUJOCO_NEW[21]
    source_mean_wrist = 0.5 * (source_body_pos_mujoco[:, L_WRIST_MUJOCO]
                               + source_body_pos_mujoco[:, R_WRIST_MUJOCO])  # (T, 3)
    target_mean_wrist = 0.5 * (new_body_pos_mujoco[:, L_WRIST_MUJOCO]
                               + new_body_pos_mujoco[:, R_WRIST_MUJOCO])     # (T, 3)
    # offset is "where the obj sits relative to the carry-hand" in source.
    # We re-attach the obj at the same offset relative to target's carry-hand.
    hand_to_obj_offset = source_obj_pos - source_mean_wrist                  # (T, 3)
    new_obj_pos = target_mean_wrist + hand_to_obj_offset                     # (T, 3)

    # 8. Assemble retargeted data tensor (clone source, overwrite the slices
    #    we changed). Channels 7:9 and 325:330 are passed through unchanged —
    #    interact2mimic.py leaves them at zero, so they should be zero either
    #    way, but cloning preserves whatever was there.
    new_data = data.clone()
    new_data[:, 0:3] = torch.from_numpy(source_root_pos)
    new_data[:, 3:7] = torch.from_numpy(pose_quat_global[:, 0, :])    # root global quat
    # 9:162 (dof_pos) preserved from source — joint angles are body-agnostic
    new_data[:, 162:318] = torch.from_numpy(new_body_pos_mujoco.reshape(T, -1))
    new_data[:, 318:321] = torch.from_numpy(new_obj_pos)
    # 321:325 (obj_rot) preserved — object rotation independent of body shape
    # 330:331, 331:383 (contact) preserved — contact pattern same as source
    new_data[:, 383:591] = torch.from_numpy(pose_quat_global.reshape(T, -1))

    # 9. Save. Use atomic write (tmp+rename) so a killed run doesn't leave
    #    half-written .pt files that downstream code mistakes for valid data
    #    (lesson from feedback_verify_upstream_before_debugging).
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_pt_path.with_suffix(".pt.tmp")
    torch.save(new_data, tmp_path)
    tmp_path.replace(output_pt_path)
    print(f"  [retarget]   wrote {output_pt_path} ({new_data.shape})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-sub", required=True,
                        help="Source subject ID (whose motion we're retargeting), e.g. sub2")
    parser.add_argument("--target-sub", required=True,
                        help="Target subject ID (whose body shape we're retargeting to), e.g. sub8")
    # Default to absolute paths since we chdir into InterAct/simulation later
    # for the smpl_local_robot imports — relative defaults break after chdir.
    parser.add_argument("--betas-npz",
                        default=(Path(__file__).resolve().parent / "omomo_betas.npz"),
                        type=Path,
                        help="Per-subject betas lookup (from extract_omomo_betas.py)")
    parser.add_argument("--source-dir", type=Path,
                        default=Path(INTERMIMIC_ROOT) / "InterAct" / "OMOMO",
                        help="Where the source subject's .pt files live")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(INTERMIMIC_ROOT) / "InterAct" / "OMOMO_cross",
                        help="Where to write retargeted .pt files")
    parser.add_argument("--mjcf-dir", type=Path,
                        default=Path(INTERMIMIC_ROOT) / "isaacgym" / "src" / "intermimic" / "data" / "assets" / "smplx",
                        help="Where per-subject MJCFs live (from generate_per_subject_mjcfs.py)")
    parser.add_argument("--interact-root", type=Path, default=Path(INTERACT_ROOT),
                        help="InterAct repo root, used to find smpl_local_robot + models")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only retarget the first N clips (smoke test)")
    args = parser.parse_args()

    # Force all path args to absolute before chdir-ing in setup_interact_paths,
    # otherwise relative paths break once cwd changes.
    args.betas_npz = args.betas_npz.resolve()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.mjcf_dir = args.mjcf_dir.resolve()

    # Chdir + sys.path before any FK-using import. See setup_interact_paths.
    setup_interact_paths(args.interact_root)

    # Load betas lookup and gender info
    betas_data = np.load(args.betas_npz, allow_pickle=True)
    genders: dict[str, str] = {}
    for entry in betas_data["_genders"]:
        sub_id, gender = str(entry).split(":", 1)
        genders[sub_id] = gender

    target_betas = np.asarray(betas_data[args.target_sub])
    target_gender = genders[args.target_sub]
    target_mjcf = args.mjcf_dir / f"smplx_omomo_{args.target_sub}.xml"
    if not target_mjcf.exists():
        print(f"ERROR: target MJCF not found: {target_mjcf}", file=sys.stderr)
        print(f"Run generate_per_subject_mjcfs.py --subjects {args.target_sub} first.",
              file=sys.stderr)
        return 1

    # Find source clips. Filename convention: sub<src>_<obj>_<idx>.pt
    source_clips = sorted(args.source_dir.glob(f"{args.source_sub}_*.pt"))
    if not source_clips:
        print(f"ERROR: no source clips found at {args.source_dir}/{args.source_sub}_*.pt",
              file=sys.stderr)
        return 1
    if args.limit is not None:
        source_clips = source_clips[:args.limit]

    print(f"Retargeting {args.source_sub} → {args.target_sub}: "
          f"{len(source_clips)} clips, target MJCF={target_mjcf.name}")
    print(f"  source body |betas|=({np.linalg.norm(np.asarray(betas_data[args.source_sub])):.2f})  "
          f"target body |betas|=({np.linalg.norm(target_betas):.2f})")

    for clip in source_clips:
        # Output naming: replace the source sub_id with "sub{src}to{tgt}".
        # E.g. sub2_largetable_007.pt -> sub2to8_largetable_007.pt
        parts = clip.stem.split("_")
        new_stem = f"{args.source_sub}to{args.target_sub[3:]}_{'_'.join(parts[1:])}"
        output_pt = args.output_dir / f"{new_stem}.pt"
        retarget_clip(clip, target_betas, target_gender, target_mjcf, output_pt)

    print(f"\nDone. {len(source_clips)} cross-body .pt file(s) under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

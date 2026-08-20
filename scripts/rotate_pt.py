#!/usr/bin/env python3
"""Apply a fixed global rotation to all position + orientation channels of an
InterMimic 591-channel motion tensor.

Use when the converted clip appears wrong-side-up or backwards in the sim
because CARI4D's world frame doesn't match what interact2mimic.py assumed.

Usage:
    # 180-degree flip around X (upside-down fix)
    python scripts/rotate_pt.py /path/to/sub99_gas_000.pt --axis x --degrees 180

    # 180-degree yaw flip (facing-backwards fix)
    python scripts/rotate_pt.py /path/to/sub99_gas_000.pt --axis z --degrees 180

The file is overwritten in place after backing up to <path>.bak.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


# 591-channel layout, from intermimic.py:_load_motion:
# 0:3 root_pos, 3:7 root_rot (xyzw), 7:9 pad, 9:162 dof_pos (51*3),
# 162:318 body_pos (52*3), 318:321 obj_pos, 321:325 obj_rot (xyzw),
# 325:330 pad, 330:331 contact_obj, 331:383 contact_human (52),
# 383:591 body_rot (52*4, xyzw)


def rotation_from_calibration(spec: str) -> sRot:
    """Return the camera-to-world rotation from an Ego-Exo4D gopro_calibs.csv.

    CARI4D reconstructs in the camera's frame, so 'up' in its output is wherever
    that camera's up happened to point. A humanoid handed to a physics engine
    that way falls over, and guessing the correction as a single-axis flip only
    works when the camera happened to be axis-aligned.

    The calibration already holds the answer. Its columns are named *_world_cam,
    so the quaternion is the camera's orientation in the world frame -- and that
    world frame is gravity-aligned, which is what makes it the right target. On
    the basketball take all four cameras sit at z = -0.05, consistent with z
    being vertical.

    Args:
        spec: "<path to gopro_calibs.csv>:<cam_uid>", e.g.
            "trajectory/gopro_calibs.csv:cam04".

    Returns:
        The rotation taking camera-frame vectors to world frame.

    Raises:
        SystemExit: if the file, the camera, or the columns are missing.
    """
    import csv

    if ":" not in spec:
        raise SystemExit("--from-calib wants <csv>:<cam_uid>, e.g. "
                         "trajectory/gopro_calibs.csv:cam04")
    path, _, cam_uid = spec.rpartition(":")
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise SystemExit(f"no calibration at {csv_path}")

    with open(csv_path) as f:
        rows = {r["cam_uid"]: r for r in csv.DictReader(f)}
    if cam_uid not in rows:
        raise SystemExit(f"{cam_uid} not in {csv_path}; found {sorted(rows)}")
    row = rows[cam_uid]

    try:
        quat = [float(row[f"q{a}_world_cam"]) for a in "xyzw"]
    except KeyError as exc:
        raise SystemExit(f"{csv_path} has no q*_world_cam columns ({exc}); this "
                         f"expects Ego-Exo4D's gopro_calibs.csv layout")

    R = sRot.from_quat(quat)
    up = R.apply([0.0, -1.0, 0.0])
    print(f"using --from-calib {cam_uid}: camera-to-world rotation "
          f"{np.round(R.as_quat(), 4)}")
    print(f"  the camera's up maps to world {np.round(up, 3)} "
          f"-- a well-conditioned fix has this close to a world axis")
    return R


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pt_path", type=Path)
    parser.add_argument("--axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--from-calib", default=None, metavar="CSV:CAM_UID",
                        help="Derive the rotation from camera calibration instead of "
                             "guessing an axis and angle. CARI4D outputs in the "
                             "camera's frame, so a humanoid's 'up' is wherever the "
                             "camera's up happened to point and it falls over in sim. "
                             "Ego-Exo4D's gopro_calibs.csv gives each camera's pose in "
                             "a gravity-aligned world frame, which is exactly the "
                             "rotation needed. Example: "
                             "--from-calib trajectory/gopro_calibs.csv:cam04")
    parser.add_argument("--align-gravity", type=int, nargs=2, metavar=("A", "Z"),
                        help="Derive the rotation from the clip's own physics: the "
                             "object's mean free-flight acceleration over frames A..Z "
                             "IS gravity, so rotate the whole scene to map it onto "
                             "world -z. No calibration, no axis guessing -- this is "
                             "the fix for residual tilt left by the fixed x-flip when "
                             "an export's frame differs from the assumed inversion "
                             "(measured 12.55 deg on the optj3d install). Pair with "
                             "--drop-to-floor and --out <new file>.")
    parser.add_argument("--fix-frame-zero", action="store_true",
                        help="Compute the rotation that makes frame 0's root_rot "
                             "into the identity quaternion, then apply that rotation "
                             "to the whole scene (root + body + object). Use this "
                             "when the CARI4D frame is offset by an arbitrary "
                             "rotation that you don't know the axis/degrees for.")
    parser.add_argument("--around-root", action="store_true",
                        help="Rotate around each frame's root_pos rather than the "
                             "world origin. Keeps the figure at its original world "
                             "location (no submerging below the floor) and preserves "
                             "figure-object relative geometry. Recommended for "
                             "fixing CARI4D's upside-down output without breaking "
                             "the motion.")
    parser.add_argument("--drop-to-floor", action="store_true",
                        help="After rotating, shift the whole clip vertically so "
                             "the feet rest on the ground. Use this INSTEAD of "
                             "--around-root: rotating about the world origin "
                             "keeps orientation and trajectory consistent, and "
                             "this fixes the height it costs. --around-root "
                             "avoids the height problem by not rotating "
                             "root_pos at all, which leaves the body facing one "
                             "way while its path still runs the other -- the "
                             "figure appears to walk backwards.")
    parser.add_argument("--floor-percentile", type=float, default=10.0,
                        help="Percentile of per-frame lowest-foot height taken "
                             "as ground contact (default: 10). Not the minimum: "
                             "one frame of a foot clipping through the floor "
                             "would then lift the entire clip into the air.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Default: overwrite input with <input>.bak backup.")
    args = parser.parse_args()

    sources = [args.from_calib is not None, args.fix_frame_zero,
               args.align_gravity is not None,
               args.axis is not None or args.degrees is not None]
    if sum(sources) != 1:
        parser.error("specify exactly one of --from-calib, --fix-frame-zero, "
                     "--align-gravity, or both --axis and --degrees")
    if args.axis is not None and args.degrees is None:
        parser.error("--axis needs --degrees")

    src = args.pt_path.expanduser().resolve()
    if not src.is_file():
        print(f"not a file: {src}", file=sys.stderr)
        return 2

    dst = args.out.expanduser().resolve() if args.out else src

    data = torch.load(str(src), map_location="cpu")
    print(f"loaded {src.name}: shape {tuple(data.shape)}, dtype {data.dtype}")
    if data.shape[-1] != 591:
        print(f"unexpected channel count {data.shape[-1]} (want 591)", file=sys.stderr)
        return 2

    if args.align_gravity:
        import numpy as _np
        a, z = args.align_gravity
        obj = data[:, 318:321].double().numpy()
        acc = _np.diff(obj, n=2, axis=0)[a:z - 1]
        g = acc.mean(0)
        g = g / _np.linalg.norm(g)
        target = _np.array([0.0, 0.0, -1.0])
        tilt = _np.degrees(_np.arccos(_np.clip(g @ target, -1, 1)))
        axis = _np.cross(g, target)
        n = _np.linalg.norm(axis)
        if n < 1e-8:
            R = sRot.identity()
        else:
            R = sRot.from_rotvec(axis / n * _np.radians(tilt))
        print(f"align-gravity: flight frames {a}-{z}, measured g dir {_np.round(g,3)}, "
              f"tilt {tilt:.2f} deg -> rotating onto -z")
    elif args.from_calib:
        R = rotation_from_calibration(args.from_calib)
    elif args.fix_frame_zero:
        frame0_root_rot = data[0, 3:7].numpy()
        R = sRot.from_quat(frame0_root_rot).inv()
        print(f"using fix-frame-zero: inverse of frame 0 root_rot = {R.as_quat()}")
    else:
        R = sRot.from_euler(args.axis, args.degrees, degrees=True)
    R_mat = torch.tensor(R.as_matrix(), dtype=data.dtype)        # (3,3)
    R_quat = R.as_quat()                                          # (4,) xyzw

    T = data.shape[0]

    # Pull out root_pos before any modifications — used as per-frame rotation
    # center if --around-root is set.
    root_pos = data[:, 0:3].clone()                               # (T, 3)

    def rot_positions(slice_):
        # .clone().reshape() instead of .view(): the .view() path silently
        # failed to write body_pos (slice 162:318) back to `data` even though
        # no exception was raised — the rotated tensor was computed but the
        # assignment didn't persist. .clone().reshape() materializes a fresh
        # contiguous tensor and the data[:, slice_] = ... assignment writes
        # correctly.
        flat = data[:, slice_].clone().reshape(T, -1, 3)          # (T, N, 3)
        if args.around_root:
            centered = flat - root_pos.unsqueeze(1)
            rotated = centered @ R_mat.T + root_pos.unsqueeze(1)
        else:
            rotated = flat @ R_mat.T
        data[:, slice_] = rotated.reshape(T, -1)

    rot_positions(slice(162, 318))                                # body_pos
    rot_positions(slice(318, 321))                                # obj_pos
    if not args.around_root:
        # When rotating around world origin, also rotate root_pos. When rotating
        # around root_pos itself, leave it (figure stays at same world location).
        rot_positions(slice(0, 3))

    # Rotations (quaternions xyzw): premultiply by R_quat regardless of mode.
    def rot_quats(slice_):
        flat = data[:, slice_].view(T, -1, 4).numpy().reshape(-1, 4)
        new = (sRot.from_quat(R_quat) * sRot.from_quat(flat)).as_quat()
        data[:, slice_] = torch.tensor(new.reshape(T, -1), dtype=data.dtype)

    rot_quats(slice(3, 7))                                        # root_rot
    rot_quats(slice(321, 325))                                    # obj_rot
    rot_quats(slice(383, 591))                                    # body_rot

    if args.drop_to_floor:
        # interact2mimic.py:853 seats the clip on the ground in ITS frame; any
        # rotation invalidates that, so the height has to be re-fit here.
        # Applied as one constant offset to every position channel, which leaves
        # the motion and the human-object relationship untouched -- only where
        # the whole scene sits vertically changes.
        feet = [7, 8, 10, 11]                # L/R Ankle, L/R Toe
        body = data[:, 162:318].clone().reshape(T, -1, 3)
        lowest = body[:, feet, 2].min(dim=1).values.numpy()
        offset = float(-np.percentile(lowest, args.floor_percentile))
        data[:, 2] += offset                                      # root_pos z
        body[:, :, 2] += offset
        data[:, 162:318] = body.reshape(T, -1)
        data[:, 320] += offset                                    # obj_pos z
        print(f"drop-to-floor: lowest foot ran {lowest.min():.2f}..{lowest.max():.2f} m, "
              f"shifted by {offset:+.3f} m")

    if dst == src:
        # Kept OUT of the motion directory. InterMimic enumerates that directory
        # and matches on the sub<N>_ prefix, so a backup sitting beside the clip
        # loads as a second motion and the replay picks between the two at
        # random -- the edited clip and its pre-edit copy, alternating for no
        # visible reason.
        backup_dir = src.parent / ".rotate_pt_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / (src.name + ".bak")
        if not backup.exists():
            shutil.copy(str(src), str(backup))
            print(f"backed up original to {backup}")

    torch.save(data, str(dst))
    print(f"wrote rotated tensor to {dst}")
    # Report the mode that actually ran: --axis/--degrees are None on the other
    # two paths, and printing "applied None° around None-axis" after a
    # successful calibration rotation reads like a failure.
    if args.from_calib:
        print(f"applied the camera-to-world rotation from {args.from_calib}")
    elif args.fix_frame_zero:
        print("applied the inverse of frame 0's root rotation")
    else:
        print(f"applied {args.degrees}° around {args.axis}-axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pt_path", type=Path)
    parser.add_argument("--axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--degrees", type=float, default=None)
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
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Default: overwrite input with <input>.bak backup.")
    args = parser.parse_args()

    if not args.fix_frame_zero and (args.axis is None or args.degrees is None):
        parser.error("specify either --fix-frame-zero, or both --axis and --degrees")

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

    if args.fix_frame_zero:
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
        flat = data[:, slice_].view(T, -1, 3)                     # (T, N, 3)
        if args.around_root:
            # Per-frame rotation around root_pos[t]. Preserves figure-object
            # relative geometry and keeps figure at its original world location.
            centered = flat - root_pos.view(T, 1, 3)
            rotated = centered @ R_mat.T + root_pos.view(T, 1, 3)
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

    if dst == src:
        backup = src.with_suffix(src.suffix + ".bak")
        if not backup.exists():
            shutil.copy(str(src), str(backup))
            print(f"backed up original to {backup.name}")

    torch.save(data, str(dst))
    print(f"wrote rotated tensor to {dst}")
    print(f"applied {args.degrees}° around {args.axis}-axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

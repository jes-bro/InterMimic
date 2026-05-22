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
    parser.add_argument("--axis", choices=["x", "y", "z"], required=True)
    parser.add_argument("--degrees", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Default: overwrite input with <input>.bak backup.")
    args = parser.parse_args()

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

    R = sRot.from_euler(args.axis, args.degrees, degrees=True)
    R_mat = torch.tensor(R.as_matrix(), dtype=data.dtype)        # (3,3)
    R_quat = R.as_quat()                                          # (4,) xyzw

    T = data.shape[0]

    # Positions: matrix multiply
    def rot_positions(slice_):
        flat = data[:, slice_].view(T, -1, 3)                     # (T, N, 3)
        rotated = flat @ R_mat.T                                  # (T, N, 3)
        data[:, slice_] = rotated.reshape(T, -1)

    rot_positions(slice(0, 3))                                    # root_pos
    rot_positions(slice(162, 318))                                # body_pos (52*3)
    rot_positions(slice(318, 321))                                # obj_pos

    # Rotations (quaternions xyzw): premultiply by R_quat
    def rot_quats(slice_):
        flat = data[:, slice_].view(T, -1, 4).numpy().reshape(-1, 4)
        new = (sRot.from_quat(R_quat) * sRot.from_quat(flat)).as_quat()
        data[:, slice_] = torch.tensor(new.reshape(T, -1), dtype=data.dtype)

    rot_quats(slice(3, 7))                                        # root_rot
    rot_quats(slice(321, 325))                                    # obj_rot
    rot_quats(slice(383, 591))                                    # body_rot (52*4)

    # dof_pos, contact labels: invariant under world rotation, no change.

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

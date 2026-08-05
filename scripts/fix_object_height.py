#!/usr/bin/env python3
"""Diagnose, and optionally correct, an object that floats above the floor.

A tracked ball that never comes within 27 cm of the ground can be wrong in two
ways that look identical in a render and want opposite fixes:

  Wrong against the floor, right against the hands
      A vertical offset in the object channel alone. Shifting it down fixes the
      bounce and leaves the interaction intact.

  Wrong against both
      The object is mistracked, and shifting it to meet the floor would drag it
      out of the player's hands. Nothing here can fix that; the depth has to
      improve.

The distinction is the wrist-to-ball distance, which this reports before and
after any shift. During a dribble the hand meets the ball once per cycle, so the
minimum of that distance over the clip should be about the ball's radius -- the
wrist sits on its surface, not at its centre.

    python scripts/fix_object_height.py --pt <clip> --mesh <ball .obj>
    python scripts/fix_object_height.py --pt <clip> --mesh <ball .obj> --apply auto
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


OBJ_POS = slice(318, 321)
BODY_POS = slice(162, 318)
L_WRIST, R_WRIST = 20, 21


def wrist_ball_gap(body: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Return the per-frame distance from the nearer wrist to the ball centre."""
    left = np.linalg.norm(body[:, L_WRIST] - obj, axis=1)
    right = np.linalg.norm(body[:, R_WRIST] - obj, axis=1)
    return np.minimum(left, right)


def report_gap(body: np.ndarray, obj: np.ndarray, radius: float, tag: str) -> float:
    """Print how close the hand comes to the ball, and return that minimum."""
    gap = wrist_ball_gap(body, obj)
    closest = float(gap.min())
    print(f"  {tag:<8} wrist-to-ball  min {closest:.3f} m  "
          f"median {float(np.median(gap)):.3f} m  "
          f"(contact would be about {radius:.3f})")
    return closest


def main() -> int:
    """Report the two relationships, and shift the object if asked."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--apply", default=None, metavar="auto|METRES",
                        help="Shift the object channel vertically. 'auto' lands "
                             "its lowest near-floor point on the ball's radius; "
                             "a number shifts by exactly that (negative is "
                             "down). Omit to diagnose without writing.")
    parser.add_argument("--max-contact-height", type=float, default=0.6,
                        help="ignore minima above this when finding the lowest "
                             "bounce (default: 0.6)")
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError:
        raise SystemExit("trimesh is required; it ships in the intermimic env")
    mesh = trimesh.load(str(args.mesh.expanduser().resolve()), force="mesh",
                        process=False)
    radius = float(np.asarray(mesh.extents, dtype=float).mean() / 2.0)

    src = args.pt.expanduser().resolve()
    data = torch.load(str(src), map_location="cpu")
    if data.shape[-1] != 591:
        raise SystemExit(f"{src.name}: {data.shape[-1]} channels, want 591")
    obj = data[:, OBJ_POS].numpy().copy()
    body = data[:, BODY_POS].reshape(data.shape[0], -1, 3).numpy()

    print(f"{src.name}: {len(obj)} frames, ball radius {radius:.3f} m")
    low = obj[:, 2].min()
    near_floor = obj[obj[:, 2] <= args.max_contact_height]
    print(f"  ball lowest point {low:.3f} m -> {low - radius:+.3f} m of clearance")
    print(f"  {len(near_floor)} of {len(obj)} frames below "
          f"{args.max_contact_height:.2f} m")
    before = report_gap(body, obj, radius, "current")

    # The hand relationship decides whether a shift is legitimate. If the hand
    # already reaches the ball, the object is only wrong against the floor.
    hand_ok = abs(before - radius) < 0.10
    print()
    if hand_ok:
        print(f"  the hand does reach the ball ({before:.3f} vs {radius:.3f} m), "
              f"so the object is wrong against the FLOOR only -- a shift is the "
              f"right fix.")
    else:
        print(f"  the hand does NOT reach the ball ({before:.3f} vs "
              f"{radius:.3f} m). The object is mistracked relative to the person "
              f"as well, and shifting it to the floor would pull it further from "
              f"his hands. Better depth is the fix, not this script.")

    if args.apply is None:
        print("\nDiagnosis only; pass --apply to write.")
        return 0

    if args.apply == "auto":
        shift = radius - low
    else:
        try:
            shift = float(args.apply)
        except ValueError:
            raise SystemExit(f"--apply wants 'auto' or a number, got {args.apply!r}")
    print(f"\napplying {shift:+.3f} m to the object channel")

    data[:, 320] += shift
    obj_after = data[:, OBJ_POS].numpy()
    after = report_gap(body, obj_after, radius, "shifted")
    if not hand_ok and abs(after - radius) > abs(before - radius):
        print("  NOTE: the shift moved the ball further from his hands, which is "
              "the outcome predicted above. Written anyway, since you asked, but "
              "the interaction is now worse than before.")

    backup_dir = src.parent / ".rotate_pt_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (src.name + ".preshift")
    if not backup.exists():
        shutil.copy(str(src), str(backup))
        print(f"  backed up original to {backup}")
    torch.save(data, str(src))
    print(f"  wrote {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

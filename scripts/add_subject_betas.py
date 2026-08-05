#!/usr/bin/env python3
"""Add a CARI4D subject's body shape to a betas file, for the mesh renderers.

render_mesh_replay.py and render_mesh_gallery_offline.py shape their SMPL-X mesh
from a betas archive keyed by subject. That archive covers the OMOMO subjects, so
a subject reconstructed from video has no entry and the renderers stop with
"has no betas entry".

The shape is already known -- cari4d_to_interact.py wrote it into the sequence's
human.npz. This copies it across.

    python scripts/add_subject_betas.py \\
        --human InterAct/data/behave_cari4d/sequences_canonical/sub100_bball_000/human.npz \\
        --subject sub100

Writes a NEW archive by default rather than editing omomo_betas.npz in place, so
the file the existing figures depend on is left alone.

One caveat worth knowing: these betas describe an SMPL-H body and the renderer
builds SMPL-X. The two shape spaces are similar but not identical, so the mesh is
a close likeness rather than the exact body the physics used. For seeing what the
retarget looks like that is fine; for measuring a body it is not.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def read_betas(path: Path) -> np.ndarray:
    """Return the (N,) shape vector from an InterAct human.npz or a .npy.

    Raises:
        SystemExit: if the file is missing or carries no recognisable betas,
            rather than writing an archive with a silently wrong shape in it.
    """
    if not path.is_file():
        raise SystemExit(f"no file at {path}")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).ravel()
    data = np.load(path)
    for key in ("beta", "betas"):
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32).ravel()
    raise SystemExit(f"{path.name} has no 'beta' or 'betas'; found {data.files}")


def main() -> int:
    """Merge one subject's betas into a copy of the archive."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, required=True,
                        help="the sequence's human.npz (or a .npy of betas)")
    parser.add_argument("--subject", required=True,
                        help="key the renderers will look up, e.g. sub100")
    parser.add_argument("--base", type=Path, default=Path("scripts/omomo_betas.npz"),
                        help="archive to start from (default: scripts/omomo_betas.npz)")
    parser.add_argument("--out", type=Path, default=Path("scripts/cari4d_betas.npz"),
                        help="archive to write (default: scripts/cari4d_betas.npz)")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting an existing key")
    args = parser.parse_args()

    betas = read_betas(args.human.expanduser().resolve())
    print(f"{args.human.name}: {betas.size} betas, "
          f"range {betas.min():+.3f}..{betas.max():+.3f}")

    entries = {}
    if args.base.is_file():
        # allow_pickle: the OMOMO archive stores object arrays, which numpy
        # refuses to read without it. Entries are carried through untouched, so
        # whatever they are they survive the round trip.
        base = np.load(args.base, allow_pickle=True)
        entries = {k: base[k] for k in base.files}
        print(f"{args.base.name}: {len(entries)} existing subjects")
    else:
        print(f"{args.base} not found; starting a new archive")

    if args.subject in entries and not args.force:
        raise SystemExit(f"'{args.subject}' is already in {args.base.name}; "
                         f"pass --force to replace it")

    # Match the width the archive already uses, so the renderer's model is built
    # with the number of coefficients it expects. CARI4D works in 10; OMOMO
    # entries may be wider, and a short vector would be read as a different body.
    # Only entries that are plainly numeric vectors can say what width to match;
    # an object array's size is the count of objects, not of coefficients, and
    # padding to that would produce a body no one asked for.
    widths = [v.size for v in entries.values()
              if getattr(v, "dtype", None) is not None
              and v.dtype != object and v.ndim == 1]
    width = max(widths, default=betas.size)
    if betas.size < width:
        padded = np.zeros(width, dtype=np.float32)
        padded[:betas.size] = betas
        print(f"padded {betas.size} betas to {width} with zeros to match the archive")
        betas = padded
    elif betas.size > width:
        print(f"truncated {betas.size} betas to {width} to match the archive")
        betas = betas[:width]

    entries[args.subject] = betas
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        if not backup.exists():
            shutil.copy(str(args.out), str(backup))
            print(f"backed up the previous {args.out.name} to {backup.name}")
    np.savez(args.out, **entries)
    print(f"wrote {args.out} with {len(entries)} subjects, including "
          f"'{args.subject}'")
    print()
    print("Render with:")
    print(f"  python scripts/render_mesh_replay.py --clip "
          f"InterAct/behave_cari4d/{args.subject}_<object>_000.pt \\")
    print(f"      --subject {args.subject} --betas {args.out} --out mesh_replay.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

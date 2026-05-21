#!/usr/bin/env python3
"""Upgrade an old-format SMPL-H .pkl to the format modern smplx expects.

Pre-2020 SMPL-H pickles (from MANO/AGORA, which CARI4D's docs reference)
use slightly different attribute names than current smplx wants. The most
common discrepancies:

    old key                         new key (what smplx expects)
    ─────────────────────────       ──────────────────────────────
    hands_components                hands_componentsl + hands_componentsr
    hands_mean                      hands_meanl + hands_meanr
    hands_coeffs                    hands_coeffsl + hands_coeffsr
    hands_components_l/_r           hands_componentsl/r        (drop underscore)
    hands_mean_l/_r                 hands_meanl/r
    hands_coeffs_l/_r               hands_coeffsl/r

This script inspects the .pkl, prints what it found, and writes a new
.pkl with renamed keys. Run it once per file:

    python scripts/upgrade_smplh_pkl.py path/to/SMPLH_male.pkl path/to/SMPLH_male_smplx.pkl

Then symlink the new file in place of the old one.
"""

import argparse
import pickle
import sys
from pathlib import Path


# Add to this mapping if a new pattern shows up.
RENAMES = {
    # underscore-style old format → smplx style
    "hands_components_l": "hands_componentsl",
    "hands_components_r": "hands_componentsr",
    "hands_mean_l":       "hands_meanl",
    "hands_mean_r":       "hands_meanr",
    "hands_coeffs_l":     "hands_coeffsl",
    "hands_coeffs_r":     "hands_coeffsr",
}


# Keys that may exist as a single combined array in really old formats.
# We DON'T auto-split these because the right semantic split is non-obvious;
# the script will print a warning and exit if it sees one without the
# corresponding l/r keys.
AMBIGUOUS_OLD = {"hands_components", "hands_mean", "hands_coeffs"}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_pkl", type=Path)
    parser.add_argument("output_pkl", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing the output.")
    args = parser.parse_args()

    if not args.input_pkl.is_file():
        print(f"not a file: {args.input_pkl}", file=sys.stderr)
        return 2

    with args.input_pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    print(f"[upgrade_smplh] {args.input_pkl.name}")
    print(f"[upgrade_smplh]   keys: {sorted(data.keys())}")

    # Check whether the keys smplx wants are already present.
    smplx_wanted = {"hands_componentsl", "hands_componentsr",
                    "hands_meanl", "hands_meanr"}
    already_present = smplx_wanted & set(data.keys())
    if already_present == smplx_wanted:
        print("[upgrade_smplh]   .pkl already has the smplx-format keys; nothing to do.")
        return 0

    new_data = dict(data)
    renamed = []
    for old, new in RENAMES.items():
        if old in new_data:
            new_data[new] = new_data.pop(old)
            renamed.append(f"{old} -> {new}")

    if renamed:
        print("[upgrade_smplh]   renamed:")
        for r in renamed:
            print(f"     {r}")
    else:
        # Nothing matched — check if there's an ambiguous single combined key.
        ambiguous = AMBIGUOUS_OLD & set(data.keys())
        if ambiguous:
            print(f"[upgrade_smplh]   found ambiguous combined keys: {sorted(ambiguous)}", file=sys.stderr)
            print(f"[upgrade_smplh]   these can't be auto-split into l/r — your .pkl may", file=sys.stderr)
            print(f"[upgrade_smplh]   be a much older format. Paste the key list above to me.", file=sys.stderr)
            return 3
        else:
            print(f"[upgrade_smplh]   no recognized old-style keys found.", file=sys.stderr)
            print(f"[upgrade_smplh]   the smplx error may be a different issue — paste key list to me.", file=sys.stderr)
            return 3

    # Verify the result actually has what smplx needs.
    missing_after = smplx_wanted - set(new_data.keys())
    if missing_after:
        print(f"[upgrade_smplh]   after rename, still missing: {sorted(missing_after)}", file=sys.stderr)
        return 3

    if args.dry_run:
        print("[upgrade_smplh]   --dry-run: not writing output")
        return 0

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[upgrade_smplh]   wrote {args.output_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

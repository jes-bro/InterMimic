#!/usr/bin/env python3
"""Extract per-subject SMPL-X betas from the raw OMOMO download.

OMOMO's raw distribution stores motion clips in joblib-pickled dicts keyed by
integer clip indices. Each clip has its own `betas` field, but in practice every
clip belonging to the same subject shares the same (1, 16) betas vector — i.e.
betas is a per-*subject* quantity that's just been redundantly duplicated per
clip. This script consolidates that redundancy into a single npz lookup:

    {'sub1': (16,), 'sub2': (16,), ..., 'sub17': (16,)}

We need this lookup for every downstream piece of the cross-body retargeting
pipeline:
  1. Generating per-subject MJCFs    (need each subject's own betas)
  2. Kinematic cross-body retarget   (need target subject's betas to run SMPL FK)
  3. Policy observation conditioning (concat source+target betas to obs)

Inputs:
  /home/jess/Downloads/data/train_diffusion_manip_seq_joints24.p   (sub1-sub15)
  /home/jess/Downloads/data/test_diffusion_manip_seq_joints24.p    (sub16, sub17)

Output:
  scripts/omomo_betas.npz with one (16,) array per 'sub<N>' key + gender info.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


# Paths to the raw OMOMO joblib dumps. These were downloaded from the OMOMO
# release into ~/Downloads/data/ — they are NOT the same as InterAct/OMOMO/,
# which contains InterMimic-ready .pt files (already SMPL FK'd, only sub2+sub8
# processed so far).
DEFAULT_TRAIN_P = Path("/home/jess/Downloads/data/train_diffusion_manip_seq_joints24.p")
DEFAULT_TEST_P = Path("/home/jess/Downloads/data/test_diffusion_manip_seq_joints24.p")


def extract_betas_from_pickle(pkl_path: Path) -> dict[str, dict]:
    """Load one OMOMO pickle and group its clips by subject.

    Returns a dict keyed by 'sub<N>' with one entry per subject:
        {
          'sub2': {
              'betas':  np.ndarray of shape (16,),  # the canonical shape vector
              'gender': 'male' or 'female',
              'count':  int  # how many clips this subject has
          },
          ...
        }

    Sanity-checks that all clips of a given subject have *exactly* the same
    betas — if not, we raise rather than silently averaging, since divergent
    betas would mean OMOMO's data model isn't what we think it is and the
    downstream pipeline assumptions break.
    """
    print(f"  loading {pkl_path} ...")
    raw = joblib.load(pkl_path)
    print(f"  loaded {len(raw)} clips")

    # Group clips under each subject and remember the first betas we saw.
    # Later clips of the same subject must match this betas exactly.
    by_subject: dict[str, dict] = {}
    for clip_idx, clip in raw.items():
        # seq_name is something like 'sub2_largetable_007'. The first
        # underscore-delimited token is always the subject ID.
        seq_name: str = clip["seq_name"]
        sub_id = seq_name.split("_")[0]

        # OMOMO stores betas as a (1, 16) row vector — flatten to (16,)
        betas = np.asarray(clip["betas"], dtype=np.float64).flatten()
        gender = str(clip.get("gender", "unknown"))

        if sub_id not in by_subject:
            by_subject[sub_id] = {
                "betas": betas,
                "gender": gender,
                "count": 0,
            }
        else:
            # All clips of the same subject MUST share the same betas, else
            # our "betas is per-subject" assumption is wrong. Bail loudly.
            if not np.allclose(by_subject[sub_id]["betas"], betas, atol=1e-6):
                diff = float(np.abs(by_subject[sub_id]["betas"] - betas).max())
                raise ValueError(
                    f"Inconsistent betas for {sub_id}: clip '{seq_name}' differs "
                    f"from earlier clips by max |diff|={diff:.6f}. The "
                    f"'betas-is-per-subject' invariant doesn't hold for this "
                    f"subject — investigate before trusting the lookup."
                )
        by_subject[sub_id]["count"] += 1

    return by_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-p", type=Path, default=DEFAULT_TRAIN_P,
                        help="Path to train_diffusion_manip_seq_joints24.p (sub1-sub15)")
    parser.add_argument("--test-p", type=Path, default=DEFAULT_TEST_P,
                        help="Path to test_diffusion_manip_seq_joints24.p (sub16, sub17)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "omomo_betas.npz",
                        help="Output npz with per-subject betas")
    args = parser.parse_args()

    # Walk both train + test pickles, merging into one subject->betas table.
    # We use both because test contains held-out subjects (sub16, sub17) which
    # are useful for the cross-body generalization story even if we don't
    # train on them.
    print("Extracting per-subject betas from OMOMO raw data:")
    all_subjects: dict[str, dict] = {}
    for tag, pkl_path in [("train", args.train_p), ("test", args.test_p)]:
        if not pkl_path.exists():
            print(f"  WARNING: {pkl_path} not found, skipping {tag}")
            continue
        print(f"[{tag}]")
        subjects = extract_betas_from_pickle(pkl_path)
        # Merge into all_subjects, asserting consistency if a subject somehow
        # appears in both train and test (shouldn't happen for OMOMO but
        # defensive — we'd rather know).
        for sub_id, info in subjects.items():
            if sub_id in all_subjects:
                if not np.allclose(all_subjects[sub_id]["betas"], info["betas"], atol=1e-6):
                    raise ValueError(
                        f"{sub_id} appears in both train and test pickles with "
                        f"different betas — investigate."
                    )
                all_subjects[sub_id]["count"] += info["count"]
            else:
                all_subjects[sub_id] = info

    # Print a tidy summary, sorted by numeric subject ID so the output is
    # easy to compare with the raw data.
    print()
    print(f"Found {len(all_subjects)} subjects total:")
    print(f"  {'sub_id':>7}  {'gender':>7}  {'count':>6}  {'|betas|':>8}  betas[:5]")
    for sub_id in sorted(all_subjects.keys(), key=lambda s: int(s[3:])):
        info = all_subjects[sub_id]
        norm = float(np.linalg.norm(info["betas"]))
        first5 = np.round(info["betas"][:5], 2)
        print(f"  {sub_id:>7}  {info['gender']:>7}  {info['count']:>6}  {norm:>8.2f}  {first5}")

    # Save to npz. We pack betas + gender into a structured layout where each
    # subject gets one (16,) array under its sub_id key, plus a separate
    # 'genders' dict-like array for metadata. Using np.savez to keep it simple
    # and importable from anywhere.
    save_dict: dict[str, np.ndarray] = {}
    genders: dict[str, str] = {}
    for sub_id, info in all_subjects.items():
        save_dict[sub_id] = info["betas"].astype(np.float32)
        genders[sub_id] = info["gender"]
    # Persist gender as a parallel npz field — we won't use it for the policy
    # obs but interact2mimic.py / SMPL_Robot may need it for MJCF generation
    # (SMPL-X has separate male/female/neutral body templates).
    save_dict["_genders"] = np.array(
        [f"{k}:{v}" for k, v in sorted(genders.items(), key=lambda kv: int(kv[0][3:]))],
        dtype=object,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save_dict)
    print()
    print(f"Saved {args.out}")
    print(f"  Keys: {[k for k in save_dict.keys() if not k.startswith('_')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate per-subject SMPL-X MJCFs for InterMimic cross-body training.

For each requested OMOMO subject, this script:
  1. Looks up that subject's SMPL-X betas + gender from omomo_betas.npz
  2. Constructs an InterAct/PHC `SMPL_Robot` parameterized by those betas
  3. Writes the resulting MJCF (a body-shape-specific humanoid robot XML)
     into InterMimic's asset directory

The output MJCFs are what InterMimic's Isaac Gym task loads to construct a
physics body matching that subject. Identical skeleton topology across all
subjects (same bodies, same DOFs) — only sizes and masses differ.

CLUSTER-ONLY: This script must run on the cluster because:
  - It imports `uhc.smpllib.smpl_local_robot` from the InterAct clone, which
    does module-level `smplx.create(MODEL_PATH, ...)` calls at import time.
    Those calls fail without the SMPL-X model files at the expected relative
    path (`../models/`).
  - The SMPL-X model files live at `/simurgh2/projects/ret-hoi/InterAct/models/`
    on the cluster, and are not present locally.

The script handles the relative-path issue by chdir-ing to
`/simurgh2/projects/ret-hoi/InterAct/simulation/` before importing, so the
hardcoded `MODEL_PATH = "../models"` inside smpl_local_robot.py resolves.

Inputs:
  scripts/omomo_betas.npz                       (from extract_omomo_betas.py)

Outputs (cluster paths, written via absolute paths):
  /simurgh2/projects/ret-hoi/InterMimic/isaacgym/src/intermimic/data/assets/smplx/omomo_sub<N>.xml

Usage on cluster:
  conda activate intermimic-gym
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cd /simurgh2/projects/ret-hoi/InterMimic
  python -u scripts/generate_per_subject_mjcfs.py --subjects sub2 sub8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# Hardcoded cluster paths per user preference (see memory:
# reference_cluster_paths). NOT $VAR placeholders.
INTERACT_ROOT = "/simurgh2/projects/ret-hoi/InterAct"
INTERMIMIC_ROOT = "/simurgh2/projects/ret-hoi/InterMimic"

# Where SMPL_Robot's write_xml output should go. InterMimic loads MJCFs from
# isaacgym/src/intermimic/data/assets/smplx/<robotType>.xml — this is the
# directory the env config's `robotType` field resolves against.
MJCF_OUTPUT_DIR = Path(INTERMIMIC_ROOT) / "isaacgym" / "src" / "intermimic" / "data" / "assets" / "smplx"


def _gender_to_int(gender_str: str) -> int:
    """Convert OMOMO's gender string to the integer code SMPL_Robot expects.

    SMPL_Robot.load_from_skeleton takes `gender=[N]` where:
        0 = neutral, 1 = male, 2 = female
    (per interact2mimic.py lines 728-734 — keep these in sync if it changes).
    """
    g = gender_str.strip().lower()
    if g == "neutral":
        return 0
    if g == "male":
        return 1
    if g == "female":
        return 2
    raise ValueError(f"Unsupported gender string: {gender_str!r}")


def generate_mjcf_for_subject(sub_id: str, betas: np.ndarray, gender: str,
                              output_path: Path, dataset_tag: str = "omomo") -> None:
    """Generate a single per-subject MJCF.

    Mirrors the relevant bits of interact2mimic.py's robot construction (lines
    ~756-783): build the same robot_cfg, instantiate SMPL_Robot, call
    load_from_skeleton with our subject's betas, then write_xml.

    Skips everything else that interact2mimic.py does (motion conversion,
    object handling, .pt writing) — we ONLY want the MJCF.
    """
    # Imports must happen AFTER chdir so module-level smplx.create() calls
    # inside smpl_local_robot.py resolve `../models` correctly. The chdir is
    # done in main() before any subject is processed; this function is called
    # only after that.
    import torch
    from uhc.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

    # Build the robot config exactly the way interact2mimic.py does for OMOMO
    # (model_type='smplx', flat_hand_mean=False per line 581). The 'beta' key
    # is what SMPL_Robot uses for its internal shape parameter; we also pass
    # it again to load_from_skeleton for symmetry with interact2mimic.py.
    robot_cfg = {
        "mesh": False,           # No vtemp mesh: use parametric capsules/boxes
        "model": "smplx",        # OMOMO is SMPL-X
        "gender": gender,        # used to pick the correct SMPL-X parser
        "upright_start": True,   # standard SMPL→MuJoCo rest pose correction
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "beta": betas.astype(np.float64),   # cfg.get(beta) → torch.from_numpy(beta[None,:])
        "flat_hand_mean": False, # OMOMO-specific (per interact2mimic.py:581)
    }

    # Model files (SMPLX_MALE.npz etc) live in MODEL_PATH/smplx, where
    # MODEL_PATH was patched into the module namespace at import-time
    # relative to the InterAct/simulation cwd. Use the same relative form
    # interact2mimic.py uses so SMPL_Robot's constructor finds them.
    data_dir = "../models/smplx"

    print(f"  [{sub_id}] constructing SMPL_Robot (gender={gender}, |betas|={np.linalg.norm(betas):.2f})...")
    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir=data_dir,
        sbj_vtemp=None,       # No subject-specific vtemp for OMOMO (only GRAB uses this)
    )

    # Drive the actual betas through the skeleton. load_from_skeleton wires
    # up the parsers, applies betas to compute body geometry/masses, and
    # builds the skeleton tree that write_xml will serialize.
    gender_number = [_gender_to_int(gender)]
    print(f"  [{sub_id}] running load_from_skeleton (gender_number={gender_number})...")
    smpl_local_robot.load_from_skeleton(
        betas=torch.from_numpy(betas.astype(np.float64)[None, :]),
        gender=gender_number,
        objs_info=None,
    )

    # write_xml accepts an absolute path; emit directly into InterMimic's
    # asset dir. The filename pattern `<model>_<dataset>_<sub>.xml` is the
    # same one interact2mimic.py uses (line 783) so anything else in the
    # pipeline that infers MJCF paths from the convention still works.
    print(f"  [{sub_id}] writing MJCF to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_local_robot.write_xml(str(output_path))

    # Sanity: confirm the file was actually written + non-trivial. Saves us
    # from the silent-failure mode we hit before with interact2mimic.py.
    if not output_path.exists():
        raise RuntimeError(f"write_xml claimed success but {output_path} doesn't exist")
    size = output_path.stat().st_size
    if size < 1000:
        raise RuntimeError(
            f"{output_path} suspiciously small ({size} bytes) — likely a malformed MJCF. "
            f"Investigate before proceeding."
        )
    print(f"  [{sub_id}] OK, {size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--betas-npz",
                        default=Path(__file__).parent / "omomo_betas.npz",
                        type=Path,
                        help="Path to omomo_betas.npz produced by extract_omomo_betas.py")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs to generate MJCFs for, e.g. sub2 sub8")
    parser.add_argument("--dataset-tag", default="omomo",
                        help="Tag inserted into MJCF filename: <model>_<tag>_<sub>.xml. Default 'omomo'.")
    parser.add_argument("--output-dir", default=MJCF_OUTPUT_DIR, type=Path,
                        help="Where to write MJCFs. Default: InterMimic's smplx asset dir on cluster.")
    parser.add_argument("--interact-root", default=INTERACT_ROOT, type=Path,
                        help="InterAct repo root (must contain simulation/ + models/). "
                             "Default: cluster path /simurgh2/projects/ret-hoi/InterAct.")
    args = parser.parse_args()

    # Chdir to InterAct/simulation BEFORE importing anything that pulls in
    # smpl_local_robot, so its module-level smplx.create() calls find the
    # SMPL-X model files at `../models/`. Also stash the simulation dir on
    # sys.path so `from uhc.smpllib.smpl_local_robot import SMPL_Robot`
    # actually resolves (uhc is not pip-installed).
    sim_dir = (args.interact_root / "simulation").resolve()
    if not sim_dir.is_dir():
        print(f"ERROR: InterAct simulation dir not found: {sim_dir}", file=sys.stderr)
        print(f"Did you run this on the cluster? SMPL-X models aren't local.", file=sys.stderr)
        return 1
    models_dir = (args.interact_root / "models" / "smplx").resolve()
    if not models_dir.is_dir():
        print(f"ERROR: SMPL-X models dir not found: {models_dir}", file=sys.stderr)
        return 1

    print(f"chdir-ing to {sim_dir} so smpl_local_robot's MODEL_PATH='../models' resolves")
    os.chdir(sim_dir)
    sys.path.insert(0, str(sim_dir))

    # Load the betas lookup. We deliberately do this AFTER chdir so we can
    # bail with a clean error if the env isn't right, before touching SMPL.
    print(f"loading betas from {args.betas_npz}")
    betas_data = np.load(args.betas_npz, allow_pickle=True)

    # The '_genders' key in the npz is a parallel array of 'sub<N>:gender'
    # strings — parse it back into a dict for gender lookup.
    genders: dict[str, str] = {}
    for entry in betas_data["_genders"]:
        sub_id, gender = str(entry).split(":", 1)
        genders[sub_id] = gender

    # Sanity-check that every requested subject is present in the npz before
    # we burn time on imports. Fail fast on typos.
    missing = [s for s in args.subjects if s not in betas_data.files or s == "_genders"]
    if missing:
        print(f"ERROR: requested subjects not in {args.betas_npz}: {missing}", file=sys.stderr)
        print(f"Available: {[k for k in betas_data.files if not k.startswith('_')]}", file=sys.stderr)
        return 1

    # Generate one MJCF per subject. If any fails, surface the error
    # immediately rather than continuing — partial output dirs are worse than
    # clear failures.
    print(f"\nGenerating MJCFs for {len(args.subjects)} subject(s):")
    for sub_id in args.subjects:
        betas = np.asarray(betas_data[sub_id])
        gender = genders.get(sub_id, "neutral")
        output_path = args.output_dir / f"smplx_{args.dataset_tag}_{sub_id}.xml"
        generate_mjcf_for_subject(sub_id, betas, gender, output_path, args.dataset_tag)

    print(f"\nAll done. MJCFs written under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

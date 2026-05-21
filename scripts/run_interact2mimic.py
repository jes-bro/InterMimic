#!/usr/bin/env python3
"""Wrapper around InterAct's simulation/interact2mimic.py.

Use this script on the cluster, in place of running interact2mimic.py directly:

    python scripts/run_interact2mimic.py \\
        --interact-root /path/to/InterAct \\
        --dataset-name behave_cari4d

By default it runs interact2mimic.py unmodified — PHC's LocalRobot emits
capsule rigs whose bone lengths are derived from the subject's SMPL-H betas
(so proportions match the CARI4D subject). This is the recommended default:
correct proportions, no rigid-body skinning artifacts.

Optional `--mesh` flag patches `"mesh": False` → `"mesh": True` in the
script's source, switching to per-bone convex-hull STLs (subject-shape
surface, but with visible seam cracks at joint rotations and convex-hull
infill — generally NOT recommended).
"""

import argparse
import os
import sys
from pathlib import Path


SOURCE_NEEDLE = '"mesh": False'
SOURCE_REPLACEMENT = '"mesh": True   # patched by run_interact2mimic.py'


# Prefix injected at the top of the exec'd interact2mimic.py source. Monkey-
# patches smplx.create() and BodyModel() so that failures (missing model
# files for SMPL-X / SMPL-H 16-beta, which our BEHAVE branch never actually
# uses) return None instead of crashing the eager module-level imports.
#
# Branches that do use those models (OMOMO, GRAB, INTERCAP, NEURALDOME, IMHD)
# would crash later if invoked — but we're not invoking them. Only behave_*
# datasets benefit from this; running another branch with missing files would
# still fail loudly, just one stack frame deeper.
SAFE_LOADERS_PREFIX = '''\
# === injected by scripts/run_interact2mimic.py ===
# Isaac Gym MUST be imported before torch (its gymdeps.py asserts this).
# smplx transitively imports torch, so load Isaac Gym first to satisfy the
# ordering requirement that interact2mimic.py itself respects.
from isaacgym import torch_utils as _ig_torch_utils_first  # noqa: F401
import smplx as _smplx_for_patch
_real_smplx_create = _smplx_for_patch.create
def _safe_smplx_create(*args, **kwargs):
    try:
        return _real_smplx_create(*args, **kwargs)
    except Exception as _e:
        print(f"[run_interact2mimic] skipping smplx.create({kwargs.get('model_type')}, {kwargs.get('gender')}): {_e!r}")
        return None
_smplx_for_patch.create = _safe_smplx_create
try:
    from human_body_prior.body_model import body_model as _body_model_for_patch
    _real_BodyModel = _body_model_for_patch.BodyModel
    class _SafeBodyModel(_real_BodyModel):
        def __new__(cls, *args, **kwargs):
            try:
                return _real_BodyModel(*args, **kwargs)
            except Exception as _e:
                print(f"[run_interact2mimic] skipping BodyModel({kwargs.get('bm_fname')}): {_e!r}")
                return None
    _body_model_for_patch.BodyModel = _SafeBodyModel
except ImportError:
    pass
# === end injected prefix ===
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interact-root", type=Path, required=True,
                        help="Path to InterAct clone (must contain simulation/interact2mimic.py).")
    parser.add_argument("--dataset-name", required=True,
                        help="Value to pass as --dataset_name to interact2mimic.py "
                             "(e.g. behave_cari4d).")
    parser.add_argument("--mesh", action="store_true",
                        help="Patch mesh=True so PHC emits per-bone convex-hull STLs. "
                             "Off by default — capsules with subject-derived bone "
                             "lengths/masses match proportions without seam artifacts.")
    parser.add_argument("--no-skip-missing-models", action="store_true",
                        help="By default, the wrapper monkey-patches smplx.create / "
                             "BodyModel so that missing model files for non-active "
                             "branches (SMPL-X, SMPL-H 16-beta) return None instead "
                             "of crashing eager imports. Pass this to disable the "
                             "patch and require ALL SMPL flavors to load.")
    args, extra = parser.parse_known_args()

    interact_root = args.interact_root.expanduser().resolve()
    sim_dir = interact_root / "simulation"
    script_path = sim_dir / "interact2mimic.py"
    if not script_path.is_file():
        print(f"[run_interact2mimic] missing {script_path}", file=sys.stderr)
        return 2

    source = script_path.read_text()
    if not args.no_skip_missing_models:
        source = SAFE_LOADERS_PREFIX + source
        print(f"[run_interact2mimic] missing-model loaders patched to return None")
    if args.mesh:
        if SOURCE_NEEDLE not in source:
            print(f"[run_interact2mimic] expected literal '{SOURCE_NEEDLE}' in "
                  f"{script_path}; refusing to run with stale assumptions.",
                  file=sys.stderr)
            return 3
        count = source.count(SOURCE_NEEDLE)
        if count != 1:
            print(f"[run_interact2mimic] expected exactly 1 occurrence of "
                  f"'{SOURCE_NEEDLE}', found {count}; refusing to patch.",
                  file=sys.stderr)
            return 3
        source = source.replace(SOURCE_NEEDLE, SOURCE_REPLACEMENT)
        print(f"[run_interact2mimic] --mesh: patched mesh=True for STL hulls")
    else:
        print(f"[run_interact2mimic] capsule mode (subject-derived bone lengths)")

    os.chdir(str(sim_dir))
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))

    sys.argv = [str(script_path), "--dataset_name", args.dataset_name, *extra]
    print(f"[run_interact2mimic] cwd={os.getcwd()}")
    print(f"[run_interact2mimic] argv={sys.argv}")

    code = compile(source, str(script_path), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

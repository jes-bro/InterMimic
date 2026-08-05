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


# The needle includes the trailing comma, and the replacement puts it back
# BEFORE the marker comment. Without the comma in both, a comment appended after
# `"mesh": True` swallows the dict's separator -- robot_cfg spans several lines
# (interact2mimic.py:758) so the entry that follows then has nothing separating
# it, and Python reports the SyntaxError at THAT key, ~28 lines off from the
# real one once the injected prefix has shifted everything.
SOURCE_NEEDLE = '"mesh": False,'
SOURCE_REPLACEMENT = '"mesh": True,  # patched by run_interact2mimic.py'

# interact2mimic.py retargets every directory under sequences_canonical with no
# way to select one (it just lists the directory, line 442). Beyond the wasted
# time, that is a correctness problem downstream: cari4d_finalize.py renames
# whatever it finds to a single --subject-id, so a second clip present here gets
# relabelled as the same subject and its MJCF overwrites the intended one.
FILTER_NEEDLE = 'data_name = sorted(os.listdir(MOTION_PATH))'
FILTER_REPLACEMENT = ('data_name = [_n for _n in sorted(os.listdir(MOTION_PATH)) '
                      'if _n in {names!r}]  # patched by run_interact2mimic.py')


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
    parser.add_argument("--only", action="append", default=None, metavar="SEQ",
                        help="Retarget only this sequence directory under "
                             "sequences_canonical, e.g. sub100_bball_000. "
                             "Repeatable. Without it every sequence present is "
                             "retargeted, which makes cari4d_finalize.py rename "
                             "them all to one subject and collide their MJCFs.")
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

    if args.only:
        # Checked here rather than trusting the injected filter to come up
        # empty: interact2mimic.py would simply retarget nothing and exit 0, and
        # a silent success that produced no output is the worst of the options.
        seq_root = interact_root / "data" / args.dataset_name.lower() / "sequences_canonical"
        present = sorted(p.name for p in seq_root.iterdir()) if seq_root.is_dir() else []
        missing = [s for s in args.only if s not in present]
        if missing:
            print(f"[run_interact2mimic] --only named {missing}, which are not in "
                  f"{seq_root}; present: {present}", file=sys.stderr)
            return 4
        if FILTER_NEEDLE not in source:
            print(f"[run_interact2mimic] expected literal '{FILTER_NEEDLE}' in "
                  f"{script_path}; refusing to run with stale assumptions.",
                  file=sys.stderr)
            return 3
        source = source.replace(
            FILTER_NEEDLE, FILTER_REPLACEMENT.format(names=set(args.only)))
        print(f"[run_interact2mimic] --only: retargeting {sorted(args.only)} "
              f"of {len(present)} sequence(s)")

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

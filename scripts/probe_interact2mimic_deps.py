#!/usr/bin/env python3
"""Probe what Python modules `interact2mimic.py` needs but can't import.

Runs in any env that has `sys` (i.e. anywhere). Installs a custom import
hook that catches every ModuleNotFoundError, stubs the missing module in
sys.modules, and continues — so Python keeps trying to import as far as
it can. At the end, prints the list of stubbed modules.

Usage on the cluster:
    cd <InterMimic>
    conda activate intermimic-gym
    python scripts/probe_interact2mimic_deps.py --interact-root <InterAct path>

It will NOT run the converter — only attempts the imports far enough to
report what's missing. Then you `pip install` everything in the list in
one shot before doing the real run.
"""

import argparse
import builtins
import os
import sys
import types
from pathlib import Path

missing = set()
_real_import = builtins.__import__


def _stub_module(name):
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as a package so submodule imports keep stubbing
    return mod


def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return _real_import(name, globals, locals, fromlist, level)
    except ModuleNotFoundError as e:
        bad = e.name or name
        missing.add(bad)
        # stub the missing module so further imports can proceed
        stub = _stub_module(bad)
        sys.modules[bad] = stub
        # if a fromlist was requested, stub each name on it too
        if fromlist:
            for attr in fromlist:
                setattr(stub, attr, types.ModuleType(f"{bad}.{attr}"))
        return stub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interact-root", required=True,
                        help="Path to InterAct clone")
    args = parser.parse_args()

    interact_root = Path(args.interact_root).expanduser().resolve()
    sim_dir = interact_root / "simulation"
    script_path = sim_dir / "interact2mimic.py"
    if not script_path.is_file():
        print(f"missing {script_path}", file=sys.stderr)
        return 2

    os.chdir(str(sim_dir))
    sys.path.insert(0, str(sim_dir))

    builtins.__import__ = _tracking_import

    # Just attempt the top-of-file imports — that's where 95% of cruft lives.
    # We don't try to run main() because that needs real data + GPU.
    try:
        with open(script_path) as f:
            source = f.read()
        # only execute up to the first 'def ' to avoid running module-level data
        # processing setup (which loads SMPL-H models eagerly)
        cutoff = source.find("def parse_npz")  # first def in the file
        if cutoff > 0:
            source = source[:cutoff]
        exec(compile(source, str(script_path), "exec"), {"__name__": "__not_main__"})
    except Exception as e:
        print(f"(probe halted at: {type(e).__name__}: {e})", file=sys.stderr)

    builtins.__import__ = _real_import

    if not missing:
        print("# Nothing missing — interact2mimic.py top-of-file imports all resolve.")
        return 0

    pip_names = sorted({_pip_name(m) for m in missing})
    print("# Missing modules:")
    for m in sorted(missing):
        print(f"#   {m}")
    print()
    print("# Install with:")
    print(f"pip install {' '.join(pip_names)}")
    return 0


def _pip_name(mod: str) -> str:
    """Map import-name → pypi-name for the few that differ."""
    return {
        "cv2": "opencv-python",
        "stl": "numpy-stl",
        "skimage": "scikit-image",
        "PIL": "Pillow",
        "human_body_prior": "human-body-prior",
        "yaml": "pyyaml",
    }.get(mod, mod)


if __name__ == "__main__":
    raise SystemExit(main())

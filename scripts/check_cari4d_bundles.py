#!/usr/bin/env python3
"""Does each downloaded CARI4D reconstruction tarball have everything the
InterMimic conversion needs?

The conversion (scripts/slurm_cari4d_to_mimic.sh -> scripts/cari4d_to_interact.py)
reads exactly three things from a bundle:

  1. output/opt/<experiment>/<clip>.pth   with a 'pr' dict holding
     smpl_pose (T,72|156), smpl_t (T,3), betas (T,10), pose_abs (T,4,4)
  2. work/<clip>/meshes-metric/<clip>_<frame>_align.obj   (the metric object mesh)
  3. the SMPL-H gender (gender.txt / meta.json)

Everything else in the tarball (viz renders, quality.json, section.json, the
glb/texture copies of the mesh) is informational. This script opens every
*-recon-*.tar.gz in a directory, extracts only those three things to a temp
dir, applies the SAME shape checks the converter applies, and prints one row
per tarball plus a verdict. It also flags the same clip exported twice, since
two exports of one clip are NOT interchangeable (different frame windows).

    python3 scripts/check_cari4d_bundles.py ~/Downloads
    python3 scripts/check_cari4d_bundles.py ~/Downloads --verbose

Exit 0 when every tarball is usable, 1 otherwise.
"""
import argparse
import glob
import json
import os
import re
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cari4d_to_interact import _load_bundle  # noqa: E402  the converter's own loader

REQUIRED_PR_KEYS = ("smpl_pose", "smpl_t", "betas", "pose_abs")


def members(tf, pattern):
    rx = re.compile(pattern)
    return [m for m in tf.getmembers() if rx.search(m.name)]


def read_text(tf, member):
    return tf.extractfile(member).read().decode()


def check_tarball(path, tmp, verbose=False):
    """Return (row dict, list of problems)."""
    problems = []
    row = {"tarball": os.path.basename(path).split("-recon-")[0]}
    with tarfile.open(path) as tf:
        # --- metadata ----------------------------------------------------
        meta_m = members(tf, r"(^|/)meta\.json$")
        gender_m = members(tf, r"(^|/)gender\.txt$")
        # Exports before 2026-09-10 have no meta.json; the clip id then comes
        # from the tarball name and the frame window is unknown. That is not a
        # defect: the converter never reads meta.json.
        meta = json.loads(read_text(tf, meta_m[0])) if meta_m else {}
        gender = read_text(tf, gender_m[0]).strip() if gender_m else meta.get("gender")
        row["clip"] = meta.get("clip") or meta.get("seq") or row["tarball"]
        row["take"] = meta.get("take", "?")
        # New schema: top-level lo/hi/n_frames. Old schema: window.chosen.
        win = meta if "lo" in meta else (meta.get("window", {}) or {}).get("chosen", {}) or {}
        row["lo_hi"] = f"{win['lo']}-{win['hi']}" if "lo" in win else "(no window)"
        row["n_meta"] = win.get("n_frames")
        row["gender"] = gender
        if gender not in ("male", "female", "neutral"):
            problems.append(f"gender '{gender}' is not male/female/neutral")
        # Older meta.json schemas carry no stage states; only a state that IS
        # recorded and is not 'done' is a problem.
        state = (meta.get("stages", {}).get("solve", {}) or {}).get("state")
        row["solve"] = state or "n/a"
        if state is not None and state != "done":
            problems.append(f"solve stage state is '{state}'")
        row["section_of"] = meta.get("section_of")

        # --- mesh --------------------------------------------------------
        # The converter wants the TOP-LEVEL <clip>_<frame>_align.obj under
        # meshes-metric (the launcher's MESH default points there).
        objs = members(tf, r"meshes-metric/[^/]+_align\.obj$")
        row["mesh"] = len(objs)
        if not objs:
            problems.append("no top-level *_align.obj under meshes-metric/")
        elif len(objs) > 1:
            problems.append(f"{len(objs)} top-level _align.obj files; the converter takes one")

        # --- the bundle --------------------------------------------------
        pths = members(tf, r"output/opt/[^/]+/[^/]+\.pth$")
        row["pth"] = len(pths)
        if len(pths) != 1:
            problems.append(f"expected exactly one output/opt/*/*.pth, found {len(pths)}")
            return row, problems
        tf.extract(pths[0], tmp)
        from pathlib import Path
        bundle = _load_bundle(Path(tmp) / pths[0].name)   # the converter's loader wants a Path
        for k in ("gt", "pr", "in"):
            if k not in bundle:
                problems.append(f"bundle missing top-level dict '{k}' (has {list(bundle)})")
        pr = bundle.get("pr", {})
        missing = [k for k in REQUIRED_PR_KEYS if k not in pr]
        if missing:
            problems.append(f"bundle['pr'] missing {missing}")
            return row, problems
        sp, st, be, pa = (pr[k].detach().cpu().numpy() for k in REQUIRED_PR_KEYS)
        T = sp.shape[0]
        row["T"] = T
        row["pose_w"] = sp.shape[1] if sp.ndim == 2 else sp.shape
        row["betas"] = be.shape[1] if be.ndim == 2 else be.shape
        # The converter's own checks, verbatim in spirit:
        if st.shape != (T, 3):
            problems.append(f"smpl_t {st.shape} != ({T}, 3)")
        if pa.shape != (T, 4, 4):
            problems.append(f"pose_abs {pa.shape} != ({T}, 4, 4)")
        if sp.ndim != 2 or sp.shape[1] not in (72, 156):
            problems.append(f"smpl_pose width {row['pose_w']} not 72/156")
        if be.ndim != 2 or be.shape[1] != 10:
            problems.append(f"betas {be.shape}: converter requires (T, 10)")
        if be.ndim == 2 and be.shape[0] != T:
            problems.append(f"betas has {be.shape[0]} rows for T={T}")
        if row["n_meta"] is not None and T != row["n_meta"]:
            problems.append(f"meta n_frames {row['n_meta']} != bundle T {T}")
        # A frozen reconstruction (all-zero pose or object) would convert
        # "successfully" and replay as a statue.
        import numpy as np
        if np.abs(sp[:, 3:]).max() < 1e-6:
            problems.append("smpl_pose body joints are all zero")
        if T > 1 and np.abs(pa[1:, :3, 3] - pa[:-1, :3, 3]).max() < 1e-9:
            problems.append("object never moves (pose_abs translation constant)")
        if not np.isfinite(sp).all() or not np.isfinite(pa).all() or not np.isfinite(st).all():
            problems.append("NaN/inf in pose, translation or object pose")
        if verbose:
            print(f"    {row['clip']}: T={T} pose{sp.shape} t{st.shape} betas{be.shape} "
                  f"pose_abs{pa.shape} | mesh {objs[0].name.split('/')[-1]}")
    return row, problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("directory", help="folder holding *-recon-*.tar.gz")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(a.directory), "*-recon-*.tar.gz")))
    if not paths:
        raise SystemExit(f"no *-recon-*.tar.gz under {a.directory}")

    rows, bad = [], 0
    by_clip = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            try:
                row, problems = check_tarball(path, tmp, a.verbose)
            except Exception as ex:  # a corrupt/partial download shows up here
                row, problems = {"tarball": os.path.basename(path)}, [f"unreadable: {ex}"]
            rows.append((row, problems))
            by_clip.setdefault(row.get("clip", row["tarball"]), []).append((path, row))
            if problems:
                bad += 1

    print(f"{'clip':<30} {'frames':>9} {'T':>4} {'pose':>5} {'betas':>5} {'gender':>6} "
          f"{'mesh':>4} {'solve':>5}  verdict")
    for row, problems in rows:
        print(f"{row.get('clip', row['tarball']):<30} {row.get('lo_hi', '?'):>9} "
              f"{str(row.get('T', '?')):>4} {str(row.get('pose_w', '?')):>5} "
              f"{str(row.get('betas', '?')):>5} {str(row.get('gender', '?')):>6} "
              f"{str(row.get('mesh', '?')):>4} {row.get('solve', '?'):>5}  "
              f"{'OK' if not problems else 'PROBLEM: ' + '; '.join(problems)}")

    dups = {c: v for c, v in by_clip.items() if len(v) > 1}
    if dups:
        print("\nSAME CLIP EXPORTED MORE THAN ONCE (not interchangeable -- pick one):")
        for c, v in dups.items():
            for path, row in v:
                print(f"  {c}: {os.path.basename(path)}  frames {row.get('lo_hi')} T={row.get('T')}")

    n = len(rows)
    print(f"\n{n - bad}/{n} tarballs usable by cari4d_to_interact.py"
          + (f", {len(dups)} clip(s) duplicated" if dups else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

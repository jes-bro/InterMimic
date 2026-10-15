#!/usr/bin/env python3
"""Count the clips an env cfg's dataSub + dataObjects filter keeps, and REFUSE
on a mismatch. The pre-flight guard of the single-object teacher launchers.

WHY. A single-object teacher is only an experiment if its data really is ONE
object. The task filters by filename (intermimic.py, _load_motion: the object is
the second-to-last '_' token of 'sub6_woodchair_012.pt'; a clip is kept only
when its target subject is in dataSub and that token is in dataObjects). A cfg
that lost dataObjects, or a copy of the dataset with stray files, would train
a 13-object generalist under a single-object arm's name and nothing in the
training log would stop it. So the launcher runs this BEFORE the GPU job:

    python3 scripts/count_motion_clips.py \\
        --cfg isaacgym/src/intermimic/data/cfg/omomo_teacher_gen4_omomo_geoall_woodchair_xf_nvadlr_nopose_msexp__f0.yaml \\
        --expect-clips 385 --expect-objects woodchair

It applies the SAME rule to the cfg's motion_file dir and exits non-zero if the
count differs from --expect-clips, or any kept clip is not one of
--expect-objects, or any expected object has no clip. It prints the per-source
breakdown either way, so the log records which sources fed the arm (every OMOMO
object is absent from 3-10 of the 13 sources; the task warns and proceeds).

The rule here is a mirror of the task's, kept in step by hand; the generator's
test (tests/test_gen4_single_object_teacher_cfgs.py) checks it against the real
OMOMO_new file list, and the task prints '[retarget] expanded N clips' at
startup for the same N, which is the runtime cross-check.
"""
import argparse
import os
import sys
from collections import Counter

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_target_subject(fname):
    """The TARGET subject number of 'sub2_largetable_007.pt' or 'sub2to8_...pt';
    None for a file the task would skip (README, non-.pt, odd name)."""
    if not fname.endswith(".pt"):
        return None
    first = fname.split("_")[0]
    if not first.startswith("sub"):
        return None
    body = first[3:]
    if "to" in body:
        body = body.split("to", 1)[1]
    return int(body) if body.isdigit() else None


def object_of(fname):
    """'sub6_woodchair_012.pt' -> 'woodchair' (intermimic.py: split('_')[-2] of the stem)."""
    return fname.rsplit(".", 1)[0].split("_")[-2]


def kept_clips(motion_dir, data_sub, data_objects):
    """[(fname, target_num, object)] the task's filter keeps, sorted like the task."""
    subs = {int(s[3:]) for s in data_sub}
    objs = set(data_objects) if data_objects else None
    out = []
    for fname in sorted(os.listdir(motion_dir)):
        tgt = parse_target_subject(fname)
        if tgt is None or tgt not in subs:
            continue
        obj = object_of(fname)
        if objs is not None and obj not in objs:
            continue
        out.append((fname, tgt, obj))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg", required=True, help="env yaml (reads motion_file, dataSub, dataObjects)")
    ap.add_argument("--motion-dir", help="override the cfg's motion_file (default: cfg value, repo-relative)")
    ap.add_argument("--expect-clips", type=int, help="refuse unless exactly this many clips are kept")
    ap.add_argument("--expect-objects", nargs="+", help="refuse unless the kept clips are exactly these objects")
    a = ap.parse_args(argv)

    env = yaml.safe_load(open(a.cfg))["env"]
    motion_dir = a.motion_dir or env["motion_file"]
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(REPO, motion_dir)
    if not os.path.isdir(motion_dir):
        print(f"[count_motion_clips] ERROR: motion dir not found: {motion_dir}", file=sys.stderr)
        return 2
    data_objects = env.get("dataObjects") or None
    clips = kept_clips(motion_dir, env["dataSub"], data_objects)

    by_src = Counter(t for _, t, _ in clips)
    by_obj = Counter(o for _, _, o in clips)
    print(f"[count_motion_clips] {os.path.basename(a.cfg)}: {len(clips)} clips kept from {motion_dir}")
    print(f"[count_motion_clips]   dataObjects = {data_objects or '<all>'}  ->  objects seen: {dict(sorted(by_obj.items()))}")
    print("[count_motion_clips]   per source: " + ", ".join(f"sub{s}={n}" for s, n in sorted(by_src.items())))
    silent = sorted({int(s[3:]) for s in env["dataSub"]} - set(by_src))
    if silent:
        print(f"[count_motion_clips]   sources with NO clip for these objects (the task warns and proceeds): "
              + " ".join(f"sub{s}" for s in silent))

    ok = True
    if a.expect_clips is not None and len(clips) != a.expect_clips:
        print(f"[count_motion_clips] ERROR: expected {a.expect_clips} clips, got {len(clips)}", file=sys.stderr)
        ok = False
    if a.expect_objects is not None:
        want = set(a.expect_objects)
        if set(by_obj) != want:
            print(f"[count_motion_clips] ERROR: expected objects {sorted(want)}, kept clips are "
                  f"{sorted(by_obj)}", file=sys.stderr)
            ok = False
    if not ok:
        print("[count_motion_clips] REFUSING: the data this cfg would load is not the arm's data.", file=sys.stderr)
        return 1
    print("[count_motion_clips] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

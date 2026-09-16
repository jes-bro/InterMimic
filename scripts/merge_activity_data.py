#!/usr/bin/env python3
"""Build the ONE data set an activity student trains on, out of several activity
teacher arms (bball7 + soccer15 + cpr13): a flat motion dir, a merged body-major
retarget tree, and the per-object physics file the env needs to mix them.

The task reads exactly ONE motion_file dir, ONE retargetedMotionDir and (until
objectPropsFile) ONE objectMass / restitution. The three activity arms each
have their own. This script, from each arm's env cfg (never typed here):

  1. links every clip of the arm's motion_file whose source is in its dataSub
     into --out-motion (absolute symlinks; a clip filename or an OBJECT NAME
     present in two arms is an error -- object names index assets and props);
  2. merges the arms' retargetedMotionDir trees into --out-retarget via
     scripts/merge_retarget_trees.py (same roster rules: every body in
     --bodies-from must exist in every tree);
  3. writes --props-out, objects: {name: {mass, restitution, arm}}, where mass
     is the arm's objectMass and restitution is solved so the ball-FLOOR
     average against the student's single plane equals the teacher's
     (utils/object_props.py: bball 0.85/0.85 -> 1.0, soccer 0.65/0.65 -> 0.6,
     cpr 0.05/0.7 -> 0.05 at plane 0.7). Also records the union dataSub so the
     student cfg can be checked against it.

    python3 scripts/merge_activity_data.py --arms bball7 soccer15 cpr13 \\
        --out-motion InterAct/behave_cari4d_act \\
        --out-retarget InterAct/behave_cari4d_act_f0_bodymajor \\
        --props-out isaacgym/src/intermimic/data/cfg/object_props_g3_act.yaml \\
        --bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_mlp_ret_stock__f0.yaml \\
        --student-plane-restitution 0.7 [--dry-run]

Run from the repo root on the machine that holds the data. Refuses to write
into a non-empty output dir.
"""
import argparse
import importlib.util
import os
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_DIR = os.path.join(REPO, "isaacgym", "src", "intermimic", "data", "cfg")
PKG = os.path.join(REPO, "isaacgym", "src", "intermimic")
ARM_CFG = "omomo_teacher_g3_{name}_geoall__f0.yaml"


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


object_props = _load_by_path(os.path.join(PKG, "utils", "object_props.py"), "object_props")
merge_trees = _load_by_path(os.path.join(REPO, "scripts", "merge_retarget_trees.py"), "merge_retarget_trees")


def read_arm(name, cfg_dir, repo):
    path = os.path.join(cfg_dir, ARM_CFG.format(name=name))
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: no env cfg for arm '{name}' at {path}")
    env = (yaml.safe_load(open(path)) or {}).get("env") or {}
    for k in ("motion_file", "retargetedMotionDir", "objectMass", "plane", "dataSub"):
        if k not in env:
            raise SystemExit(f"ERROR: {path} lacks '{k}' (every activity arm sets it)")
    return {
        "name": name,
        "motion_dir": os.path.join(repo, env["motion_file"]),
        "retarget_dir": os.path.join(repo, env["retargetedMotionDir"]),
        "mass": float(env["objectMass"]),
        "obj_r": float((env.get("objectShapeProps") or {}).get("restitution", 0.05)),
        "plane_r": float(env["plane"]["restitution"]),
        "dataSub": [str(s) for s in env["dataSub"]],
    }


def arm_clips(arm):
    d = arm["motion_dir"]
    if not os.path.isdir(d):
        raise SystemExit(f"ERROR: {arm['name']}: motion dir not found: {d}")
    subs = set(arm["dataSub"])
    keep, skipped = [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pt"):
            continue
        (keep if f.split("_")[0] in subs else skipped).append(f)
    if not keep:
        raise SystemExit(f"ERROR: {arm['name']}: no clips of dataSub {sorted(subs)} in {d}")
    return keep, skipped


def plan(arms, student_plane_r):
    """(links, props, data_sub) or SystemExit on any collision / infeasible restitution."""
    links, props, seen_clip, seen_obj = [], {}, {}, {}
    for arm in arms:
        keep, skipped = arm_clips(arm)
        if skipped:
            print(f"  {arm['name']}: skipping {len(skipped)} clip(s) outside its dataSub, e.g. {skipped[:2]}")
        r = object_props.teacher_pair_to_student_object(arm["obj_r"], arm["plane_r"], student_plane_r)
        for f in keep:
            if f in seen_clip:
                raise SystemExit(f"ERROR: clip {f} is in both {seen_clip[f]} and {arm['name']}")
            seen_clip[f] = arm["name"]
            obj = f[:-3].split("_")[-2]
            if obj in seen_obj and seen_obj[obj] != arm["name"]:
                raise SystemExit(f"ERROR: object name {obj!r} is used by both {seen_obj[obj]} and "
                                 f"{arm['name']} -- object names index assets and props")
            seen_obj[obj] = arm["name"]
            props[obj] = {"mass": arm["mass"], "restitution": round(r, 6), "arm": arm["name"]}
            links.append((os.path.join(arm["motion_dir"], f), f))
        print(f"  {arm['name']}: {len(keep)} clips, mass {arm['mass']} kg, teacher restitution "
              f"obj {arm['obj_r']} / plane {arm['plane_r']} -> student object {r:.3f}")
    data_sub = sorted({s for a in arms for s in a["dataSub"]}, key=lambda s: int(s[3:]))
    return links, props, data_sub


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, help="activity teacher arm names")
    ap.add_argument("--out-motion", required=True)
    ap.add_argument("--out-retarget", required=True)
    ap.add_argument("--props-out", required=True)
    ap.add_argument("--bodies-from", required=True, help="student env cfg (subjectBodies roster)")
    ap.add_argument("--student-plane-restitution", type=float, required=True)
    ap.add_argument("--cfg-dir", default=CFG_DIR)
    ap.add_argument("--repo", default=REPO, help="root the cfgs' relative data paths resolve against")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    arms = [read_arm(n, a.cfg_dir, a.repo) for n in a.arms]
    links, props, data_sub = plan(arms, a.student_plane_restitution)
    print(f"union dataSub ({len(data_sub)}): {data_sub}")

    for out in (a.out_motion, a.out_retarget):
        if os.path.isdir(out) and os.listdir(out):
            raise SystemExit(f"ERROR: {out} exists and is not empty -- pick a new dir")
    if os.path.exists(a.props_out):
        raise SystemExit(f"ERROR: {a.props_out} exists -- refusing to overwrite a props file")

    tree_args = ["--sources", *[x["retarget_dir"] for x in arms], "--out", a.out_retarget,
                 "--bodies-from", a.bodies_from] + (["--dry-run"] if a.dry_run else [])
    rc = merge_trees.main(tree_args)
    if rc != 0:
        raise SystemExit(f"ERROR: retarget tree merge failed (rc {rc}); nothing else written")

    if a.dry_run:
        print(f"DRY RUN: would link {len(links)} clips -> {a.out_motion}, write {len(props)} object props -> {a.props_out}")
        return links, props, data_sub
    os.makedirs(a.out_motion, exist_ok=True)
    for src, f in links:
        os.symlink(os.path.abspath(src), os.path.join(a.out_motion, f))
    with open(a.props_out, "w") as fh:
        fh.write("# GENERATED by scripts/merge_activity_data.py -- per-object mass/restitution\n"
                 "# for the activity student env (utils/object_props.py). student plane "
                 f"restitution {a.student_plane_restitution}\n")
        yaml.safe_dump({"student_plane_restitution": a.student_plane_restitution,
                        "dataSub": data_sub, "objects": props}, fh, sort_keys=True)
    print(f"linked {len(links)} clips -> {a.out_motion}; {len(props)} object props -> {a.props_out}")
    return links, props, data_sub


if __name__ == "__main__":
    main()

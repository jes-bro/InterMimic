#!/usr/bin/env python3
"""Gather the latest g3 teacher checkpoints into a teacherPolicy dir + teachers.yaml.

InterMimicDistillG3 routes each env to the teacher that owns its clip's SOURCE
subject, using <out>/teachers.yaml (utils/distill_g3.py documents the format).
This writes that dir from the fleet's checkpoint tree:

  OMOMO per-source teachers   checkpoints/smplx_teacher_g3_omomo_geoall_src{S}__f0/nn/
      EXCEPT sub2, the base arm the recipe was tuned on, which has no _src2:
                              checkpoints/smplx_teacher_g3_omomo_geoall__f0/nn/
  activity teachers           checkpoints/smplx_teacher_g3_{name}_geoall__f0/nn/
      one checkpoint serving every source in that arm's dataSub, which is READ
      FROM THE ARM'S ENV CFG (data/cfg/omomo_teacher_g3_{name}_geoall__f0.yaml),
      never typed here.

"Latest" = the highest-numbered mimic_<n>.pth snapshot, else mimic.pth (same
rule as collect_source_teachers.py). The epoch and origin path of every teacher
go into the manifest, because the student inherits whatever epoch each teacher
had reached and that must be on record.

  # the two OMOMO students
  python3 scripts/collect_g3_teachers.py --omomo-sources 1 2 3 5 6 7 8 9 11 12 14 15 17 \\
      --out checkpoints/teachers/g3_omomo
  # the two activity students
  python3 scripts/collect_g3_teachers.py --activities bball7 soccer15 cpr13 \\
      --out checkpoints/teachers/g3_act

Run from the repo root on the machine that holds the checkpoints (--root to
point elsewhere). Refuses a partial set: a missing teacher would silently drop
its sources' clips from the student, or mis-route them.
"""
import argparse
import os
import re
import shutil
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_DIR = os.path.join(REPO, "isaacgym", "src", "intermimic", "data", "cfg")

OMOMO_EXP = "smplx_teacher_g3_omomo_geoall_src{S}__f0"
# sub2 is the base arm (the recipe was tuned on it); its dir carries no _src2.
OMOMO_EXP_OVERRIDE = {2: "smplx_teacher_g3_omomo_geoall__f0"}
ACT_EXP = "smplx_teacher_g3_{name}_geoall__f0"
ACT_CFG = "omomo_teacher_g3_{name}_geoall__f0.yaml"


def latest_ckpt(nn_dir):
    """(path, epoch) of the newest numbered snapshot, else (mimic.pth, None), else (None, None)."""
    if not os.path.isdir(nn_dir):
        return None, None
    snaps = []
    for f in os.listdir(nn_dir):
        m = re.fullmatch(r"mimic_(\d+)\.pth", f)
        if m:
            snaps.append((int(m.group(1)), f))
    if snaps:
        ep, f = max(snaps)
        return os.path.join(nn_dir, f), ep
    mp = os.path.join(nn_dir, "mimic.pth")
    return (mp, None) if os.path.isfile(mp) else (None, None)


def activity_sources(name, cfg_dir):
    """The arm's dataSub as ints, from its env cfg -- the same list the teacher trained on."""
    path = os.path.join(cfg_dir, ACT_CFG.format(name=name))
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: no env cfg for activity '{name}' at {path}")
    with open(path) as fh:
        env = (yaml.safe_load(fh) or {}).get("env") or {}
    subs = env.get("dataSub")
    if not subs:
        raise SystemExit(f"ERROR: {path} has no dataSub")
    out = []
    for s in subs:
        m = re.fullmatch(r"sub(\d+)", str(s))
        if not m:
            raise SystemExit(f"ERROR: {path}: dataSub entry {s!r} is not sub<N>")
        out.append(int(m.group(1)))
    return out


def plan_teachers(root, omomo_sources, activities, cfg_dir):
    """[(file, sources, origin, epoch)] or a SystemExit listing what is missing."""
    plan, missing = [], []
    for s in omomo_sources:
        exp = OMOMO_EXP_OVERRIDE.get(s, OMOMO_EXP.format(S=s))
        ck, ep = latest_ckpt(os.path.join(root, exp, "nn"))
        if ck is None:
            missing.append(f"sub{s}: {os.path.join(root, exp, 'nn')} has no mimic*.pth")
        else:
            plan.append((f"sub{s}.pth", [s], ck, ep))
    for name in activities:
        exp = ACT_EXP.format(name=name)
        srcs = activity_sources(name, cfg_dir)
        ck, ep = latest_ckpt(os.path.join(root, exp, "nn"))
        if ck is None:
            missing.append(f"{name}: {os.path.join(root, exp, 'nn')} has no mimic*.pth")
        else:
            plan.append((f"{name}.pth", srcs, ck, ep))
    if missing:
        raise SystemExit("ERROR: missing teacher checkpoints -- refusing a partial set:\n  "
                         + "\n  ".join(missing))
    if not plan:
        raise SystemExit("ERROR: nothing requested (--omomo-sources and/or --activities)")
    # a source owned twice would be caught again by the task, but say it here first
    owner = {}
    for f, srcs, _, _ in plan:
        for s in srcs:
            if s in owner:
                raise SystemExit(f"ERROR: source sub{s} owned by both {owner[s]} and {f}")
            owner[s] = f
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omomo-sources", type=int, nargs="*", default=[], metavar="S")
    ap.add_argument("--activities", nargs="*", default=[], metavar="NAME",
                    help="activity arm names, e.g. bball7 soccer15 cpr13")
    ap.add_argument("--out", required=True, help="teacherPolicy dir to write")
    ap.add_argument("--root", default=os.path.join(REPO, "checkpoints"),
                    help="checkpoint tree (default: <repo>/checkpoints)")
    ap.add_argument("--cfg-dir", default=CFG_DIR, help="where the activity env cfgs live")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    plan = plan_teachers(a.root, a.omomo_sources, a.activities, a.cfg_dir)
    for f, srcs, ck, ep in plan:
        print(f"  {f:<14} sources {srcs}  <- {os.path.relpath(ck, a.root)}  (epoch {ep})")
    if a.dry_run:
        print(f"DRY RUN: would write {len(plan)} teachers + teachers.yaml into {a.out}")
        return plan
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"ERROR: {a.out} exists and is not empty -- pick a new dir (never overwrite a teacher set)")
    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for f, srcs, ck, ep in plan:
        shutil.copy2(ck, os.path.join(a.out, f))
        manifest.append({"file": f, "sources": srcs,
                         "from": os.path.relpath(ck, REPO) if ck.startswith(REPO) else ck,
                         "epoch": ep})
    with open(os.path.join(a.out, "teachers.yaml"), "w") as fh:
        yaml.safe_dump({"teachers": manifest}, fh, sort_keys=False)
    print(f"wrote {len(plan)} teachers + teachers.yaml -> {a.out}")
    return plan


if __name__ == "__main__":
    main()

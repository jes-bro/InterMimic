#!/usr/bin/env python3
"""Size everything one teacher arm needs, so a VM can be provisioned from facts.

Moving an arm to a cloud VM means bringing four things that are NOT in git:

  1. the arm's retarget tree      (retargetedMotionDir)   -- the big one
  2. the source clips it enumerates (motion_file, filtered by dataSub)
  3. the per-body MJCFs for every subjectBodies entry (incl. synthetic sub100+)
  4. the betas file the cfg names

plus, only if you are MIGRATING a run rather than starting fresh, its existing
checkpoints so it can resume from its own snapshots.

This measures all of them on disk and prints a per-arm total plus a recommended
boot-disk size. Run it on the machine that HAS the data (the cluster).

  python3 scripts/size_arm_payload.py --all-g3
  python3 scripts/size_arm_payload.py g3_omomo_geoall_src5__f0 g3_omomo_geoall_src3__f0
  python3 scripts/size_arm_payload.py --all-g3 --no-checkpoints   # fresh runs only

A missing path is reported as MISSING, never counted as zero -- a silently
under-reported total is how you provision a disk that fills up on day three.

NOTE ON RAM: disk size is a bad proxy for resident memory. The src1 arm OOM'd
above 192 GB while its tree is ~10 GB on disk. Get RAM from the scheduler:
    sacct -u $USER --format=JobID,JobName%32,MaxRSS,ReqMem,State
"""
import argparse
import glob
import os
import sys

import yaml

CFG_DIR = "isaacgym/src/intermimic/data/cfg"
MJCF_DIR = "isaacgym/src/intermimic/data/assets/smplx"
GB = 1024 ** 3


def dir_bytes(path):
    """Total bytes under path. Returns (bytes, n_files) or None if absent.

    Follows the tree with os.walk and skips symlinks in the size total -- the
    merged trees (src2src6 etc.) are symlink farms pointing INTO the
    single-source trees, so counting the link targets would double-count data
    you are already staging separately.
    """
    if not os.path.isdir(path):
        return None
    total, n = 0, 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):          # merged-tree symlink: not its own data
                continue
            try:
                total += os.path.getsize(p)
                n += 1
            except OSError:                # vanished mid-walk; don't crash a survey
                pass
    return total, n


def clips_bytes(motion_dir, sources):
    """Bytes of just the clips this arm enumerates: <motion_dir>/<source>_*.pt."""
    if not os.path.isdir(motion_dir):
        return None
    total, n = 0, 0
    for s in sources:
        for p in glob.glob(os.path.join(motion_dir, f"{s}_*.pt")):
            total += os.path.getsize(p)
            n += 1
    return total, n


def files_bytes(paths):
    """Bytes of an explicit file list; also returns the ones that don't exist."""
    total, missing = 0, []
    for p in paths:
        if os.path.isfile(p):
            total += os.path.getsize(p)
        else:
            missing.append(p)
    return total, missing


def experiment_name(arm):
    """full_experiment_name from the arm's TRAIN cfg -- the checkpoint dir name.

    The config name and the checkpoint dir do not always match, so read it
    rather than guessing it from the arm name.
    """
    trainc = os.path.join(CFG_DIR, "train", "rlg", f"omomo_teacher_{arm}.yaml")
    if not os.path.isfile(trainc):
        return None
    cfg = yaml.safe_load(open(trainc)) or {}
    return (((cfg.get("params") or {}).get("config") or {}).get("full_experiment_name")
            or cfg.get("full_experiment_name"))


def size_arm(arm, want_checkpoints=True):
    envc = os.path.join(CFG_DIR, f"omomo_teacher_{arm}.yaml")
    if not os.path.isfile(envc):
        return {"arm": arm, "error": f"no env cfg at {envc}"}
    env = (yaml.safe_load(open(envc)) or {}).get("env", {})

    row = {"arm": arm, "parts": {}, "missing": []}

    tree = env.get("retargetedMotionDir")
    if not tree:
        row["missing"].append("retargetedMotionDir not set in cfg")
    else:
        got = dir_bytes(tree)
        if got is None:
            row["missing"].append(f"tree MISSING: {tree}")
        else:
            row["parts"]["retarget tree"] = got

    motion_dir, sources = env.get("motion_file"), env.get("dataSub") or []
    got = clips_bytes(motion_dir, sources) if motion_dir else None
    if got is None:
        row["missing"].append(f"motion dir MISSING: {motion_dir}")
    else:
        row["parts"]["source clips"] = got

    bodies = env.get("subjectBodies") or []
    mjcfs = [os.path.join(MJCF_DIR, f"smplx_omomo_{b}.xml") for b in bodies]
    tot, miss = files_bytes(mjcfs)
    row["parts"]["MJCFs"] = (tot, len(mjcfs) - len(miss))
    if miss:
        row["missing"].append(f"{len(miss)} MJCF(s) MISSING, e.g. {miss[:3]}")

    betas = env.get("betas_file")
    if betas:
        tot, miss = files_bytes([betas])
        row["parts"]["betas"] = (tot, 0 if miss else 1)
        if miss:
            row["missing"].append(f"betas MISSING: {betas}")

    if want_checkpoints:
        exp = experiment_name(arm)
        if exp:
            got = dir_bytes(os.path.join("checkpoints", exp, "nn"))
            row["parts"]["checkpoints"] = got if got else (0, 0)
            if got is None:
                row["note"] = f"no checkpoints yet for {exp} (fresh run)"
        else:
            row["missing"].append("could not read full_experiment_name from train cfg")

    row["total"] = sum(b for b, _ in row["parts"].values())
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="arm names, e.g. g3_omomo_geoall_src5__f0")
    ap.add_argument("--all-g3", action="store_true", help="every g3 teacher env cfg")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="size a FRESH run (skip existing checkpoints)")
    ap.add_argument("--base-gb", type=float, default=18.0,
                    help="conda env + Isaac Gym + repo overhead per VM (default 18)")
    ap.add_argument("--headroom", type=float, default=1.5,
                    help="multiplier on payload for growth, incl. new checkpoints")
    a = ap.parse_args(argv)

    arms = list(a.arms)
    if a.all_g3:
        for p in sorted(glob.glob(os.path.join(CFG_DIR, "omomo_teacher_g3_*.yaml"))):
            arms.append(os.path.basename(p)[len("omomo_teacher_"):-len(".yaml")])
    if not arms:
        ap.error("pass arm names or --all-g3")

    rows = [size_arm(x, want_checkpoints=not a.no_checkpoints) for x in dict.fromkeys(arms)]

    hdr = f"{'arm':<34}{'tree':>9}{'clips':>9}{'ckpts':>9}{'other':>8}{'TOTAL':>9}{'disk':>8}"
    print(hdr); print("-" * len(hdr))
    grand = 0
    for r in rows:
        if r.get("error"):
            print(f"{r['arm']:<34}  ERROR: {r['error']}")
            continue
        g = lambda k: r["parts"].get(k, (0, 0))[0] / GB
        other = g("MJCFs") + g("betas")
        tot = r["total"] / GB
        grand += tot
        disk = tot * a.headroom + a.base_gb
        print(f"{r['arm']:<34}{g('retarget tree'):>8.1f}G{g('source clips'):>8.2f}G"
              f"{g('checkpoints'):>8.1f}G{other:>7.2f}G{tot:>8.1f}G{disk:>7.0f}G")
    print("-" * len(hdr))
    print(f"{'sum of payloads (staging bucket)':<34}{grand:>42.1f}G")

    for r in rows:
        for m in r.get("missing", []):
            print(f"  !! {r['arm']}: {m}")
        if r.get("note"):
            print(f"  -- {r['arm']}: {r['note']}")

    print("\nRAM is NOT derivable from these numbers -- src1's tree is ~10G on disk "
          "and it OOM'd above 192G resident. Size VMs with:\n"
          "  sacct -u $USER --format=JobID,JobName%32,MaxRSS,ReqMem,State")
    return 0


if __name__ == "__main__":
    sys.exit(main())

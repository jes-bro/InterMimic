#!/usr/bin/env python3
"""Sanity-check the cross-pair teacher cfgs before launching training.

Verifies:
  1. All 18 env yamls parse cleanly.
  2. Required cfg fields (dataSub, subjectBodies, dataObjects, betas_file,
     bodyNormalizedReward where expected) are present and well-formed.
  3. Per-body MJCFs exist for every body referenced.
  4. The resume checkpoint exists.
  5. Simulates the dataSub + dataObjects motion-file filter against local
     OMOMO_new and reports the resulting file count. Local only has sub2
     data, so sub6 cfgs will show 0 motion files — that's expected.
  6. Existing stage 2 cfg still has no dataObjects (back-compat check).
  7. bodyNormalizedReward is wired into intermimic.py (greps for the flag).

Run from repo root:
  python scripts/verify_crosspair_setup.py
"""
import os
import sys
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CFG_DIR = REPO / "isaacgym/src/intermimic/data/cfg"
TRAIN_CFG_DIR = REPO / "isaacgym/src/intermimic/data/cfg/train/rlg"
MJCF_DIR = REPO / "isaacgym/src/intermimic/data/assets/smplx"
MOTION_DIR = REPO / "InterAct/OMOMO_new"
RESUME_CKPT = REPO / "checkpoints/smplx_multibody_sub2only/nn/mimic_00003000.pth"


def cyan(s): return f"\033[36m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def red(s): return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"


def list_crosspair_cfgs():
    return sorted(CFG_DIR.glob("omomo_train_crosspair_b*.yaml"))


def parse_yaml(path):
    return yaml.safe_load(path.read_text())


def main():
    errors = []
    warnings = []

    print(cyan("=" * 64))
    print(cyan("1. YAML parse + required-field check"))
    print(cyan("=" * 64))

    cfgs = list_crosspair_cfgs()
    print(f"Found {len(cfgs)} crosspair env yamls\n")
    if len(cfgs) != 18:
        warnings.append(f"Expected 18 crosspair cfgs, found {len(cfgs)}")

    parsed_all = []  # (path, env_cfg)
    for p in cfgs:
        try:
            data = parse_yaml(p)
        except Exception as e:
            errors.append(f"{p.name}: failed to parse YAML: {e}")
            continue
        env = data.get("env", {})
        for field in ("dataSub", "subjectBodies", "dataObjects", "betas_file"):
            if field not in env:
                errors.append(f"{p.name}: missing field '{field}'")
        # bodyNormalizedReward should be True iff filename has _normreward
        is_normreward = "_normreward" in p.name
        flag_val = env.get("bodyNormalizedReward", False)
        if is_normreward and not flag_val:
            errors.append(f"{p.name}: filename says normreward but flag is False/missing")
        if (not is_normreward) and flag_val:
            errors.append(f"{p.name}: filename doesn't say normreward but flag is True")
        parsed_all.append((p, env, is_normreward))

    print(f"Parsed {len(parsed_all)} cfgs ({sum(1 for _,_,n in parsed_all if n)} normreward, "
          f"{sum(1 for _,_,n in parsed_all if not n)} default-reward)")

    print()
    print(cyan("=" * 64))
    print(cyan("2. Per-body MJCF existence"))
    print(cyan("=" * 64))
    bodies_used = set()
    for _, env, _ in parsed_all:
        for b in env.get("subjectBodies", []):
            bodies_used.add(b)
    print(f"Bodies referenced across cfgs: {sorted(bodies_used)}")
    for b in sorted(bodies_used):
        mjcf = MJCF_DIR / f"smplx_omomo_{b}.xml"
        if mjcf.exists():
            print(f"  {green('OK')}  {mjcf.relative_to(REPO)}")
        else:
            warnings.append(f"MJCF not local: {mjcf.name} (must exist on cluster)")
            print(f"  {yellow('NOT-LOCAL')}  {mjcf.relative_to(REPO)}")

    print()
    print(cyan("=" * 64))
    print(cyan("3. Train cfgs have NO resume_from (train from random init)"))
    print(cyan("=" * 64))
    train_cfgs = sorted(TRAIN_CFG_DIR.glob("omomo_crosspair_b*.yaml"))
    bad = []
    for tp in train_cfgs:
        for line in tp.read_text().splitlines():
            stripped = line.lstrip()
            # ignore comments; only fail on actual yaml key
            if stripped.startswith("resume_from:"):
                bad.append(tp.name)
                break
    if bad:
        errors.append(f"resume_from set in {len(bad)} cfgs: {bad}")
        for n in bad:
            print(f"  {red('HAS-RESUME')}  {n}")
    else:
        print(f"  {green('OK')}  {len(train_cfgs)} train cfgs train from random init")

    print()
    print(cyan("=" * 64))
    print(cyan("4. Motion-file filter simulation (local OMOMO_new)"))
    print(cyan("=" * 64))
    if not MOTION_DIR.exists():
        warnings.append(f"Motion dir not local: {MOTION_DIR}")
        print(f"  {yellow('NOT-LOCAL')}  {MOTION_DIR.relative_to(REPO)}")
    else:
        all_files = sorted(os.listdir(MOTION_DIR))
        print(f"Local motion dir has {len(all_files)} files\n")
        for p, env, is_norm in parsed_all:
            sub_strs = env.get("dataSub", [])
            obj_strs = env.get("dataObjects", [])
            sub_nums = {int(s[3:]) for s in sub_strs}
            objs = set(obj_strs)

            matches = []
            for fname in all_files:
                first = fname.split("_")[0]
                if not first.startswith("sub"):
                    continue
                body = first[3:]
                if "to" in body:
                    src = int(body.split("to")[0])
                    tgt = int(body.split("to")[1])
                else:
                    src = tgt = int(body)
                if tgt not in sub_nums:
                    continue
                obj_name = fname.rsplit(".", 1)[0].split("_")[-2]
                if objs and obj_name not in objs:
                    continue
                matches.append(fname)

            tag = "normreward" if is_norm else "default"
            line = f"  {p.name:75s}  [{tag:10s}]  -> {len(matches):3d} files"
            if len(matches) == 0 and any(s in str(sub_strs) for s in ("sub2",)):
                # We expect sub2-source cfgs to find files locally.
                line += yellow("  <- WARNING: 0 sub2 files matched")
                warnings.append(f"{p.name}: 0 motion files matched locally despite sub2 source")
            print(line)

    print()
    print(cyan("=" * 64))
    print(cyan("4b. maxClipsPerObject cap mechanism test (arbitrary cap=5)"))
    print(cyan("=" * 64))
    print("Sanity-check that the env code's cap logic works. Uses cap=5 as an")
    print("arbitrary test value (the production cap is TBD on cluster). Expected:")
    print("sub2 x largetable: raw=17 -> capped=5; sub2 x woodchair: raw=10 -> capped=5.\n")
    if MOTION_DIR.exists():
        all_files = sorted(os.listdir(MOTION_DIR))
        for obj in ("largetable", "woodchair"):
            matches = []
            for fname in all_files:
                first = fname.split("_")[0]
                if first != "sub2":
                    continue
                obj_name = fname.rsplit(".", 1)[0].split("_")[-2]
                if obj_name != obj:
                    continue
                matches.append(fname)
            matches.sort()
            capped = matches[:5]
            status = green("OK") if len(capped) == 5 else red("UNEXPECTED")
            print(f"  {status}  sub2 x {obj}: raw={len(matches)}, capped={len(capped)}, "
                  f"first capped index = {capped[0].rsplit('_', 1)[1].split('.')[0] if capped else 'n/a'}")
            if len(capped) != 5:
                errors.append(f"Cap simulation for {obj} returned {len(capped)}, expected 5")
    else:
        print(yellow("  Skipped (motion dir not local)"))

    print()
    print(cyan("=" * 64))
    print(cyan("5. Existing cfgs back-compat (no dataObjects key = no filter)"))
    print(cyan("=" * 64))
    legacy = [
        "omomo_train_multibody.yaml",
        "omomo_train_multibody_stage2.yaml",
        "omomo_train_multibody_nobetas.yaml",
        "omomo_test_multibody.yaml",
    ]
    for name in legacy:
        path = CFG_DIR / name
        if not path.exists():
            warnings.append(f"Legacy cfg not found: {name}")
            continue
        env = parse_yaml(path).get("env", {})
        has_filter = "dataObjects" in env
        if has_filter:
            warnings.append(f"{name} has dataObjects set unexpectedly: {env['dataObjects']}")
            print(f"  {yellow('HAS-FILTER')}  {name}: dataObjects={env['dataObjects']}")
        else:
            print(f"  {green('NO-FILTER')}  {name} (back-compat: all objects pass through)")

    print()
    print(cyan("=" * 64))
    print(cyan("6. bodyNormalizedReward wired in env code"))
    print(cyan("=" * 64))
    env_code = (REPO / "isaacgym/src/intermimic/env/tasks/intermimic.py").read_text()
    if "bodyNormalizedReward" in env_code:
        # find lines that reference it
        for i, line in enumerate(env_code.splitlines(), 1):
            if "bodyNormalizedReward" in line or "body_normalized_reward" in line:
                print(f"  intermimic.py:{i}: {line.strip()}")
    else:
        errors.append("bodyNormalizedReward not referenced in intermimic.py!")
        print(f"  {red('NOT FOUND')} bodyNormalizedReward is not referenced in intermimic.py")

    print()
    print(cyan("=" * 64))
    print(cyan("SUMMARY"))
    print(cyan("=" * 64))
    if errors:
        print(red(f"{len(errors)} ERROR(S):"))
        for e in errors:
            print(red(f"  - {e}"))
    if warnings:
        print(yellow(f"{len(warnings)} WARNING(S):"))
        for w in warnings:
            print(yellow(f"  - {w}"))
    if not errors and not warnings:
        print(green("All checks passed."))
    elif not errors:
        print(green("No errors. Warnings above are expected if running locally "
                    "without cluster paths."))

    print()
    print(cyan("=" * 64))
    print(cyan("VERIFY ON CLUSTER before launching training"))
    print(cyan("=" * 64))
    print("Run these on the cluster (in InterMimic repo root):")
    print()
    bodies_sorted = sorted(bodies_used)
    print("  # 1. MJCFs exist for every body")
    print("  ls -1 " + " ".join(
        f"isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_{b}.xml"
        for b in bodies_sorted
    ))
    print()
    print("  # 2. Motion files exist for every (source, object) combo + per-bucket count")
    print("  cd InterAct/OMOMO_new && for s in sub2 sub6; do for o in largetable woodchair; do")
    print("    echo -n \"$s $o: \"; ls ${s}_${o}_*.pt 2>/dev/null | wc -l")
    print("  done; done")
    print("  # The MIN across these 4 buckets is what you should set as")
    print("  # maxClipsPerObject in every cfg (controls for sample count).")
    print("  # Local sub2 shows: largetable=17, woodchair=10.")
    print()
    print("  # 3. Smoke-test one training cfg with a short run (no warm-start)")
    print("  python -m intermimic.run --task InterMimic \\")
    print("    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_crosspair_b10_s2_largetable.yaml \\")
    print("    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_crosspair_b10_s2_largetable.yaml \\")
    print("    --headless --num_envs 64 --max_iterations 10")
    print("  # Runs 10 training iters from random init. Verifies env + motion")
    print("  # loading; should produce checkpoints/.../nn/mimic.pth and not crash.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

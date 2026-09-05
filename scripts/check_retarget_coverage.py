#!/usr/bin/env python3
"""Does an arm's retarget directory cover the bodies you want to evaluate?

Since 2026-09-04 each arm is graded against its OWN contact-retargeted reference,
so the task needs <retargetedMotionDir>/<body>/<clip>.pt for every (body, clip)
pair it will run (intermimic.py:302-334). A missing file is a hard
FileNotFoundError -- deliberately, because the alternative is silently scoring
against the source reference, which is a different target.

Failing loud is right, but failing loud *after* sbatch costs a queue slot and a
GPU allocation. This answers the same question in a second, before you submit,
and prints the exact command to fill any gap.

    python3 scripts/check_retarget_coverage.py --arm g2_mlp_ret_stock__f0
    python3 scripts/check_retarget_coverage.py --arm g3_bball__f0 \\
        --bodies sub10 sub13 sub16

With no --bodies it checks the set eval_gen2_allbodies.sh actually submits: all
16 real OMOMO subjects except sub4 (its MJCF crashes the simulator). That set
INCLUDES the held-out bodies, which is the usual gap -- the retarget job is
normally run with --targets-from the arm's own cfg, and that cfg by construction
lists only TRAINING bodies.
"""
import argparse
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Same roster eval_gen2_allbodies.sh uses: every real OMOMO subject but sub4.
DEFAULT_BODIES = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
                  "sub10", "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
                  "sub17"]


def clips_for(motion_dir, sources):
    """The clip filenames the task will load, applying the same dataSub filter.

    intermimic.py enumerates *.pt in the motion dir and keeps those whose
    sub<N>_ prefix is in dataSub. Mirrored here so the two agree about which
    clips must exist per body.
    """
    d = os.path.join(REPO, motion_dir)
    if not os.path.isdir(d):
        return None, f"motion dir not found: {motion_dir}"
    want = {int(re.match(r"sub(\d+)", s).group(1)) for s in sources
            if re.match(r"sub(\d+)", s)}
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pt"):
            continue
        m = re.match(r"sub(\d+)_", f)
        if m and int(m.group(1)) in want:
            out.append(f)
    return out, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arm", required=True)
    p.add_argument("--bodies", nargs="+", default=DEFAULT_BODIES)
    args = p.parse_args(argv)

    import check_eval_cfg as cec
    env = cec.load(cec.resolve(args.arm))["env"]
    rt = env.get("retargetedMotionDir")
    if not rt:
        print(f"{args.arm} does not use retargeted references "
              f"(no retargetedMotionDir) -- nothing to check.")
        return 0

    # The SOURCE set comes from the arm's train cfg, not the eval cfg, whose
    # dataSub is a per-pair placeholder.
    sources = cec.load(cec.train_cfg_for(args.arm))["env"].get("dataSub") or []
    clips, err = clips_for(env["motion_file"], sources)
    if err:
        print(f"CANNOT CHECK: {err}")
        print("  (run this on the cluster, where the motion data lives)")
        return 1

    rt_abs = os.path.join(REPO, rt)
    if not os.path.isdir(rt_abs):
        print(f"CANNOT CHECK: retarget dir not found: {rt}")
        print("  (run this on the cluster, where the retargeted refs live)")
        return 1

    print(f"arm      : {args.arm}")
    print(f"retarget : {rt}")
    print(f"clips    : {len(clips)} (sources {' '.join(sources)})")
    print()

    missing_bodies = []
    for b in args.bodies:
        gone = [c for c in clips if not os.path.exists(os.path.join(rt_abs, b, c))]
        if gone:
            missing_bodies.append(b)
            print(f"  MISSING {b:<8} {len(gone)}/{len(clips)} clips  e.g. {rt}/{b}/{gone[0]}")
        else:
            print(f"  ok      {b:<8} {len(clips)}/{len(clips)}")

    if not missing_bodies:
        print(f"\nAll {len(args.bodies)} bodies covered. The eval will run.")
        return 0

    print(f"\n{len(missing_bodies)} body/bodies incomplete: {' '.join(missing_bodies)}")
    print("The eval would raise FileNotFoundError at startup rather than score them\n"
          "against the source reference. Generate the missing solves -- the job is\n"
          "ADDITIVE, it writes new per-body dirs beside the existing ones and does\n"
          "not touch bodies that are already there:\n")
    print(f"  SOURCE={sources[0] if sources else 'sub2'} \\")
    print(f"  TARGETS='{' '.join(missing_bodies)}' \\")
    print(f"  OUT={rt} \\")
    print(f"  sbatch slurm_retarget_gen.sh")
    print("\nThen re-run this check. Until it passes, either wait for the solve or\n"
          "evaluate only the covered bodies (BODIES=\"...\" sh scripts/eval_one.sh ...).")
    return 2


if __name__ == "__main__":
    sys.exit(main())

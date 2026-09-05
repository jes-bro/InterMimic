#!/usr/bin/env python3
"""Resolve an arm to its eval config, and prove that config still mirrors the arm.

Two jobs, one implementation, because both are about the same claim: *the
environment a checkpoint is scored in is the environment it was trained in,
except for the handful of keys an eval must change.*

    python3 scripts/check_eval_cfg.py --arm g3_bball__f0
        -> prints the eval cfg path (and nothing else, so shell can capture it)

    python3 scripts/check_eval_cfg.py --check-all
        -> verifies EVERY eval cfg against every arm it claims to serve

An eval cfg declares which arms it serves in a top-level `evalFor:` list. That
list lives in the file it describes rather than in a lookup table somewhere else,
so the two cannot drift apart; the task loader only reads cfg['env'] and
cfg['sim'], so an extra top-level key is inert at runtime.

WHY THIS EXISTS. Evals used to resolve their environment by a binary arch guess:
useTransformerObs set -> the 6524-dim template, else the 3230-dim one. That
template, the old shared template (omomo_test_multibody.yaml), was a chunk-1 multi-body smoke test, and it
silently supplied its OWN value for every feature added after it was written --
no retargeting, gendered betas, no free-flight gate, no obsHorizons, and the
PhysX buffer multiplier that OOM'd. Half the gen-2 grid was therefore scored
against a reference it was never trained to track, and gen-3 could not be scored
at all (a 6-horizon MLP falls to the 3230 template and dies on obs width).

The guard is the point: a key that drifts between an arm and its eval config is
an error here, not a number in a CSV.
"""
import argparse
import glob
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

# The ONLY keys an eval config may differ from its arm on. Everything else --
# obs layout, betas, retargeting, reward shape, reset thresholds, object physics,
# the free-flight gate, the PhysX buffer -- is the arm's identity and must be
# copied exactly. Each entry is a LEAF key name, matched anywhere in the tree.
#
# Keep this set as SMALL as it can be. Every name added here is a way for an eval
# to stop describing its arm without anything firing. `dataObjects` was in here
# and has been removed: no g2/g3 arm sets it and neither does any eval cfg, so it
# only ever widened the hole. It existed for the retired the old shared template (omomo_test_multibody.yaml),
# which carried a student-eval leftover ['largetable','woodchair'] that filtered
# most subjects to empty and had to be undone with a flag.
EVAL_OWNED = {
    # --- forced: the eval does not work without these ---

    # intermimic.py:169 -- enable_evaluation = enableEvaluation AND stateInit is
    # Start. In any other mode the task force-disables evaluation, so Hybrid (what
    # every arm trains in) yields no metrics at all.
    "stateInit",
    # train cfgs never set it; the metric block at intermimic.py:1677 needs it
    "enableEvaluation",
    # THE SUBTLE ONE. The episode is cut at progress - start >= rolloutLength-1
    # (humanoid.py:553) and success is _max_execution_steps >= max_episode_length-1
    # (intermimic.py:1703). A rollout window shorter than the clip cuts the episode
    # before it can ever satisfy success, so inheriting g3's 50 would report 0%
    # success for every arm no matter how good it is.
    "rolloutLength",
    # bodies are assigned round-robin across envs, so an eval that kept the arm's
    # 43-body roster would average over 43 bodies while the CSV row names one
    "subjectBodies",

    # --- placeholders and judgement calls, NOT requirements ---

    # numEnvs is eval-owned because the arm's TRAIN yaml does not state the truth
    # about it: all 30 g2/g3 launchers pass --num_envs "${NUM_ENVS:-2048}", which
    # config.py:91 uses to overwrite the 4096 sitting in those yamls. So comparing
    # eval-vs-train on this key would compare against a number that was never in
    # effect. The invariant that DOES matter is enforced separately, in
    # check_num_envs_agree(): every eval cfg must agree with every other, because
    # this is a scoring-budget knob (success = best attempt per clip) and two arms
    # scored at different budgets cannot be compared.
    "numEnvs",
    # set per pair on the command line (--data_sub); the file's value is inert
    "dataSub",
    # NOT forced. PSI harvests only where clip length >= rolloutLength, and
    # hoi_refs is topk IDENTICAL copies of the mocap reference (intermimic.py:863),
    # so at eval the buffer never diverges from it -- leaving 3 would sample among
    # identical copies. 1 is chosen for determinism and to drop a 3x allocation.
    "physicalBufferSize",
}

# NOT in the set above, deliberately: default_buffer_size_multiplier and
# cpuMotionData. Both were argued to be mere "resource knobs" -- PhysX buffer
# SIZING, and whether reference tensors live on CPU or GPU -- neither of which
# touches dynamics or observations, so one eval cfg could have covered the buf20
# and gpumotion arms too.
#
# That argument was rejected, and it is worth recording WHY: it is the identical
# reasoning that produced the bug this whole rewrite exists to fix. The retired
# the old shared template (omomo_test_multibody.yaml) supplied its own value for key after key on the
# grounds that the difference "shouldn't matter" -- and that is how half the
# gen-2 grid came to be scored against an un-retargeted reference and how the
# gen-3 arms would have been scored with their free-flight gate off.
#
# The rule is therefore: an eval config matches its arm UNLESS the eval provably
# cannot work otherwise. Nothing about evaluation requires these two to change, so
# the buf20 and gpumotion arms get their own eval configs and this set stays small.


def flatten(node, prefix=""):
    """Nested dict -> {'a.b.c': value}. Lists are leaves, compared by value."""
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# The v1 configs (omomo_eval_v1_*) are excluded from every check here. They are
# the renamed old shared template and its object-restricted siblings, kept ONLY
# for the pre-gen-2 checkpoints that genuinely trained in that environment --
# the smplx_multibody_* baselines, the distilled students, the crosspair
# teachers. They serve no `arm` in the gen-2/gen-3 sense (there is no
# omomo_teacher_<arm>.yaml to mirror), so they carry no evalFor and there is
# nothing to compare them against. Excluding them here is also what stops them
# ever being resolved for a gen-2/gen-3 arm by accident.
V1_PREFIX = "omomo_eval_v1_"


def eval_cfgs():
    """-> {path: [arms it serves]} for every per-arm eval config (v1 excluded)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(CFG, "omomo_eval_*.yaml"))):
        if os.path.basename(p).startswith(V1_PREFIX):
            continue
        cfg = load(p)
        arms = cfg.get("evalFor")
        if not arms:
            raise SystemExit(
                f"ERROR: {os.path.basename(p)} has no top-level `evalFor:` list.\n"
                f"       An eval config must name the arms it serves, or nothing "
                f"can resolve to it and nothing can check it.")
        out[p] = list(arms)
    return out


def resolve(arm):
    """arm name -> its eval config path. Zero or several matches is an error."""
    hits = [p for p, arms in eval_cfgs().items() if arm in arms]
    if not hits:
        known = sorted(a for arms in eval_cfgs().values() for a in arms)
        raise SystemExit(
            f"ERROR: no eval config serves arm '{arm}'.\n"
            f"       Refusing to fall back to a generic template -- that is how "
            f"half the gen-2 grid got scored against the wrong reference.\n"
            f"       Write cfg/omomo_eval_{arm}.yaml (mirror the arm's train cfg, "
            f"change only {sorted(EVAL_OWNED)}) and list '{arm}' in its evalFor.\n"
            f"       Arms currently served: {', '.join(known)}")
    if len(hits) > 1:
        raise SystemExit(
            f"ERROR: arm '{arm}' is claimed by {len(hits)} eval configs: "
            f"{[os.path.basename(h) for h in hits]}. Exactly one must serve it.")
    return hits[0]


def train_cfg_for(arm):
    p = os.path.join(CFG, f"omomo_teacher_{arm}.yaml")
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: no train env cfg for arm '{arm}': {p}")
    return p


def obs_width(env):
    """The obs width this config implies (intermimic.py:348-358).

    Derived, never looked up from a table of magic numbers: such a table only
    knows the two STOCK horizon sets and rejects every multi-horizon arm, which
    is all of gen-3.
    """
    arch = "transformer" if env.get("useTransformerObs") else "mlp"
    horizons = env.get("obsHorizons") or ([0, 1, 4, 16] if arch == "transformer"
                                          else [1, 16])
    betas = 32 if env.get("betas_file") else 0
    if arch == "transformer":
        return arch, horizons, betas, len(horizons) * (1599 + betas)
    return arch, horizons, betas, len(horizons) * 1599 + betas


def launcher_num_envs(arm):
    """The env count the arm's launcher PASSES -> the intended training budget.

    The train yaml is not evidence: it says numEnvs 4096 while the launcher passes
    --num_envs "${NUM_ENVS:-2048}", and config.py:91 lets the flag win. So the
    launcher is the closest thing in the repo to what the arm trained at.
    Returns (value, note); value None means it could not be established.
    """
    path = os.path.join(REPO, f"slurm_teacher_{arm}.sh")
    if not os.path.exists(path):
        return None, f"no launcher at slurm_teacher_{arm}.sh"
    src = open(path).read()
    m = re.search(r'^NUM_ENVS="\$\{NUM_ENVS:-(\d+)\}"', src, re.M)
    if not m:
        return None, f"slurm_teacher_{arm}.sh does not set a NUM_ENVS default"
    if not re.search(r"--num_envs\s+\"\$NUM_ENVS\"", src):
        return None, (f"slurm_teacher_{arm}.sh sets NUM_ENVS but never passes "
                      f"--num_envs; the yaml's numEnvs would be in effect instead")
    return int(m.group(1)), "launcher default"


def log_num_envs(arm, log_dir):
    """What ACTUALLY ran, per the arm's training logs -> {value: [files]}.

    The launcher default is only a default: `NUM_ENVS=4096 sbatch ...` overrides
    it at submission and leaves no trace in the repo. The only record is the
    launcher's own echo, `... num_envs=$NUM_ENVS`, in teacher-<arm>-<jobid>.out.
    An empty result means NOT CHECKED, and is reported as such rather than passing.
    """
    hits = {}
    for f in sorted(glob.glob(os.path.join(log_dir, f"teacher-{arm}-*.out"))):
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r"num_envs=(\d+)", text):
            hits.setdefault(int(m.group(1)), []).append(os.path.basename(f))
    return hits


def check_budget(log_dir=None):
    """-> complaints about the SCORING BUDGET, the knob no CSV would reveal.

    Success is the best attempt per CLIP -- _max_execution_steps is a running max
    indexed by seq_id over a clip-count denominator (intermimic.py:1685-1703) --
    so a bigger numEnvs can only raise the success rate and lower the pose errors.
    Three separate things therefore have to line up, and only the third is
    evidence of what happened:
      1. the eval cfgs agree with EACH OTHER   (else arms aren't comparable)
      2. that value matches the LAUNCHER default (the intended training budget)
      3. it matches what the TRAINING LOGS recorded (what actually ran)
    """
    problems = []

    # 1. eval cfgs vs each other
    seen = {}
    for path in eval_cfgs():
        n = (load(path).get("env") or {}).get("numEnvs")
        seen.setdefault(n, []).append(os.path.basename(path))
    if len(seen) > 1:
        problems.append("eval configs disagree on numEnvs -- NOT comparable:")
        for n, files in sorted(seen.items(), key=lambda kv: (kv[0] is None, kv[0])):
            problems.append(f"    numEnvs={n}: {', '.join(files)}")
        return problems                      # no single value to check 2 and 3 against
    eval_n = next(iter(seen)) if seen else None
    if eval_n is None:
        return ["eval configs do not set numEnvs at all"]

    # 2. vs each served arm's launcher
    for path, arms in eval_cfgs().items():
        for arm in arms:
            got, note = launcher_num_envs(arm)
            if got is None:
                problems.append(f"  {arm}: cannot establish the training budget "
                                f"({note}) -- NOT CHECKED, do not assume it matches")
            elif got != eval_n:
                problems.append(
                    f"  {arm}: eval scores at numEnvs={eval_n} but the arm trained "
                    f"at {got} ({note}).")

    # 3. vs what the logs say actually ran
    if log_dir is None:
        problems.append(
            "  training logs NOT CHECKED (pass --logs DIR). The launcher default "
            "can be overridden at submission with NUM_ENVS=N sbatch, which leaves "
            "no trace in the repo -- only 'num_envs=N' in teacher-<arm>-<jobid>.out.")
    else:
        for path, arms in eval_cfgs().items():
            for arm in arms:
                hits = log_num_envs(arm, log_dir)
                if not hits:
                    problems.append(f"  {arm}: no teacher-{arm}-*.out under "
                                    f"{log_dir} -- what actually ran is UNKNOWN")
                elif set(hits) != {eval_n}:
                    for n, files in sorted(hits.items()):
                        if n != eval_n:
                            problems.append(
                                f"  {arm}: a training log records num_envs={n}, not "
                                f"{eval_n} ({', '.join(files[:3])})")
    return problems


def check(eval_path, train_path):
    """-> list of complaint strings; empty means the eval config mirrors the arm."""
    ev, tr = load(eval_path), load(train_path)
    a, b = flatten(ev.get("env")), flatten(tr.get("env"))
    a.update(flatten({"sim": ev.get("sim")}))
    b.update(flatten({"sim": tr.get("sim")}))
    ABSENT = object()

    problems = []
    for key in sorted(set(a) | set(b)):
        leaf = key.split(".")[-1]
        if leaf in EVAL_OWNED:
            continue
        va, vb = a.get(key, ABSENT), b.get(key, ABSENT)
        if va != vb:
            fmt = lambda v: "<absent>" if v is ABSENT else repr(v)
            problems.append(f"  {key}: eval={fmt(va)}  arm={fmt(vb)}")

    # numObs is load-bearing and fails in two different ways -- disagreeing with
    # the arm (caught above) and disagreeing with its own horizons/betas (caught
    # here). The second is what a hand-written config gets wrong.
    for label, cfg in (("eval", ev), ("arm", tr)):
        env = cfg.get("env", {})
        arch, horizons, betas, want = obs_width(env)
        got = env.get("numObs")
        if got != want:
            problems.append(
                f"  [{label}] numObs={got} disagrees with arch={arch}, "
                f"horizons={horizons}, betas={bool(betas)} -- expected {want}")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm", help="print the eval cfg path serving this arm")
    g.add_argument("--check-all", action="store_true",
                   help="verify every eval cfg against every arm it serves")
    g.add_argument("--default-source", metavar="ARM",
                   help="print the first source subject this arm trained on, for "
                        "callers that need a sensible --data_sub default. Assuming "
                        "sub2 is wrong for the bball arm, whose only source is "
                        "sub100 -- any other value selects zero clips.")
    p.add_argument("--no-check", action="store_true",
                   help="with --arm, resolve only; skip the mirror check")
    p.add_argument("--logs", metavar="DIR",
                   help="directory of teacher-<arm>-<jobid>.out training logs, so "
                        "the scoring budget is checked against what ACTUALLY ran "
                        "rather than against the launcher's default. Without it "
                        "that check is reported as NOT CHECKED, never as passing.")
    args = p.parse_args(argv)

    if args.default_source:
        env = load(train_cfg_for(args.default_source)).get("env", {})
        src = env.get("dataSub") or []
        if not src:
            print(f"ERROR: {args.default_source} has no dataSub to take a default "
                  f"source from", file=sys.stderr)
            return 2
        print(src[0])
        return 0

    if args.arm:
        path = resolve(args.arm)
        if not args.no_check:
            problems = check(path, train_cfg_for(args.arm))
            if problems:
                print(f"ERROR: {os.path.basename(path)} no longer mirrors "
                      f"omomo_teacher_{args.arm}.yaml:", file=sys.stderr)
                print("\n".join(problems), file=sys.stderr)
                print(f"  (an eval config may differ from its arm ONLY in "
                      f"{sorted(EVAL_OWNED)})", file=sys.stderr)
                return 2
        print(path)
        return 0

    rc = 0
    for path, arms in eval_cfgs().items():
        for arm in arms:
            problems = check(path, train_cfg_for(arm))
            tag = f"{os.path.basename(path)} vs {arm}"
            if problems:
                rc = 2
                print(f"FAIL {tag}")
                print("\n".join(problems))
            else:
                print(f"ok   {tag}")

    # The scoring budget is checked separately because it is the one setting the
    # train yaml cannot testify about (the launcher's --num_envs overrides it) and
    # the one no CSV would ever reveal.
    print("\n-- scoring budget (numEnvs) --")
    budget = check_budget(args.logs)
    if budget:
        print("\n".join(budget))
        # An unverifiable budget is not a failure of the configs, but it must not
        # read as a pass either -- say so and let the caller decide.
        if any("NOT CHECKED" not in p and "UNKNOWN" not in p for p in budget):
            rc = 2
    else:
        print("ok   every eval cfg, launcher and training log agrees")
    return rc


if __name__ == "__main__":
    sys.exit(main())

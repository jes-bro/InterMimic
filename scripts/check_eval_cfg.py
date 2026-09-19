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

    # numEnvs is eval-owned because it is not part of the arm's identity at all:
    # it sets how many rollouts run CONCURRENTLY while scoring, nothing else. The
    # attempt budget is the player's episode count (see check_budget). The
    # invariant that matters is enforced there: every eval cfg uses the same
    # value, and that value is EVAL_NUM_ENVS (InterMimic's own eval scripts).
    "numEnvs",
    # set per pair on the command line (--data_sub); the file's value is inert
    "dataSub",
    # NOT forced. PSI harvests only where clip length >= rolloutLength, and
    # hoi_refs is topk IDENTICAL copies of the mocap reference (intermimic.py:863),
    # so at eval the buffer never diverges from it -- leaving 3 would sample among
    # identical copies. 1 is chosen for determinism and to drop a 3x allocation.
    "physicalBufferSize",
    # Arm A's twin envs exist only for the contrastive loss, which the player
    # never computes. At eval subjectBodies is ONE body per pair, and
    # body_features.twin_partners refuses to pair two envs that share a body
    # (every pair would), so the task cannot even be constructed with twins on.
    # Off at eval; the scored rollouts are identical either way (2026-09-19).
    "twinEnvs",
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

# SCORING VARIANTS. An eval id is normally just an arm name and its config must
# mirror that arm. `<arm>+<variant>` names a second scoring of the SAME
# checkpoint under a deliberately different rule -- the one case so far is
# `g3_bball7_geoall_nogate__f0+gatedscore`: the nogate policy scored under the
# base's termination rule, because mirroring `freeFlightGate.resets: false`
# into the eval makes the referee, not the policy, end every free-flight
# episode. A variant config carries a top-level `scoringVariant:` block naming
# the arm, the variant, and EXACTLY which keys differ (and to what); the check
# allows those keys and nothing else, and refuses an override that does not
# actually differ from the arm (a no-op variant is a mislabelled duplicate).
VARIANT_SEP = "+"


def split_variant(spec):
    """'g3_x__f0+gatedscore' -> ('g3_x__f0', 'gatedscore'); 'g3_x__f0' -> ('g3_x__f0', None)."""
    if VARIANT_SEP in spec:
        arm, variant = spec.split(VARIANT_SEP, 1)
        if not arm or not variant:
            raise SystemExit(f"ERROR: malformed eval id {spec!r} (want <arm>{VARIANT_SEP}<variant>)")
        return arm, variant
    return spec, None


def variant_block(cfg, path):
    """The validated `scoringVariant:` block of a variant cfg, or None for a plain one."""
    sv = cfg.get("scoringVariant")
    if sv is None:
        return None
    name = os.path.basename(path)
    for key in ("name", "arm", "overrides"):
        if key not in sv:
            raise SystemExit(f"ERROR: {name}: scoringVariant lacks `{key}:`")
    if not isinstance(sv["overrides"], dict) or not sv["overrides"]:
        raise SystemExit(f"ERROR: {name}: scoringVariant.overrides must be a non-empty "
                         f"mapping of dotted env keys to the value this eval uses")
    return sv


def eval_cfgs():
    """-> {path: [eval ids it serves]} for every per-arm eval config (v1 excluded).

    An id is an arm name, or `<arm>+<variant>` for a scoring variant. A variant
    cfg may serve only variant ids of its own arm, and a plain cfg no variant
    ids -- so a variant can never be resolved for the arm by accident.
    """
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
        sv = variant_block(cfg, p)
        for spec in arms:
            arm, variant = split_variant(spec)
            if sv is None and variant is not None:
                raise SystemExit(f"ERROR: {os.path.basename(p)} serves variant id {spec!r} "
                                 f"but has no scoringVariant block")
            if sv is not None and (variant != sv["name"] or arm != sv["arm"]):
                raise SystemExit(f"ERROR: {os.path.basename(p)} is scoringVariant "
                                 f"{sv['arm']}{VARIANT_SEP}{sv['name']} but serves {spec!r}")
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


def is_student(arm):
    """g3 students are named by their cfg stem `student_g3_<set>_<student>__f0`
    (cfg omomo_student_..., launcher slurm_student_..., log student-...,
    checkpoints smplx_student_...); teachers by the bare arm behind omomo_teacher_."""
    return arm.startswith("student_")


def train_cfg_for(spec):
    """Train env cfg of the arm behind an eval id (variant suffix ignored)."""
    arm, _ = split_variant(spec)
    p = os.path.join(CFG, f"omomo_{arm}.yaml" if is_student(arm) else f"omomo_teacher_{arm}.yaml")
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: no train env cfg for arm '{arm}': {p}")
    return p


def launcher_for(arm):
    return f"slurm_{arm}.sh" if is_student(arm) else f"slurm_teacher_{arm}.sh"


def log_glob_for(arm):
    # launchers write teacher-<arm>-<jobid>.out / student-g3_...-<jobid>.out
    return f"{arm.replace('student_', 'student-', 1)}-*.out" if is_student(arm) else f"teacher-{arm}-*.out"


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


# The env count every eval scores at. InterMimic's own evaluation scripts
# (isaacgym/scripts/eval_teacher.sh, eval_student.sh) pass --num_envs 1024; we
# match them. Until 2026-09-18 this was 2048 on the belief that it was the
# attempt budget -- it is not (see check_budget), so those CSVs stay comparable.
EVAL_NUM_ENVS = 1024


def train_rlg_for(spec):
    """The arm's rl_games train cfg (carries the player block, if any)."""
    arm, _ = split_variant(spec)
    return os.path.join(CFG, "train", "rlg",
                        f"omomo_{arm}.yaml" if is_student(arm) else f"omomo_teacher_{arm}.yaml")


def check_budget():
    """-> complaints about the SCORING BUDGET, the knob no CSV would reveal.

    Success is the best attempt per CLIP (_max_execution_steps is a running max
    indexed by seq_id, intermimic.py), so the number of attempts a clip gets
    decides the score. That number is NOT numEnvs. The player runs
    games_num * n_game_life * 10 episodes per (body, source) pair and stops on
    that count (learning/intermimic_players.py:52-60, 173, 363, 392); with
    rl_games' defaults (common/player.py:41-43: games_num 2000, n_game_life 1)
    that is 20,000 episodes whether 1024 or 2048 envs run them. numEnvs is
    concurrency. So three things must hold:
      1. every eval cfg uses the same numEnvs        (one exam for every arm)
      2. that value is EVAL_NUM_ENVS                  (InterMimic's own scripts)
      3. no served arm's train cfg overrides the player's games_num /
         n_game_life                                  (else its budget differs)
    A `--logs` comparison against the TRAINING env count existed until
    2026-09-18; the training count is not the scoring budget, so it was dropped.
    """
    problems = []

    # 1 + 2. eval cfgs vs each other, and vs the fleet value
    seen = {}
    for path in eval_cfgs():
        n = (load(path).get("env") or {}).get("numEnvs")
        seen.setdefault(n, []).append(os.path.basename(path))
    if len(seen) > 1:
        problems.append("eval configs disagree on numEnvs -- NOT comparable:")
        for n, files in sorted(seen.items(), key=lambda kv: (kv[0] is None, kv[0])):
            problems.append(f"    numEnvs={n}: {', '.join(files)}")
        return problems
    eval_n = next(iter(seen)) if seen else None
    if eval_n is None:
        return ["eval configs do not set numEnvs at all"]
    if eval_n != EVAL_NUM_ENVS:
        problems.append(f"  every eval cfg says numEnvs={eval_n}, but the fleet value is "
                        f"{EVAL_NUM_ENVS} (InterMimic's isaacgym/scripts/eval_*.sh)")

    # 3. the player's episode budget must be the default for every served arm
    for path, arms in eval_cfgs().items():
        for spec in arms:
            rlg = train_rlg_for(spec)
            if not os.path.exists(rlg):
                problems.append(f"  {spec}: no train rlg cfg at {os.path.basename(rlg)} -- "
                                f"cannot confirm the player budget is the default")
                continue
            player = ((load(rlg).get("params") or {}).get("config") or {}).get("player") or {}
            bad = {k: player[k] for k in ("games_num", "n_game_life") if k in player}
            if bad:
                problems.append(f"  {spec}: train cfg overrides the player episode budget "
                                f"{bad} -- its attempts per clip differ from every other arm")
    return problems


def check(eval_path, train_path, spec=None):
    """-> list of complaint strings; empty means the eval config mirrors the arm.

    `spec` is the eval id being checked. For a plain arm id the config may differ
    from the arm only in EVAL_OWNED keys. For `<arm>+<variant>` the config's
    scoringVariant.overrides are ALSO allowed -- and each must be present in the
    eval at the declared value and differ from the arm, or the variant is a lie.
    """
    ev, tr = load(eval_path), load(train_path)
    a, b = flatten(ev.get("env")), flatten(tr.get("env"))
    a.update(flatten({"sim": ev.get("sim")}))
    b.update(flatten({"sim": tr.get("sim")}))
    ABSENT = object()
    fmt = lambda v: "<absent>" if v is ABSENT else repr(v)

    problems = []
    _, variant = split_variant(spec) if spec else (None, None)
    sv = variant_block(ev, eval_path)
    overrides = {}
    if variant is None and sv is not None:
        problems.append(f"  config carries a scoringVariant block ({sv['name']}) but is "
                        f"being checked as the plain eval of its arm")
    elif variant is not None and sv is None:
        problems.append(f"  eval id names variant '{variant}' but the config has no "
                        f"scoringVariant block declaring what differs")
    elif variant is not None:
        overrides = dict(sv["overrides"])
        for key, want in overrides.items():
            va, vb = a.get(key, ABSENT), b.get(key, ABSENT)
            if va != want:
                problems.append(f"  override {key}: declared {want!r} but the eval "
                                f"has {fmt(va)}")
            if vb == want:
                problems.append(f"  override {key}: arm already has {want!r} -- this "
                                f"variant changes nothing and must not exist")

    for key in sorted(set(a) | set(b)):
        leaf = key.split(".")[-1]
        if leaf in EVAL_OWNED or key in overrides:
            continue
        va, vb = a.get(key, ABSENT), b.get(key, ABSENT)
        if va != vb:
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
    g.add_argument("--arm", help="print the eval cfg path serving this eval id "
                                 "(an arm, or <arm>+<variant> for a scoring variant)")
    g.add_argument("--check-all", action="store_true",
                   help="verify every eval cfg against every arm it serves")
    g.add_argument("--default-source", metavar="ARM",
                   help="print the first source subject this arm trained on, for "
                        "callers that need a sensible --data_sub default. Assuming "
                        "sub2 is wrong for the bball arm, whose only source is "
                        "sub100 -- any other value selects zero clips.")
    p.add_argument("--no-check", action="store_true",
                   help="with --arm, resolve only; skip the mirror check")
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
            problems = check(path, train_cfg_for(args.arm), args.arm)
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
            problems = check(path, train_cfg_for(arm), arm)
            tag = f"{os.path.basename(path)} vs {arm}"
            if problems:
                rc = 2
                print(f"FAIL {tag}")
                print("\n".join(problems))
            else:
                print(f"ok   {tag}")

    # The scoring budget is checked separately because no CSV would ever reveal it.
    print("\n-- scoring budget (numEnvs uniform + fleet value; player episode count default) --")
    budget = check_budget()
    if budget:
        print("\n".join(budget))
        rc = 2
    else:
        print(f"ok   every eval cfg scores at numEnvs={EVAL_NUM_ENVS} with the default "
              f"20,000-episode player budget")
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Staged SOURCE-TEACHER: one FIXED source's motion, folding BODIES in over stages.

curriculum_runner.py folds whole SUBJECTS in as both source and target (one
policy learning every source x every body). A source-teacher is the opposite
shape: a SINGLE fixed source S drives a big set of bodies. This driver stages
THAT -- it holds dataSub = [subS] constant and grows the live BODY set:

  stage 0 : all 13 real train bodies + every 'inhull' synthetic body  (live)
  stage k : + one 'extrapolated' synthetic body (cumulative), resuming the
            previous stage's checkpoint.

Rationale (see the reward analysis): the reward is a multiplicative AND-gate and
the mean is averaged over ~52 bodies, so dumping the hard, off-distribution
EXTRAPOLATED synthetic bodies in at step 0 flattens the early curve. The inhull
synthetics sit within the real shape range (per analyze_synthetic_bodies.py), so
they ride along with the reals from the start; only the extrapolated tail is
folded one at a time, from a policy that's already competent.

Mechanism: the env's subjectPairWeightsFile (JSON "b{B}_s{S}" -> weight, 0 masks
a pair) plus maskDeadEnvs. Not-yet-live bodies still run but contribute NO
gradient. subjectBodies always holds the FULL set; only the weight mask changes
per stage. Everything else is byte-identical to omomo_teacher_src{S}_xf_aug so
this is a clean A/B against the no-stage run.

Run (inside one slurm job, one GPU):
  python scripts/staged_source_teacher_runner.py --source 2 --run-name src2_staged
Validate config/weight generation WITHOUT launching training:
  python scripts/staged_source_teacher_runner.py --source 2 --dry-run
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Reuse the proven templates + launch/plateau machinery so the staged run is
# format-identical to the curriculum runner (and, via matching params, to the
# no-stage teacher config).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_runner import (  # noqa: E402
    REPO, ENV_TMPL, TRAIN_TMPL, compute_pair_weights, run_stage,
    latest_checkpoint, yaml_sub_list,
)

CFG = REPO / "isaacgym/src/intermimic/data/cfg"
HELD_OUT = {4, 10, 13, 16}   # never a body; sanity-checked below


def rel(p):
    """Path relative to repo root (configs are referenced from the repo root)."""
    return os.path.relpath(str(p), str(REPO))


def load_base_bodies(source):
    """subjectBodies from the existing omomo_teacher_src{S}_xf_aug.yaml.

    Reading the no-stage config (rather than re-deriving) guarantees the staged
    arm trains on the EXACT same body set -- including the sub121 removal.
    """
    base = CFG / f"omomo_teacher_src{source}_xf_aug.yaml"
    if not base.is_file():
        sys.exit(f"ERROR: base config not found: {base}\n"
                 f"       (need the no-stage src{source} teacher to mirror its body set)")
    env = yaml.safe_load(open(base))["env"]
    return [int(b[3:]) for b in env["subjectBodies"]], rel(base)


def synthetic_kinds():
    """subject-num -> 'inhull' | 'extrap' from the generator's tag file."""
    z = np.load(REPO / "scripts/synthetic_bodies_neutral.npz", allow_pickle=True)
    keys = [k for k in z.files if k.startswith("sub") and k[3:].isdigit()]
    return {int(k[3:]): str(t) for k, t in zip(keys, z["_kinds"])}


def build_stages(source, bodies, kinds):
    """Fixed-source body-fold stage plan.

    Returns a list of dicts: {stage, exp, live_bodies(set), new(int|None)}.
    live_bodies grows: stage 0 = reals+inhull, then one extrap body per stage.
    """
    reals  = sorted(b for b in bodies if b < 100)
    syn    = [b for b in bodies if b >= 100]
    inhull = sorted(b for b in syn if kinds.get(b) == "inhull")
    extrap = sorted(b for b in syn if kinds.get(b) == "extrap")
    unknown = [b for b in syn if b not in kinds]
    if unknown:
        sys.exit(f"ERROR: synthetic bodies with no inhull/extrap tag: {unknown} "
                 f"(regenerate scripts/synthetic_bodies_neutral.npz)")

    stages, live = [], set()
    live |= {b for b in reals + inhull}
    stages.append(dict(stage=0, exp=f"smplx_teacher_src{source}_staged_s00",
                       live_bodies=set(live), new=None,
                       desc=f"{len(reals)} real + {len(inhull)} inhull (base)"))
    for i, e in enumerate(extrap, start=1):
        live = set(live); live.add(e)
        stages.append(dict(stage=i, exp=f"smplx_teacher_src{source}_staged_s{i:02d}",
                           live_bodies=set(live), new=e,
                           desc=f"+ extrap sub{e}"))
    return stages, reals, inhull, extrap


def write_stage_cfgs(source, bodies, stage, work, resume_from):
    """Emit the per-stage env + train YAML and the weight-mask JSON. Returns paths."""
    S = source
    # weights: 1.0 for live (S, b) pairs, explicit 0.0 for masked -- the env
    # REQUIRES a key for every (body, source) pair or it hard-errors.
    live_pairs = {(S, b) for b in stage["live_bodies"]}
    weights = compute_pair_weights(bodies=bodies, sources=[S],
                                   live=live_pairs, exposure={})
    # sanity: a key must exist for every body (source is the single fixed S)
    missing = [b for b in bodies if f"b{b}_s{S}" not in weights]
    if missing:
        sys.exit(f"ERROR: stage {stage['stage']} weights missing bodies {missing}")

    wpath = work / f"weights_s{stage['stage']:02d}.json"
    wpath.write_text(json.dumps(weights, indent=2, sort_keys=True))

    env_path   = work / f"env_s{stage['stage']:02d}.yaml"
    train_path = work / f"train_s{stage['stage']:02d}.yaml"
    active = f"src{S} -> {stage['desc']} ({int(sum(v>0 for v in weights.values()))} live pairs)"

    env_path.write_text(ENV_TMPL.format(
        stage=f"{stage['stage']:02d}", active=active, num_envs=4096,
        datasub=yaml_sub_list([S]), bodies=yaml_sub_list(bodies),
        pair_weights_line=f"  subjectPairWeightsFile: {rel(wpath)}",
        counts_line="  # pairSampleCountsFile omitted (mask-only staging)",
        mask_dead_envs="True", num_obs=6524,
        betas_file="scripts/omomo_betas_neutral_aug.npz", body_norm="true",
        heights_line="  subjectHeightsFile: scripts/synthetic_heights.json",
        use_transformer_obs="true", cpu_motion="false",
        pose_term_block="  rewardTerms:\n    pose:\n      enable: true\n      lambda: 0.02"))

    train_path.write_text(TRAIN_TMPL.format(
        stage=f"{stage['stage']:02d}", active=active, exp_name=stage["exp"],
        save_frequency=100, resume_from=resume_from,
        mask_dead_envs="True", network_name="intermimic_transformer"))
    return rel(env_path), rel(train_path), wpath, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=int, required=True, help="fixed motion source, e.g. 2")
    ap.add_argument("--run-name", default=None, help="work-dir name (default src{S}_staged)")
    ap.add_argument("--min-epochs", type=int, default=800,
                    help="min epochs in a stage before a plateau can advance it")
    ap.add_argument("--patience", type=int, default=300,
                    help="advance if no >improve-frac gain for this many epochs")
    ap.add_argument("--improve-frac", type=float, default=0.01)
    ap.add_argument("--stage-max-epochs", type=int, default=4000,
                    help="hard cap per stage (base stage 0 gets 2x this)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write all stage cfgs+weights and validate; do NOT train")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages that already have a checkpoint (survives a "
                         "requeue mid-curriculum; assumes an existing stage ckpt "
                         "means that stage completed)")
    a = ap.parse_args()

    S = a.source
    if S in HELD_OUT:
        print(f"[staged] NOTE: source sub{S} is a held-out body; it's still a valid "
              f"MOTION source (held-out applies to target bodies).", flush=True)
    run_name = a.run_name or f"src{S}_staged"
    work = REPO / "staged_work" / run_name
    work.mkdir(parents=True, exist_ok=True)

    bodies, base_rel = load_base_bodies(S)
    kinds = synthetic_kinds()
    stages, reals, inhull, extrap = build_stages(S, bodies, kinds)

    print(f"[staged] source=sub{S}  bodies={len(bodies)} "
          f"({len(reals)} real + {len(inhull)} inhull + {len(extrap)} extrap)")
    print(f"[staged] base config: {base_rel}")
    print(f"[staged] {len(stages)} stages: stage0 = reals+inhull, "
          f"then +1 extrapolated body/stage over {len(extrap)} stages")
    print(f"[staged] extrap fold-in order: {['sub%d' % e for e in extrap]}")

    for idx, st in enumerate(stages):
        # resume: stage 0 cold; later stages warm-start from the prior stage's ckpt
        if idx == 0:
            resume_from = "'None'"
        else:
            prev_exp = stages[idx - 1]["exp"]
            ckpt = latest_checkpoint(prev_exp)
            resume_from = rel(ckpt) if ckpt else f"checkpoints/{prev_exp}/nn/mimic.pth"

        env_cfg, train_cfg, wpath, weights = write_stage_cfgs(S, bodies, st, work, resume_from)
        n_live = int(sum(v > 0 for v in weights.values()))
        print(f"\n[staged] === stage {st['stage']:02d}  exp={st['exp']} ===")
        print(f"[staged]   {st['desc']}  ({n_live}/{len(bodies)} bodies live)")
        print(f"[staged]   env={env_cfg} train={train_cfg} weights={rel(wpath)}")
        print(f"[staged]   resume_from={resume_from}")

        if a.dry_run:
            continue

        # requeue support: a stage that already has a checkpoint is treated as
        # done, so a restarted job continues from the first unfinished stage
        # instead of redoing the whole curriculum from scratch.
        if a.resume and latest_checkpoint(st["exp"]) is not None:
            print(f"[staged]   --resume: checkpoint exists, skipping stage {st['stage']:02d}")
            continue

        # base stage 0 deserves a bigger budget than each small extrap add-on
        stage_max = a.stage_max_epochs * (2 if idx == 0 else 1)
        log_path = work / f"stage_s{st['stage']:02d}.log"
        epochs, exited = run_stage(env_cfg, train_cfg, st["exp"], log_path, a,
                                   a.min_epochs, a.patience, stage_max)
        ckpt = latest_checkpoint(st["exp"])
        print(f"[staged] stage {st['stage']:02d} done: {epochs} epochs, ckpt={ckpt}")
        if ckpt is None:
            sys.exit(f"ERROR: stage {st['stage']:02d} produced no checkpoint "
                     f"(training failed?) -- stopping so the next stage can't "
                     f"resume from nothing.")

    if a.dry_run:
        print(f"\n[staged] DRY RUN ok: {len(stages)} stages written to {rel(work)} "
              f"(no training launched). Final policy would be "
              f"checkpoints/{stages[-1]['exp']}/nn/.")
    else:
        print(f"\n[staged] curriculum complete. Final policy: "
              f"checkpoints/{stages[-1]['exp']}/nn/  (eval this one)")


if __name__ == "__main__":
    main()

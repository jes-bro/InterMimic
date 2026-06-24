#!/usr/bin/env python3
"""objectAug curriculum runner: widen object perturbation across warm-started stages.

InterMimic / Isaac Gym can't rescale collision geometry in a live sim (scale is
baked at env creation, before prepare_sim). So the object-diversity curriculum
is a SEQUENCE of stages: each stage re-bakes the envs with a WIDER objectAug
range and warm-starts the previous stage's checkpoint. This sidesteps the static
sim entirely -- every stage is a fresh process.

The reward COMBO (objterms + pose + hold) is FIXED for the whole run -- pick it
with --variant (one of the 8 omomo_objectaug_* combos); only the perturbation
ranges widen across stages. The plateau-advance + state.json resume + warm-start
machinery is reused from curriculum_runner (run_stage / latest_checkpoint), and
the per-stage env YAML is built by generate_objectaug_cfgs.render() so configs
match the standalone files exactly.

Run inside one slurm job / one GPU:
  python scripts/objectaug_runner.py --run-name oa_dropboth --variant drop_both \\
      --init-checkpoint checkpoints/<foldin_policy>/nn/mimic.pth
Resume after a requeue (continues from state.json):
  python scripts/objectaug_runner.py --run-name oa_dropboth --variant drop_both \\
      --init-checkpoint <same> --resume
Dry-run (write every stage's configs, launch nothing -- safe without Isaac Gym):
  python scripts/objectaug_runner.py --run-name oa_dropboth --variant drop_both \\
      --init-checkpoint <any> --dry-run

Edit DEFAULT_LADDER to change the stages (e.g. a milder ladder for 'keep_*'
runs, where the stock object terms tolerate only small perturbation).
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# scripts/ on sys.path so the sibling modules import cleanly.
sys.path.insert(0, str(REPO / "scripts"))
import generate_objectaug_cfgs as gen                       # render(): the env YAML
from curriculum_runner import run_stage, latest_checkpoint, TRAIN_TMPL  # reused machinery

# Widening ladder: (scaleMin, scaleMax, yawRad, translateM). Each stage adds
# diversity; warm-starting carries the policy forward. Stage 0 is ~identity, a
# cheap warm-start sanity (reward should sit near the fold-in baseline before
# any real perturbation). Trim/extend freely.
DEFAULT_LADDER = [
    (1.00, 1.00, 0.00, 0.00),   # 0: identity  -- warm-start sanity
    (0.95, 1.05, 0.17, 0.05),   # 1: gentle    (~10 deg, 5 cm)
    (0.90, 1.10, 0.35, 0.08),   # 2: medium    (~20 deg, 8 cm)
    (0.80, 1.25, 0.52, 0.10),   # 3: full      (~30 deg, 10 cm)
]

# The 8 named variants -> (objectTermsEnable, pose_enable, hold_enable), as YAML
# literals, matching generate_objectaug_cfgs.
VARIANTS = {
    f"{o_slug}_{t_slug}": (o_val, p_val, h_val)
    for o_slug, o_val in (("drop", "false"), ("keep", "true"))
    for t_slug, p_val, h_val in (("base", "false", "false"), ("pose", "true", "false"),
                                 ("hold", "false", "true"), ("both", "true", "true"))
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True, help="names the work/output dirs")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS),
                    help="which reward combo to hold fixed across the ladder")
    ap.add_argument("--init-checkpoint", required=True,
                    help="fold-in policy .pth to warm-start stage 0 from")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=300,
                    help="advance after this many epochs with no >improve-frac gain")
    ap.add_argument("--min-epochs", type=int, default=300,
                    help="never advance before this many epochs in a stage")
    ap.add_argument("--improve-frac", type=float, default=0.01,
                    help="fractional reward gain that counts as improvement")
    ap.add_argument("--stage-max-epochs", type=int, default=8000,
                    help="hard cap per stage if it never plateaus")
    ap.add_argument("--save-frequency", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="resume from state.json in the run dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="write all stage configs and exit (no training)")
    args = ap.parse_args()

    os.chdir(REPO)  # so all repo-relative paths resolve
    object_terms, pose_en, hold_en = VARIANTS[args.variant]
    ladder = DEFAULT_LADDER

    work = REPO / "objectaug_work" / args.run_name
    cfgdir = work / "cfgs"
    cfgdir.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"

    start_idx, prev_ckpt = 0, None
    if args.resume and state_path.exists():
        st = json.loads(state_path.read_text())
        start_idx = st["next_stage"]
        prev_ckpt = st.get("last_ckpt")
        print(f"[objectaug] resuming at stage {start_idx}; last_ckpt={prev_ckpt}", flush=True)

    def rel(p):
        """Repo-relative path string. Absolute paths are made relative to REPO;
        already-relative inputs (e.g. a repo-relative --init-checkpoint) pass
        through unchanged (cwd is REPO)."""
        p = Path(p)
        return str(p.relative_to(REPO)) if p.is_absolute() else str(p)

    for idx in range(start_idx if not args.dry_run else 0, len(ladder)):
        sm, sx, yaw, tr = ladder[idx]
        exp_name = f"smplx_objectaug_{args.run_name}_stage{idx}"
        env_cfg = cfgdir / f"env_stage{idx}.yaml"
        train_cfg = cfgdir / f"train_stage{idx}.yaml"

        header = (
            f"# objectAug curriculum '{args.run_name}' variant={args.variant} -- stage {idx}\n"
            f"# GENERATED by scripts/objectaug_runner.py -- do not edit by hand.\n"
            f"# ranges: scale [{sm},{sx}], yaw +/-{yaw}rad, translate +/-{tr}m\n"
        )
        env_cfg.write_text(gen.render(
            object_terms, pose_en, hold_en, header=header,
            scale_min=sm, scale_max=sx, yaw_rad=yaw, translate_m=tr))

        # Warm-start: stage 0 from the fold-in policy; later stages from the
        # previous stage's checkpoint. On --resume, prefer this stage's OWN
        # in-flight snapshots if it was interrupted mid-run (mirrors the
        # curriculum runner's caveat-D fix).
        inflight = None if args.dry_run else latest_checkpoint(exp_name)
        if inflight is not None:
            resume_from = rel(inflight)
            print(f"[objectaug] stage {idx} resuming from its own in-flight "
                  f"checkpoint {resume_from}", flush=True)
        elif idx == 0 and prev_ckpt is None:
            resume_from = rel(args.init_checkpoint)     # fold-in policy
        elif prev_ckpt:
            resume_from = rel(prev_ckpt)                # previous stage
        else:
            # dry-run: previous stage hasn't trained; show where it WILL come from.
            resume_from = f"checkpoints/smplx_objectaug_{args.run_name}_stage{idx - 1}/nn/mimic.pth"

        # Reuse the curriculum train template (same MLP arch as the fold-in
        # policy, so the warm-start loads). mask_dead_envs is off here.
        train_cfg.write_text(TRAIN_TMPL.format(
            stage=idx, active=args.variant, exp_name=exp_name,
            save_frequency=args.save_frequency, resume_from=resume_from,
            mask_dead_envs="false"))

        print(f"\n[objectaug] === stage {idx}/{len(ladder) - 1} [{args.variant}] "
              f"scale[{sm},{sx}] yaw+/-{yaw} tr+/-{tr} exp={exp_name} "
              f"resume={resume_from} ===", flush=True)
        if args.dry_run:
            print(f"  wrote {rel(env_cfg)}, {rel(train_cfg)}", flush=True)
            continue

        log_path = work / f"stage{idx}.log"
        epochs_run, exited = run_stage(rel(env_cfg), rel(train_cfg), exp_name, log_path,
                                       args, args.min_epochs, args.patience,
                                       args.stage_max_epochs)
        ckpt = latest_checkpoint(exp_name)
        if ckpt is None:
            print(f"[objectaug] ERROR: stage {idx} produced no checkpoint "
                  f"(exited_on_own={exited}). Aborting; inspect {log_path}.", flush=True)
            sys.exit(1)
        prev_ckpt = str(ckpt)
        state_path.write_text(json.dumps(
            {"next_stage": idx + 1, "last_ckpt": prev_ckpt}, indent=2))
        print(f"[objectaug] stage {idx} done: {epochs_run} epochs, ckpt {rel(ckpt)}", flush=True)

    print("[objectaug] all stages complete." if not args.dry_run
          else "[objectaug] dry-run complete (configs generated, nothing launched).", flush=True)


if __name__ == "__main__":
    main()

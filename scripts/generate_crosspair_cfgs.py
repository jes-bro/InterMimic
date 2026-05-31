#!/usr/bin/env python3
"""Generate per-(body, source, object) cross-pair teacher cfg sets.

For each (body, source, object) triple, writes 4 files:
  env   yaml: isaacgym/src/intermimic/data/cfg/omomo_train_crosspair_b{body}_s{src}_{obj}.yaml
  train yaml: isaacgym/src/intermimic/data/cfg/train/rlg/omomo_crosspair_b{body}_s{src}_{obj}.yaml
  shell:      isaacgym/scripts/train_crosspair_b{body}_s{src}_{obj}.sh
  slurm:      slurm_crosspair_b{body}_s{src}_{obj}.sh

Default-reward set: 4 bodies x 2 sources x 2 objects = 16 teachers.
Body-normalized-reward variants: 2 teachers, at fixed (body, source) on each
object, for an A/B against the corresponding default-reward teachers.

Run from repo root:
  python scripts/generate_crosspair_cfgs.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BODIES  = [10, 17, 9, 3]
SOURCES = [2, 6]
OBJECTS = ["largetable", "woodchair"]

# (body, source, object) triples that ALSO get a bodyNormalizedReward=true
# variant. The default-reward teacher at the same triple is still trained,
# so we get a clean A/B per object.
REWARD_VARIANTS = [
    (10, 2, "largetable"),
    (10, 2, "woodchair"),
]


ENV_YAML_TMPL = """\
# Cross-pair teacher: body=sub{body} x source=sub{src} x object={obj}.
# Single-object specialization for cleaner demonstration quality.
# Body controlled at runtime is sub{body} (subjectBodies); the motion file
# subject is sub{src} (dataSub). The policy must adapt sub{src}'s motion
# to sub{body}'s body.{reward_blurb}
env:
  numEnvs: 4096
  envSpacing: 2
  episodeLength: 300
  isFlagrun: False
  enableDebugVis: False
  playdataset: False
  projtype: "None"
  saveImages: False
  initVel: False
  moreRigid: False
  pdControl: True
  powerScale: 1.0
  controlFrequencyInv: 2
  stateInit: "Hybrid"
  rolloutLength: 300
  hybridInitProb: 0.1
  dataFPS: 30
  dataFramesScale: 1
  dataSub: ['sub{src}']
  subjectBodies: ['sub{body}']
  dataObjects: ['{obj}']
  betas_file: scripts/omomo_betas.npz{reward_flag}
  ballSize: 1.
  numObs: 3230
  motion_file: InterAct/OMOMO_new
  robotType: "smplx/omomo.xml"
  objectDensity: 200
  localRootObs: False
  keyBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"]
  contactBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3","L_Thumb3","R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3","R_Thumb3"]
  terminationHeight: 0.15
  enableEarlyTermination: True
  physicalBufferSize: 3

  asset:
    assetRoot: "intermimic/data/assets"

  plane:
    staticFriction: 0.9
    dynamicFriction: 0.9
    restitution: 0.7

  rewardWeights:
    p: 30.
    r: 2.5
    pv: 0.
    rv: 0.
    op: 5.0
    or: 0.1
    opv: 0.1
    orv: 0.
    ig: 5.
    cg_hand: 5.
    cg_other: 5.
    cg_all: 3.
    eg1: 0.00002
    eg2: 0.00002
    eg3: 0.000000001

sim:
  substeps: 2
  physx:
    num_threads: 4
    solver_type: 1
    num_position_iterations: 4
    num_velocity_iterations: 1
    contact_offset: 0.02
    rest_offset: 0.0
    bounce_threshold_velocity: 0.2
    max_depenetration_velocity: 100.0
    default_buffer_size_multiplier: 20.0
    max_gpu_contact_pairs: 34603008

  flex:
    num_inner_iterations: 10
    warm_start: 0.25
"""

TRAIN_YAML_TMPL = """\
params:
  seed: -1

  algo:
    name: intermimic

  model:
    name: intermimic

  network:
    name: intermimic
    separate: True

    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: -2.9
        fixed_sigma: True
        learn_sigma: False

    mlp:
      units: [1024, 1024, 512]
      activation: relu
      d2rl: False

      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: False

  config:
    name: mimic
    # Cross-pair teacher: body=sub{body} x source=sub{src} x object={obj}{reward_tag}.
    full_experiment_name: {exp_name}
    env_name: rlgpu
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: False
    reward_shaper:
      scale_value: 1
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 2e-5
    lr_schedule: constant
    score_to_win: 20000
    max_epochs: 100000
    save_best_after: 100
    save_frequency: 100
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.0
    truncate_grads: False
    ppo: True
    e_clip: 0.2
    horizon_length: 32
    minibatch_size: 16384
    mini_epochs: 6
    critic_coef: 5
    clip_value: False
    seq_len: 4
    bounds_loss_coef: 10
    enable_eps_greedy: False
    save_intermediate: True
    # No resume_from: train from random init so sub2-source and sub6-source
    # teachers are treated symmetrically. Warm-starting from sub2-only stage-1
    # would give sub2 teachers an unfair head start.
"""

SHELL_TMPL = """\
#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${{SCRIPT_DIR}}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair teacher: body=sub{body} x source=sub{src} x object={obj}{reward_tag}.
# Trains from random init (no warm-start) so sub2 and sub6 sources are symmetric.
# Saves to checkpoints/{exp_name}/nn/.
python -u -m intermimic.run \\
    --task InterMimic \\
    --cfg_env isaacgym/src/intermimic/data/cfg/{env_basename}.yaml \\
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/{train_basename}.yaml \\
    --headless \\
    --output checkpoints
"""

SLURM_TMPL = """\
#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="{job_name}"
#SBATCH --output={job_name}-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

source ~/.bashrc
conda deactivate
conda activate intermimic-gym2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"

sh isaacgym/scripts/{shell_basename}.sh
"""


def write_one(body, src, obj, norm_reward):
    """Write 4 files for one (body, src, obj) triple."""
    suffix = "_normreward" if norm_reward else ""
    slug = f"b{body}_s{src}_{obj}{suffix}"
    exp_name = f"smplx_crosspair_{slug}"
    env_basename = f"omomo_train_crosspair_{slug}"
    train_basename = f"omomo_crosspair_{slug}"
    shell_basename = f"train_crosspair_{slug}"
    slurm_basename = f"slurm_crosspair_{slug}"
    job_name = f"cp_{slug}"

    reward_blurb = (
        "\n# bodyNormalizedReward=true: pose-error term is divided by per-env "
        "body height\n# so absolute pose errors don't penalize taller bodies "
        "disproportionately."
        if norm_reward else ""
    )
    reward_flag = "\n  bodyNormalizedReward: True" if norm_reward else ""
    reward_tag = " [body-normalized reward]" if norm_reward else ""

    env_yaml = ENV_YAML_TMPL.format(
        body=body, src=src, obj=obj,
        reward_blurb=reward_blurb, reward_flag=reward_flag,
    )
    train_yaml = TRAIN_YAML_TMPL.format(
        body=body, src=src, obj=obj, reward_tag=reward_tag, exp_name=exp_name,
    )
    shell = SHELL_TMPL.format(
        body=body, src=src, obj=obj, reward_tag=reward_tag, exp_name=exp_name,
        env_basename=env_basename, train_basename=train_basename,
    )
    slurm = SLURM_TMPL.format(
        job_name=job_name, shell_basename=shell_basename,
    )

    env_path   = REPO / "isaacgym/src/intermimic/data/cfg" / f"{env_basename}.yaml"
    train_path = REPO / "isaacgym/src/intermimic/data/cfg/train/rlg" / f"{train_basename}.yaml"
    shell_path = REPO / "isaacgym/scripts" / f"{shell_basename}.sh"
    slurm_path = REPO / f"{slurm_basename}.sh"

    env_path.write_text(env_yaml)
    train_path.write_text(train_yaml)
    shell_path.write_text(shell)
    shell_path.chmod(0o755)
    slurm_path.write_text(slurm)
    slurm_path.chmod(0o755)
    return env_path, train_path, shell_path, slurm_path


def main():
    written = []
    # default-reward set
    for body in BODIES:
        for src in SOURCES:
            for obj in OBJECTS:
                written.append(write_one(body, src, obj, norm_reward=False))
    # body-normalized-reward variants
    for body, src, obj in REWARD_VARIANTS:
        written.append(write_one(body, src, obj, norm_reward=True))

    print(f"Wrote {len(written)} cfg sets ({len(written) * 4} files total):")
    for tup in written:
        env, *_ = tup
        print(f"  {env.relative_to(REPO)}")


if __name__ == "__main__":
    main()

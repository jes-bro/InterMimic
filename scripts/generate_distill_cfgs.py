#!/usr/bin/env python3
"""Generate the 12 distillation cfg sets:

  3 object configurations:
    - largetable only (8 teachers)
    - woodchair only  (8 teachers)
    - both objects    (16 teachers)

  x 2 betas configurations:
    - with betas    (student obs includes body shape, numObsRetarget=3230)
    - no betas      (student obs strips betas channel, numObsRetarget=3198) — ablation

  x 2 reward configurations:
    - normreward    (body-normalized reward in PPO loss)
    - default       (no body-shape normalization)

For each variant, writes 4 files:
    env yaml:    isaacgym/src/intermimic/data/cfg/omomo_distill_{slug}.yaml
    train yaml:  isaacgym/src/intermimic/data/cfg/train/rlg/omomo_distill_{slug}.yaml
    shell:       isaacgym/scripts/train_distill_{slug}.sh
    slurm:       slurm_distill_{slug}.sh

Slug format:
    <object>[_nobetas][_normreward]
e.g.:
    largetable                       (with betas, default reward)
    largetable_normreward            (with betas, body-norm reward)
    largetable_nobetas               (no betas, default reward)
    largetable_nobetas_normreward    (no betas, body-norm reward)

Run from repo root:
    python scripts/generate_distill_cfgs.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (object_tag, dataObjects_yaml_list, teacher_dir_suffix, comment)
OBJECT_CONFIGS = [
    ("largetable", "['largetable']", "crosspair_largetable",
     "Single-object distillation on largetable (8 teachers)."),
    ("woodchair", "['woodchair']", "crosspair_woodchair",
     "Single-object distillation on woodchair (8 teachers)."),
    ("both", "['largetable', 'woodchair']", "crosspair_both",
     "Multi-object distillation on largetable+woodchair (16 teachers)."),
]


ENV_YAML_TMPL = """\
# Cross-pair distillation: {object_comment}
#
# Student is body-conditioned (betas obs channel). Supervised by the
# teachers in {teacher_dir} via online query during distillation.
#
# Reward variant: {reward_comment}
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
  dataSub: ['sub2', 'sub6']
  subjectBodies: ['sub10', 'sub17', 'sub9', 'sub3']
  dataObjects: {data_objects}
  maxClipsPerObject: 15
  betas_file: scripts/omomo_betas.npz
  bodyNormalizedReward: {body_norm}
  ballSize: 1.
  numObs: 3230
  numObsRetarget: {num_obs_retarget}
  useTransformerObs: False
  motion_file: InterAct/OMOMO_new
  motion_file_retarget: InterAct/OMOMO_new
  robotType: "smplx/omomo.xml"
  objectDensity: 200
  localRootObs: False
  keyBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"]
  contactBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3","L_Thumb3","R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3","R_Thumb3"]
  terminationHeight: 0.15
  enableEarlyTermination: True
  physicalBufferSize: 3

  teacherPolicy: {teacher_dir}
  teacherPolicyCFG: intermimic/data/cfg/train/rlg/omomo_crosspair_b10_s2_largetable.yaml

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
    # Cross-pair distillation: {object_comment} {reward_comment}.
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
    save_best_after: 500
    save_frequency: 500
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.0
    truncate_grads: False
    ppo: True
    e_clip: 0.2
    horizon_length: 16
    minibatch_size: 8192
    mini_epochs: 6
    critic_coef: 5
    clip_value: False
    seq_len: 4
    bounds_loss_coef: 10
    expert_loss_coef: 1
    enable_eps_greedy: False
    resume_from: 'None'
    save_intermediate: True
"""

SHELL_TMPL = """\
#!/bin/sh
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${{SCRIPT_DIR}}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Cross-pair distillation: {object_comment} {reward_comment}.
# Saves to checkpoints/{exp_name}/nn/.
python -u -m intermimic.run_distill \\
    --task InterMimic_CrossPair \\
    --cfg_env isaacgym/src/intermimic/data/cfg/{env_basename}.yaml \\
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/{train_basename}.yaml \\
    --headless \\
    --output checkpoints
"""

SLURM_TMPL = """\
#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=15:00:00
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


def write_one(object_tag, data_objects, teacher_dir_suffix, object_comment, body_norm, use_betas):
    betas_suffix = "" if use_betas else "_nobetas"
    reward_suffix = "_normreward" if body_norm else ""
    slug = f"{object_tag}{betas_suffix}{reward_suffix}"
    exp_name = f"smplx_distill_{slug}"
    env_basename = f"omomo_distill_{slug}"
    train_basename = f"omomo_distill_{slug}"
    shell_basename = f"train_distill_{slug}"
    slurm_basename = f"slurm_distill_{slug}"
    job_name = f"dst_{slug}"

    reward_comment = (
        "body-normalized reward in PPO loss (divides pose-error by per-env body height)"
        if body_norm
        else "default reward (no body-shape normalization)"
    )
    betas_comment = (
        "student receives the 32-dim betas channel (body shape conditioning)"
        if use_betas
        else "student does NOT receive betas (ablation; numObsRetarget=3198)"
    )

    teacher_dir = f"checkpoints/teachers/{teacher_dir_suffix}"
    body_norm_str = "True" if body_norm else "False"
    num_obs_retarget = 3230 if use_betas else 3198

    env_yaml = ENV_YAML_TMPL.format(
        object_comment=f"{object_comment} | Betas: {betas_comment}",
        reward_comment=reward_comment,
        data_objects=data_objects, body_norm=body_norm_str,
        num_obs_retarget=num_obs_retarget,
        teacher_dir=teacher_dir,
    )
    train_yaml = TRAIN_YAML_TMPL.format(
        object_comment=object_comment, reward_comment=reward_comment,
        exp_name=exp_name,
    )
    shell = SHELL_TMPL.format(
        object_comment=object_comment, reward_comment=reward_comment,
        exp_name=exp_name,
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

    return env_path


def main():
    written = []
    for object_tag, data_objects, teacher_dir_suffix, object_comment in OBJECT_CONFIGS:
        for use_betas in [True, False]:
            for body_norm in [True, False]:
                p = write_one(
                    object_tag, data_objects, teacher_dir_suffix, object_comment,
                    body_norm, use_betas,
                )
                written.append(p)

    print(f"Wrote {len(written)} distillation cfg sets ({len(written) * 4} files total):")
    for env in written:
        print(f"  {env.relative_to(REPO)}")


if __name__ == "__main__":
    main()

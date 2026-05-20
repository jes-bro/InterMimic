# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InterMimic (CVPR 2025) is a research codebase for training a unified policy for whole-body physics-based human-object interaction (HOI). It supports two humanoids — SMPL-X and Unitree G1 — and runs on **two separate simulators** that live side by side in this repo:

- `isaacgym/` — the original Isaac Gym implementation (training + inference, the canonical path)
- `isaaclab/` — a newer Isaac Lab port (data replay + policy inference; training is not yet ported)

These two trees do **not** share a Python process or conda environment. They have their own scripts, configs, and entrypoints. Isaac Gym checkpoints (`.pth`) are forward-compatible with the Isaac Lab inference path via `isaaclab/src/intermimic_lab/policy_loader.py`.

The third top-level tree, `InterAct/`, contains motion-capture data (e.g. `InterAct/OMOMO`, `InterAct/OMOMO_new`) that the simulators load.

## Environments

There are two distinct setups; pick by which simulator you are running.

**Isaac Gym (Python 3.8 conda env, training + inference):**
```bash
conda activate intermimic-gym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
The `LD_LIBRARY_PATH` export is required after every activation so Isaac Gym can find `libpython3.8.so` — without it Gym scripts fail to import.

**Isaac Lab (uses Isaac Sim's bundled Python via `$ISAACLAB_PATH/isaaclab.sh`):**
```bash
export ISAACLAB_PATH=/path/to/IsaacLab   # recommended: Isaac Sim 5.1.0 + IsaacLab v2.3.1
```
Isaac Lab scripts (`isaaclab/scripts/*.sh`) source `isaaclab.sh` and won't run without `ISAACLAB_PATH`.

## Common Commands

All shell scripts in `isaacgym/scripts/` and `isaaclab/scripts/` set their own `PYTHONPATH` and must be invoked from the **repo root** (they reference `isaacgym/src/intermimic/data/cfg/...` as relative paths).

**Isaac Gym — Training:**
- `sh isaacgym/scripts/train_teacher.sh` — teacher policy, OMOMO
- `sh isaacgym/scripts/train_teacher_new.sh` — high-fidelity (slower, more realistic) teacher variant
- `sh isaacgym/scripts/train_student.sh` — student distillation (MLP)
- `sh isaacgym/scripts/train_student_transformer.sh` — student distillation (transformer)
- `sh isaacgym/scripts/train_teacher_multigpu.sh` — multi-GPU via `torchrun`; defaults to all GPUs, override with `NUM_GPUS=N`. When using multi-GPU, reduce `mini_epochs` and `minibatch_size` in the training YAML since gradients average across ranks.

**Isaac Gym — Inference / Eval:**
- `sh isaacgym/scripts/test_teacher.sh` — visual rollout (requires `checkpoints/smplx_teachers/sub2.pth`)
- `sh isaacgym/scripts/eval_teacher.sh` — headless, `--num_envs 1024`, emits metrics (steps, pose error, success rate)
- `sh isaacgym/scripts/test_student.sh` / `eval_student.sh` — student variants
- `sh isaacgym/scripts/test_g1.sh` — Unitree G1 humanoid (`checkpoints/g1/sub8.pth`)
- `sh isaacgym/scripts/data_replay.sh` — replay ground-truth motion (`--play_dataset` mode of the test runner)

**Isaac Lab — Replay and inference (require `ISAACLAB_PATH`):**
- `./isaaclab/scripts/run_data_replay.sh --num-envs 8 --motion-dir InterAct/OMOMO_new`
- `./isaaclab/scripts/test_policy.sh --checkpoint checkpoints/smplx_teachers/sub2.pth --num_envs 16` — loads an Isaac Gym checkpoint via the policy loader and validates env/obs/stepping

## Architecture

### Entrypoints and the rl_games integration

The Isaac Gym training/inference stack is built on top of [rl_games](https://github.com/Denys88/rl_games):

- `isaacgym/src/intermimic/run.py` — main CLI for teacher training/inference (`--task InterMimic` or `InterMimicG1`). Detects torchrun (`RANK`/`LOCAL_RANK`) and binds each process to its local GPU; per-rank seed offsets are applied automatically.
- `isaacgym/src/intermimic/run_distill.py` — student distillation entrypoint (`--task InterMimic_All`).
- Two config files are always passed: `--cfg_env` (environment YAML in `data/cfg/`) and `--cfg_train` (rl_games YAML in `data/cfg/train/rlg/`). Don't conflate them — env configs control simulator/task, train configs control the rl_games algo/model/network.

Task selection happens via Python `eval()` on the `--task` name in `utils/parse_task.py`, so the task class must be importable there. Two wrappers exist:
- `VecTaskPythonWrapper` — standard PPO/teacher path (`parse_task`)
- `VecTaskDAggerWrapper` — DAgger-style distillation path (`parse_task_distill`)

### Task hierarchy (Isaac Gym)

In `isaacgym/src/intermimic/env/tasks/`:

```
base_task.py            → vec_task.py → humanoid.py (Humanoid_SMPLX)
                                     → humanoid_g1.py
                                                      ↓
                                            intermimic.py (InterMimic, 1252 lines — the main task)
                                            intermimic_g1.py (InterMimicG1)
                                            intermimic_all.py (InterMimic_All — multi-subject distill)
```

`InterMimic.__init__` reads the motion directory from `cfg['env']['motion_file']` and enumerates files filtered by `cfg['env']['dataSub']` (e.g. `['sub2']`). The set of unique object names is derived from filename suffixes (`*_<object>_*.npy`). Four `StateInit` modes exist (`Default`/`Start`/`Random`/`Hybrid`); evaluation metrics are only valid in `Start` mode — the task auto-disables `enableEvaluation` if `stateInit` differs and prints a warning.

**PSI (Physically-corrected State Initialization)**: enable by setting `physicalBufferSize: N` (N > 1) in the env YAML. This is the mechanism described in the paper for better-than-mocap state init.

### Learning code (Isaac Gym)

`isaacgym/src/intermimic/learning/` contains rl_games subclasses:

- `intermimic_agent.py` / `intermimic_agent_distill.py` — PPO agent + DAgger student
- `intermimic_models.py` / `intermimic_models_teacher.py` — model wrappers
- `intermimic_network_builder.py` / `intermimic_transformer_network_builder.py` — actor/critic networks (MLP and transformer variants for student)
- `intermimic_players.py` / `intermimic_players_distill.py` — inference-time rollouts
- `a2c_common.py`, `common_agent.py`, `common_player.py` — local forks of rl_games base classes

### Isaac Lab port

`isaaclab/src/intermimic_lab/` is a self-contained `DirectRLEnv` implementation:

- `intermimic_env.py` — `InterMimicEnv` (1900+ lines) — owns motion loading, USD object spawning via `MeshConverter`, contact sensing, reference-motion visualization markers
- `config/` — `InterMimicEnvCfg`, `SMPLXHumanoidCfg`, `InterMimicSceneCfg` (Isaac Lab cfg-class style)
- `policy_loader.py` — `IsaacGymPolicyWrapper` that adapts an Isaac Gym `.pth` checkpoint to the Isaac Lab observation layout. Bridges the two simulators for inference.
- `torch_utils_gym.py` / `torch_utils.py` — math helpers; the `_gym` suffix one preserves Isaac Gym's exact conventions where they differ from Isaac Lab's
- `assets/usd/` — generated USD files (Isaac Lab needs USD, not the MJCF/URDF in `isaacgym/src/intermimic/data/assets/`)

### Data layout

- `InterAct/OMOMO`, `InterAct/OMOMO_new` — motion data (the `_new` build has minor fixes; results may differ slightly from the paper)
- `InterAct/OMOMO_retarget` — retargeting tooling/data
- `isaacgym/src/intermimic/data/assets/` — `smplx/` (humanoid MJCF), `g1/` (Unitree G1 URDF+meshes), `objects/` — Isaac Gym side
- `isaaclab/src/intermimic_lab/assets/usd/` — Isaac Lab side (USD, regenerated from the above when needed)
- `checkpoints/` (gitignored / downloaded) — pretrained weights go here; scripts expect `checkpoints/smplx_teachers/sub2.pth` and `checkpoints/g1/sub8.pth` by default

## Gotchas

- Scripts must be run from the repo root; they use relative paths to YAML configs.
- The `intermimic-gym` conda env requires the `LD_LIBRARY_PATH` export every session — symptom of forgetting it is an import-time failure looking for `libpython3.8.so`.
- Isaac Lab scripts will refuse to start without `ISAACLAB_PATH`; they exit with an explicit error message.
- The `requirement.txt` pins `numpy==1.21.1`, `rl-games==1.1.4`, `protobuf==3.20.0` — these are not arbitrary; Isaac Gym and the local rl_games subclasses depend on these exact versions.
- Don't add the task name to a script and expect `parse_task.py` to pick it up — you must also import the class in `parse_task.py` so `eval()` can resolve it.
- `--record-video` (Isaac Lab) needs `imageio` + `imageio-ffmpeg` installed into the Isaac Sim Python, not the conda env: `$ISAACLAB_PATH/isaaclab.sh -p -m pip install --upgrade imageio imageio-ffmpeg`.

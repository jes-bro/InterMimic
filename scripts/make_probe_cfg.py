#!/usr/bin/env python3
"""Write a throughput-probe cfg pair by EDITING a real arm's cfgs.

A probe must differ from the arm it represents by exactly the knobs under test,
so this never copies a cfg by hand -- it loads the real one, applies the knobs,
and writes the result. If the real arm changes, every future probe picks that up.

Two safety properties the probe cfgs must have, both enforced here:
  * self-terminating -- max_epochs is overridden, so the job ends on its own.
  * cannot collide   -- full_experiment_name becomes probe_<tag>, and max_epochs
                        is checked to stay under save_best_after so no checkpoint
                        is ever written into a real run's directory.

Every knob is fail-loud: setting one whose key is missing from the base cfg
raises rather than silently adding a key the task code would ignore (a probe that
silently measures nothing is worse than one that crashes).

Usage:
    python3 scripts/make_probe_cfg.py \
        --base-env  isaacgym/src/intermimic/data/cfg/omomo_teacher_src2_xf_aug_retarget.yaml \
        --base-train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_src2_xf_aug_retarget.yaml \
        --out-env /tmp/env.yaml --out-train /tmp/train.yaml \
        --tag cpumotion_env6144 --epochs 20 --cpu-motion --num-envs 6144
"""

import argparse
import sys

import yaml

# Knobs that live under cfg['sim']['physx'] rather than cfg['env'].
PHYSX_KNOBS = {"default_buffer_size_multiplier", "max_gpu_contact_pairs"}


def _set_env(cfg, key, value, applied):
    """Set an env-section knob, refusing to invent a key the task won't read."""
    if key not in cfg["env"]:
        raise KeyError(
            f"'{key}' is not in the base cfg's env section. Adding it here would "
            f"produce a probe that measures nothing -- check the key name against "
            f"KNOWN_ENV_KEYS in intermimic.py.")
    cfg["env"][key] = value
    applied.append(f"{key}={value}")


def _set_physx(cfg, key, value, applied):
    """Set a physx knob, refusing to invent a key PhysX won't read."""
    if key not in cfg["sim"]["physx"]:
        raise KeyError(f"'{key}' is not in the base cfg's sim.physx section.")
    cfg["sim"]["physx"][key] = value
    applied.append(f"physx.{key}={value}")


def build(base_env, base_train, tag, epochs, num_envs=None, cpu_motion=False,
          buffer_mult=None, contact_pairs=None):
    """Return (env_cfg, train_cfg, applied_knobs) for one probe."""
    env = yaml.safe_load(open(base_env))
    train = yaml.safe_load(open(base_train))
    applied = []

    if num_envs is not None:
        _set_env(env, "numEnvs", int(num_envs), applied)
    if cpu_motion:
        # cpuMotionData may be absent from the base cfg (it defaults False in
        # intermimic.py), so this is the one knob allowed to add its key.
        env["env"]["cpuMotionData"] = True
        applied.append("cpuMotionData=True")
    if buffer_mult is not None:
        _set_physx(env, "default_buffer_size_multiplier", float(buffer_mult), applied)
    if contact_pairs is not None:
        _set_physx(env, "max_gpu_contact_pairs", int(contact_pairs), applied)

    cfg = train["params"]["config"]
    save_after = cfg.get("save_best_after", 0)
    if int(epochs) >= save_after:
        raise ValueError(
            f"--epochs {epochs} is >= save_best_after {save_after}: this probe "
            f"would write checkpoints. Lower --epochs or raise save_best_after.")
    cfg["max_epochs"] = int(epochs)
    cfg["full_experiment_name"] = f"probe_{tag}"

    return env, train, applied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-env", required=True)
    ap.add_argument("--base-train", required=True)
    ap.add_argument("--out-env", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--cpu-motion", action="store_true")
    ap.add_argument("--buffer-mult", type=float, default=None)
    ap.add_argument("--contact-pairs", type=int, default=None)
    args = ap.parse_args(argv)

    env, train, applied = build(
        args.base_env, args.base_train, args.tag, args.epochs,
        num_envs=args.num_envs, cpu_motion=args.cpu_motion,
        buffer_mult=args.buffer_mult, contact_pairs=args.contact_pairs)

    yaml.safe_dump(env, open(args.out_env, "w"), sort_keys=False)
    yaml.safe_dump(train, open(args.out_train, "w"), sort_keys=False)
    print(f"[probe-cfg] {args.tag}: {', '.join(applied) if applied else 'NONE (control)'}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Widen an InterMimic teacher checkpoint to accept extra obs dimensions.

The first linear layer in actor_mlp and critic_mlp is widened by N input
columns (zero-init), and running_mean_std stats are padded to match
(mean=0, var=1 for new dims). Output is ready to be loaded into a network
with obs_dim = original + N.

Why zero-init: starting weights for the new betas columns are 0, so on
load the network produces exactly the same outputs it did before. The new
input channel literally has no effect until training updates it.

Why mean=0, var=1: rl_games' RunningMeanStd normalizes obs as
    (x - mean) / sqrt(var). New dims with mean=0, var=1 pass through
identity. After fine-tuning starts, the stats will update from real data.

Example:
    python scripts/widen_checkpoint.py \\
        --input  checkpoints/smplx_teachers/sub2.pth \\
        --output checkpoints/smplx_teachers/sub2_betas.pth \\
        --extra-dims 32
"""

import argparse
from pathlib import Path

import torch


def widen_linear_weight(w, n_extra):
    """Linear weight [out_dim, in_dim] -> [out_dim, in_dim + n_extra], zero-init new columns."""
    out_dim, in_dim = w.shape
    new = torch.zeros(out_dim, in_dim + n_extra, dtype=w.dtype, device=w.device)
    new[:, :in_dim] = w
    return new


def widen_running_stat(t, n_extra, fill):
    """1-D running stat [in_dim] -> [in_dim + n_extra], padded with `fill`."""
    in_dim = t.shape[0]
    new = torch.full((in_dim + n_extra,), fill, dtype=t.dtype, device=t.device)
    new[:in_dim] = t
    return new


def _print_structure(ckpt, prefix="ckpt"):
    """Recursively print keys + shapes of a checkpoint for debugging."""
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"  {prefix}.{k}: dict ({len(v)} keys)")
                _print_structure(v, prefix=f"{prefix}.{k}")
            elif isinstance(v, torch.Tensor):
                print(f"  {prefix}.{k}: tensor {tuple(v.shape)}")
            else:
                print(f"  {prefix}.{k}: {type(v).__name__} = {repr(v)[:80]}")
    elif isinstance(ckpt, torch.Tensor):
        print(f"  {prefix}: tensor {tuple(ckpt.shape)}")


def widen_checkpoint(ckpt, n_extra, verbose=True):
    if verbose:
        print("\n[INPUT STRUCTURE]")
        _print_structure(ckpt)

    # rl_games checkpoints typically have a 'model' top-level key wrapping
    # the network's state_dict.
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # Find first-layer weights for actor and critic
    # Standard rl_games key naming: a2c_network.actor_mlp.0.weight
    first_layer_keys = [
        k for k in state_dict.keys()
        if k.endswith('actor_mlp.0.weight') or k.endswith('critic_mlp.0.weight')
    ]
    if not first_layer_keys:
        # Print state_dict keys for debugging if expected pattern missing
        print("\n[ERROR] No actor_mlp.0.weight or critic_mlp.0.weight found.")
        print("state_dict keys:")
        for k in state_dict.keys():
            print(f"  {k}: {tuple(state_dict[k].shape)}")
        raise SystemExit("Could not find first MLP layer to widen.")

    print(f"\n[WIDENING WEIGHTS by {n_extra} input columns]")
    for k in first_layer_keys:
        old_shape = tuple(state_dict[k].shape)
        state_dict[k] = widen_linear_weight(state_dict[k], n_extra)
        new_shape = tuple(state_dict[k].shape)
        print(f"  {k}: {old_shape} -> {new_shape}")

    # Running mean/std — search in common locations
    # rl_games puts it at top-level under 'running_mean_std' key, or sometimes
    # 'reward_running_mean_std' for the value head's reward normalizer.
    print("\n[EXTENDING RUNNING STATS]")
    rms_locations = []
    if isinstance(ckpt, dict):
        if 'running_mean_std' in ckpt and isinstance(ckpt['running_mean_std'], dict):
            rms_locations.append(('running_mean_std', ckpt['running_mean_std']))
        # Some rl_games versions also have value running mean/std for the
        # value head's targets — we don't widen that (it's 1-D, scalar value).

    if not rms_locations:
        print("  no running_mean_std found in checkpoint (will rely on env init)")

    for name, rms in rms_locations:
        for stat_key, fill in [('running_mean', 0.0), ('running_var', 1.0)]:
            if stat_key in rms:
                old_shape = tuple(rms[stat_key].shape)
                if len(old_shape) != 1:
                    print(f"  {name}.{stat_key}: unexpected shape {old_shape}, skipping")
                    continue
                rms[stat_key] = widen_running_stat(rms[stat_key], n_extra, fill)
                new_shape = tuple(rms[stat_key].shape)
                print(f"  {name}.{stat_key}: {old_shape} -> {new_shape}")
        # 'count' stays scalar; no shape change needed

    # Drop optimizer state — its exp_avg / exp_avg_sq shapes match the
    # original (pre-widen) parameters, so reloading them into the widened
    # network would crash. Fine-tuning with a fresh Adam state is standard
    # practice when the obs structure changes; we just lose the momentum
    # estimates from the canonical-training run, which is fine.
    if isinstance(ckpt, dict) and 'optimizer' in ckpt:
        print("\n[DROPPING OPTIMIZER STATE]")
        print("  (Adam exp_avg/exp_avg_sq for first-layer weights would mismatch;")
        print("   fresh Adam start is the standard for obs-changing fine-tune.)")
        ckpt['optimizer'] = None

    return ckpt


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--extra-dims", type=int, required=True,
                   help="Number of new input dims to add.")
    p.add_argument("--probe-only", action="store_true",
                   help="Print checkpoint structure and exit; no widening.")
    args = p.parse_args()

    print(f"Loading {args.input}")
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)

    if args.probe_only:
        print("\n[STRUCTURE PROBE — no changes]")
        _print_structure(ckpt)
        return

    ckpt = widen_checkpoint(ckpt, args.extra_dims)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving widened checkpoint to {args.output}")
    torch.save(ckpt, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

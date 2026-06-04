#!/usr/bin/env python3
"""Inspect a saved rl_games checkpoint and print useful metadata so you
can verify what training run it came from.

Usage:
    python scripts/inspect_checkpoint.py <path_to_checkpoint.pth>
"""
import sys
import torch
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_checkpoint.py <path>.pth")
        sys.exit(1)
    path = Path(sys.argv[1]).expanduser()
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    print(f"Inspecting: {path}")
    print(f"File size:  {path.stat().st_size / 1e6:.1f} MB")
    print()
    ck = torch.load(str(path), map_location="cpu")

    print("Top-level keys:", list(ck.keys()))
    print()

    # Common rl_games metadata
    print(f"epoch:              {ck.get('epoch', '(not present)')}")
    print(f"frame:              {ck.get('frame', '(not present)')}")
    print(f"last_mean_rewards:  {ck.get('last_mean_rewards', '(not present)')}")
    print()

    # Network architecture — first layer input dim tells us obs dim
    if "model" in ck:
        sd = ck["model"]
        first_layer_keys = [k for k in sd.keys() if "actor_mlp.0.weight" in k]
        if first_layer_keys:
            k = first_layer_keys[0]
            shape = tuple(sd[k].shape)
            print(f"Actor first layer ({k}): shape {shape}")
            obs_dim = shape[1]
            if obs_dim == 3230:
                print("  → 3230 obs = WITH betas conditioning")
            elif obs_dim == 3198:
                print("  → 3198 obs = NO betas conditioning")
            elif obs_dim == 6396:
                print("  → 6396 obs = transformer student (4 timesteps)")
            else:
                print(f"  → unrecognized obs dim {obs_dim}")
        action_keys = [k for k in sd.keys() if "mu.weight" in k]
        if action_keys:
            shape = tuple(sd[action_keys[0]].shape)
            print(f"Action head: shape {shape} (action dim = {shape[0]})")

    print()
    print("Quick checks:")
    epoch = ck.get("epoch", 0)
    if isinstance(epoch, int):
        if epoch < 500:
            print("  ⚠ Tiny epoch — could be a NaN-crashed teacher.")
        elif epoch < 4000:
            print("  Moderate epoch — could be a still-training run or early checkpoint.")
        else:
            print("  Large epoch — looks like a fully-trained run.")


if __name__ == "__main__":
    main()

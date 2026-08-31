#!/usr/bin/env python3
"""How far did a checkpoint actually train? Prints epoch, frame and reward.

A rolling `mimic.pth` carries no epoch in its filename, so two runs' finals can
sit 40,000 epochs apart and look identical on disk. That is exactly the question
raised by the gen-2 grid: every fold-0 run scores 75-83% while every fold-1 run
scores 38-48%, on the same recipe and the same 43 bodies -- undertraining and a
real fold effect predict the same table, and only the epoch counts separate them.

rl_games writes 'epoch', 'frame' and 'last_mean_rewards' alongside the weights
(a2c_common.py get_full_state_weights), so the answer is in the file.

    python3 scripts/checkpoint_epoch.py checkpoints/*/nn/mimic.pth
    python3 scripts/checkpoint_epoch.py collab/jm/checkpointsjm/*__f1/nn/mimic.pth

Loads onto CPU with weights_only=False -- these are your own checkpoints, but
the flag is explicit because newer torch defaults it to True and would refuse
the pickled optimiser state.
"""
import argparse
import sys
from pathlib import Path

import torch

# What we report, in the order rl_games conceptually produces it. Anything
# missing prints as "--" rather than being guessed at from another field: a
# checkpoint saved by a different code path may genuinely not carry it.
KEYS = ["epoch", "frame", "last_mean_rewards"]


def read(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None, f"not a dict ({type(ckpt).__name__}) -- not an rl_games checkpoint"
    out = {}
    for k in KEYS:
        v = ckpt.get(k)
        if hasattr(v, "item"):          # 0-d tensors show as tensor(1.23) otherwise
            try:
                v = v.item()
            except (ValueError, RuntimeError):
                pass
        out[k] = v
    if all(out[k] is None for k in KEYS):
        return None, f"none of {KEYS} present; keys are {sorted(ckpt)[:8]}"
    return out, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    a = p.parse_args()

    print(f"{'epoch':>9} {'frame':>14} {'reward':>8}  checkpoint")
    print("-" * 78)
    bad = 0
    for path in a.checkpoints:
        if not path.exists():
            print(f"{'--':>9} {'--':>14} {'--':>8}  {path}   [MISSING]")
            bad += 1
            continue
        try:
            vals, err = read(path)
        except Exception as e:                       # noqa: BLE001 - report, don't die
            print(f"{'--':>9} {'--':>14} {'--':>8}  {path}   [{type(e).__name__}: {e}]")
            bad += 1
            continue
        if vals is None:
            print(f"{'--':>9} {'--':>14} {'--':>8}  {path}   [{err}]")
            bad += 1
            continue
        f = lambda v, w, spec: (format(v, spec) if isinstance(v, (int, float))
                                else "--").rjust(w)
        print(f"{f(vals['epoch'], 9, 'd' if isinstance(vals['epoch'], int) else '.0f')} "
              f"{f(vals['frame'], 14, ',d' if isinstance(vals['frame'], int) else '.0f')} "
              f"{f(vals['last_mean_rewards'], 8, '.3f')}  {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

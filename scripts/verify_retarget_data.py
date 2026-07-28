#!/usr/bin/env python3
"""Verify a directory of retargeted references is real, current, and correct.

Counting files is not enough: the bug that killed the first smoke run wrote a
full, complete-looking set of clips whose body_pos was in the WRONG FRAME. This
checks CONTENT, per body:

  1. world frame   -- body 0 must equal root_pos exactly. Root-local output (the
                      shipped bug) fails here by ~1 m.
  2. right body    -- bone lengths must match THAT body's MJCF, so <body>/ really
                      holds that body's retarget and not a copy of the source.
  3. self-consistent -- FK(dof, root) must reproduce the written body_pos.
  4. untouched fields -- root_pos/root_rot/obj_pos identical to the source clip.
  5. freshness     -- newest/oldest mtime, so a stale set left over from a
                      previous (skipped, because resumable) run is visible.

Usage:
    python3 scripts/verify_retarget_data.py InterAct/OMOMO_retarget_contact_smoke
    python3 scripts/verify_retarget_data.py <dir> --source-dir InterAct/OMOMO_new
    python3 scripts/verify_retarget_data.py <dir> --per-body 5   # sample more clips

Exit code 0 = all bodies PASS. Non-zero = do not train on this data.
"""

import argparse
import datetime
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_contact import (  # noqa: E402
    MJCFChain, NB, I_ROOTP, I_ROOTQ, I_DOF, I_BODY, I_OBJP, quat_xyzw_to_mat)

WORLD_TOL_M = 1e-4      # body0 vs root_pos; exact in practice
BONE_TOL_M = 2e-3       # vs MJCF rest offsets (measured agreement is ~4e-5)
FK_TOL_M = 5e-3         # written body_pos vs FK of written dof


def check_clip(path, body, source_dir):
    """Return (ok, [messages]) for one retargeted clip."""
    msgs = []
    clip = torch.load(path, map_location="cpu", weights_only=False).detach().double()
    T = clip.shape[0]
    body_pos = clip[:, I_BODY].reshape(T, NB, 3)

    # 1. world frame: body 0 IS the root
    d0 = (body_pos[:, 0] - clip[:, I_ROOTP]).norm(dim=-1).mean().item()
    if d0 > WORLD_TOL_M:
        msgs.append(f"NOT WORLD FRAME: body0 is {d0*100:.1f} cm from root_pos "
                    f"(root-local output looks like ~100 cm)")

    # 2. right body: bone lengths must match this body's MJCF
    chain = MJCFChain(body)
    worst, worst_b = 0.0, None
    for b in range(1, NB):
        L = (body_pos[:, b] - body_pos[:, chain.parent[b]]).norm(dim=-1).mean().item()
        e = abs(L - chain.offset[b].norm().item())
        if e > worst:
            worst, worst_b = e, chain.names[b]
    if worst > BONE_TOL_M:
        msgs.append(f"WRONG BODY: bone '{worst_b}' differs from {body}'s MJCF by "
                    f"{worst*1000:.2f} mm")

    # 3. self-consistency: written body_pos == FK(written dof, written root)
    p = chain.fk(clip[:, I_DOF], root_pos=clip[:, I_ROOTP],
                 root_rot=quat_xyzw_to_mat(clip[:, I_ROOTQ]))
    fk_e = (p - body_pos).norm(dim=-1).max().item()
    if fk_e > FK_TOL_M:
        msgs.append(f"INCONSISTENT: FK(dof) vs written body_pos off by {fk_e*1000:.2f} mm")

    # 4. fields the solve does not own must be untouched
    src = os.path.join(source_dir, os.path.basename(path))
    if os.path.exists(src):
        s = torch.load(src, map_location="cpu", weights_only=False).detach().double()
        if s.shape != clip.shape:
            msgs.append(f"SHAPE {tuple(clip.shape)} != source {tuple(s.shape)}")
        else:
            for name, sl in [("root_pos", I_ROOTP), ("root_rot", I_ROOTQ), ("obj_pos", I_OBJP)]:
                if (clip[:, sl] - s[:, sl]).abs().max().item() > 1e-6:
                    msgs.append(f"{name} was MODIFIED (must be copied through)")
            moved = (body_pos - s[:, I_BODY].reshape(T, NB, 3)).norm(dim=-1).mean().item()
            msgs.append(f"info: moved {moved*100:.2f} cm from source"
                        + (" (identity: expect ~0)" if body == os.path.basename(path).split("_")[0]
                           else ""))
    else:
        msgs.append(f"info: no source clip at {src}, skipped field-preservation check")

    return not any(not m.startswith("info:") for m in msgs), msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="retargeted data dir (contains <body>/ subdirs)")
    ap.add_argument("--source-dir", default="InterAct/OMOMO_new",
                    help="original clips, for the field-preservation check")
    ap.add_argument("--per-body", type=int, default=2,
                    help="clips to deep-check per body (default 2)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"FAIL: {args.dir} does not exist"); return 2
    bodies = sorted(d for d in os.listdir(args.dir)
                    if os.path.isdir(os.path.join(args.dir, d)))
    if not bodies:
        print(f"FAIL: no <body>/ subdirs under {args.dir}"); return 2

    print(f"verifying {args.dir}  ({len(bodies)} bodies)\n")
    all_ok = True
    for body in bodies:
        clips = sorted(glob.glob(os.path.join(args.dir, body, "*.pt")))
        if not clips:
            print(f"  {body:>8}: FAIL -- no clips"); all_ok = False; continue
        mt = [os.path.getmtime(c) for c in clips]
        fmt = "%Y-%m-%d %H:%M"
        span = (f"{datetime.datetime.fromtimestamp(min(mt)):{fmt}}"
                f" .. {datetime.datetime.fromtimestamp(max(mt)):{fmt}}")
        ok, notes = True, []
        for c in clips[:args.per_body]:
            cok, msgs = check_clip(c, body, args.source_dir)
            ok &= cok
            notes += [f"      {os.path.basename(c)}: {m}" for m in msgs]
        all_ok &= ok
        print(f"  {body:>8}: {'PASS' if ok else 'FAIL'}  {len(clips):>4} clips  written {span}")
        for n in notes:
            print(n)
    print(f"\n{'DATA OK -- safe to train on' if all_ok else 'DATA BAD -- do not train on this'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

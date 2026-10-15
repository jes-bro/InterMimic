#!/usr/bin/env python3
"""A 591-channel InterMimic motion .pt -> the DUMP_TRAJ .npz layout.

So a reference motion (a novice's own reconstructed clip, an expert's clip, a
retargeted reference) and a policy rollout are the SAME kind of file and go
through the same renderer (render_mesh_replay.py --dump) and the same
differencing (motion_feedback.py): body_pos (T,52,3), body_rot (T,52,4 xyzw),
obj_pos (T,3), obj_rot (T,4 xyzw), subject.

Channel layout of the .pt (intermimic.py _load_motion; verified on data:
quaternions unit-norm, xyzw, world z-up):
    root_pos 0:3  root_rot 3:7  dof_pos 9:162  body_pos 162:318
    obj_pos 318:321  obj_rot 321:325  body_rot 383:591

    python3 scripts/clip_to_dump.py InterAct/<tag>/sub204_bball_000.pt --out novice_original.npz
"""
import argparse
import os
import re
import sys

import numpy as np

BODY_POS, BODY_ROT = slice(162, 318), slice(383, 591)
OBJ_POS, OBJ_ROT = slice(318, 321), slice(321, 325)
N_BODIES = 52


def clip_to_dump(x, subject):
    """(T,591) float array -> the dump dict. Refuses a non-591 width or non-unit quaternions."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 591:
        raise ValueError(f"expected (T, 591), got {x.shape}")
    body_rot = x[:, BODY_ROT].reshape(len(x), N_BODIES, 4)
    obj_rot = x[:, OBJ_ROT]
    for name, q in (("body_rot", body_rot), ("obj_rot", obj_rot)):
        n = np.linalg.norm(q, axis=-1)
        if not np.allclose(n, 1.0, atol=1e-3):
            raise ValueError(f"{name}: quaternions are not unit-norm (min {n.min():.4f}, max {n.max():.4f}); "
                             f"not a 591-channel InterMimic clip?")
    return {"body_pos": x[:, BODY_POS].reshape(len(x), N_BODIES, 3), "body_rot": body_rot,
            "obj_pos": x[:, OBJ_POS], "obj_rot": obj_rot, "subject": subject}


def subject_of(path):
    m = re.match(r"(sub\d+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"clip name does not start with sub<N>_: {path}")
    return m.group(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip", help="591-channel .pt")
    ap.add_argument("--out", required=True, help=".npz in the DUMP_TRAJ layout")
    ap.add_argument("--subject", default=None, help="default: from the clip name (sub<N>_...)")
    a = ap.parse_args(argv)
    import torch
    x = torch.load(a.clip, map_location="cpu", weights_only=False).detach().numpy()
    d = clip_to_dump(x, a.subject or subject_of(a.clip))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **d)
    print(f"wrote {a.out}: {d['body_pos'].shape[0]} frames, subject {d['subject']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

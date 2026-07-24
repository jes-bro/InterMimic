#!/usr/bin/env python3
"""See the contact retargeting: the TARGET body's hands driven by the SOURCE's
dof_pos (BEFORE, the reference the policy currently gets) vs the retargeted dof_pos
(AFTER), next to the object surface. If the retarget worked, the hands snap ONTO
the object in the AFTER panel on contact frames.

Renders a side-by-side (before | after) 3D skeleton + object-point-cloud animation
to mp4. Pure CPU (reuses retarget_contact's MJCF FK + smplx_pose's quat_to_mat);
no Isaac Gym, no SMPL-X models.

  python3 scripts/visualize_retarget.py --clip InterAct/OMOMO_new/sub2_largetable_000.pt \
      --source sub2 --target sub9 --retarget-dir InterAct/OMOMO_retarget_contact_smoke \
      --out ~/Downloads/retarget_sub9_largetable.mp4
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_contact import MJCFChain, I_DOF, I_BODY  # noqa: E402
from smplx_pose import quat_to_mat                       # noqa: E402

I_ROOT_POS = slice(0, 3)
I_ROOT_ROT = slice(3, 7)          # quat xyzw
I_OBJ_POS = slice(318, 321)
I_OBJ_ROT = slice(321, 325)       # quat xyzw
I_CONTACT_H = slice(331, 383)     # 52 per-body contact flags
OBJ_ASSETS = "isaacgym/src/intermimic/data/assets/objects/objects"

INK, HAND, OBJ = "#52514e", "#e34948", "#2a78d6"


def fk_world(chain, dof, root_pos, root_rot_quat):
    """Target-body joint world positions (T,52,3) with the clip's real root."""
    T = dof.shape[0]
    R = torch.stack([torch.tensor(quat_to_mat(q), dtype=torch.float64)
                     for q in root_rot_quat.numpy()])          # (T,3,3)
    return chain.fk(dof.to(torch.float64), root_pos.to(torch.float64), R).numpy()


def obj_cloud(obj_name, obj_pos, obj_rot_quat):
    """Object surface points in world per frame (T,P,3), or None if no samples."""
    p = os.path.join(OBJ_ASSETS, obj_name, "sample_points.npy")
    if not os.path.exists(p):
        return None
    pts = np.load(p).astype(np.float64)                        # (P,3) canonical
    if pts.shape[0] > 400:                                     # thin for render speed
        pts = pts[np.random.default_rng(0).choice(pts.shape[0], 400, replace=False)]
    out = np.empty((obj_pos.shape[0], pts.shape[0], 3))
    for t in range(obj_pos.shape[0]):
        R = quat_to_mat(obj_rot_quat[t].numpy())
        out[t] = pts @ R.T + obj_pos[t].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", default="sub2")
    ap.add_argument("--target", required=True)
    ap.add_argument("--retarget-dir", required=True)
    ap.add_argument("--stride", type=int, default=4, help="render every Nth frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = os.path.basename(a.clip)
    rpath = os.path.join(a.retarget_dir, a.target, base)
    if not os.path.exists(rpath):
        raise SystemExit(f"no retargeted clip at {rpath} (generate it first)")
    src = torch.load(a.clip, map_location="cpu", weights_only=False).detach()
    ret = torch.load(rpath, map_location="cpu", weights_only=False).detach()
    obj_name = base.split("_")[-2]
    chain = MJCFChain(a.target)

    root_pos, root_rot = src[:, I_ROOT_POS], src[:, I_ROOT_ROT]
    before = fk_world(chain, src[:, I_DOF], root_pos, root_rot)   # target driven by SOURCE dof
    after = fk_world(chain, ret[:, I_DOF], root_pos, root_rot)    # target driven by RETARGETED dof
    obj = obj_cloud(obj_name, src[:, I_OBJ_POS], src[:, I_OBJ_ROT])
    contact = (src[:, I_CONTACT_H].numpy() > 0.5)
    hands = [i for i, n in enumerate(chain.names)
             if "Wrist" in n or any(f in n for f in ("Index", "Middle", "Pinky", "Ring", "Thumb"))]
    parent = chain.parent

    # Shared bounds so before/after are comparable and the object stays in view.
    allpts = [before.reshape(-1, 3), after.reshape(-1, 3)]
    if obj is not None:
        allpts.append(obj.reshape(-1, 3))
    P = np.concatenate(allpts)
    ctr, rad = P.mean(0), np.abs(P - P.mean(0)).max()

    # GIF by default (no ffmpeg dependency); pass --out something.mp4 to force mp4
    # if imageio-ffmpeg is installed.
    out = a.out or os.path.expanduser(f"~/Downloads/retarget_{a.target}_{obj_name}.gif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames_idx = list(range(0, before.shape[0], a.stride))
    rendered = []
    print(f"[viz] {a.source}->{a.target} {obj_name}: {len(frames_idx)} frames -> {out}")

    def draw(ax, pos, t, title):
        ax.clear()
        for j, pj in enumerate(parent):        # bones
            if pj >= 0:
                ax.plot(*zip(pos[j], pos[pj]), color=INK, lw=1.0, alpha=0.7)
        ax.scatter(*pos.T, s=6, color=INK)
        ax.scatter(*pos[hands].T, s=28, color=HAND, label="hands")     # hand joints
        if obj is not None:
            ax.scatter(*obj[t].T, s=3, color=OBJ, alpha=0.5)
        ax.set_title(f"{title}\nframe {t}{'  (contact)' if contact[t].any() else ''}",
                     fontsize=10, color=INK)
        for lim, c in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            lim(ctr[c] - rad, ctr[c] + rad)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=12, azim=-70)

    for t in frames_idx:
        fig = plt.figure(figsize=(9, 4.6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        draw(ax1, before[t], t, f"BEFORE  (target {a.target} + source dof)")
        draw(ax2, after[t], t, f"AFTER  (retargeted: hands on object)")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        rendered.append(img)
        plt.close(fig)
    try:
        imageio.mimsave(out, rendered, fps=a.fps)
    except Exception as e:                      # mp4 without ffmpeg -> fall back to gif
        alt = os.path.splitext(out)[0] + ".gif"
        print(f"[viz] {out} failed ({e!r}); writing {alt} instead")
        imageio.mimsave(alt, rendered, fps=a.fps)
        out = alt
    print(f"[viz] done -> {out}")


if __name__ == "__main__":
    main()

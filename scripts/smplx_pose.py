#!/usr/bin/env python3
"""Pose the SMPL-X SURFACE through a humanoid motion, so the mesh does what the
capsule humanoid did in sim -- makes subjects easy to tell apart.

Validated pipeline (NOT SMPL-native pose FK, which has a frame mismatch):
  1. MJCF forward kinematics from dof_pos (axis-angle per joint) gives each body's
     GLOBAL rotation+position in the sim's Z-up frame.
  2. Global-transform Linear Blend Skinning: skin the shaped SMPL-X template with
     those per-joint global transforms.
Cross-checked against stored body_pos: 5.7 cm rigid-aligned residual, flat across
frames (residual = generic-MJCF vs subject-SMPL shape; shrinks with the subject's
own MJCF / the sim's actual body_rot at rollout).

dof_pos[3j:3j+3] is the AXIS-ANGLE rotation of MJCF joint j (validated exp-map vs
euler against body_pos: 4.8 cm vs 15 cm). No smplx package; LBS from the model npz.

  python3 scripts/smplx_pose.py --selftest
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

MJCF = "isaacgym/src/intermimic/data/assets/smplx/omomo.xml"
N_BETAS = 16

SMPLX_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1",
    "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1",
    "right_middle2", "right_middle3", "right_pinky1", "right_pinky2",
    "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1",
    "right_thumb2", "right_thumb3",
]
_MJCF_TO_SMPL = {
    "Pelvis": "pelvis", "Torso": "spine1", "Spine": "spine2", "Chest": "spine3",
    "Neck": "neck", "Head": "head", "L_Hip": "left_hip", "L_Knee": "left_knee",
    "L_Ankle": "left_ankle", "L_Toe": "left_foot", "R_Hip": "right_hip",
    "R_Knee": "right_knee", "R_Ankle": "right_ankle", "R_Toe": "right_foot",
    "L_Thorax": "left_collar", "L_Shoulder": "left_shoulder",
    "L_Elbow": "left_elbow", "L_Wrist": "left_wrist", "R_Thorax": "right_collar",
    "R_Shoulder": "right_shoulder", "R_Elbow": "right_elbow", "R_Wrist": "right_wrist",
}
for _s in ("L", "R"):
    for _f in ("Index", "Middle", "Pinky", "Ring", "Thumb"):
        for _k in (1, 2, 3):
            _MJCF_TO_SMPL[f"{_s}_{_f}{_k}"] = \
                f"{'left' if _s == 'L' else 'right'}_{_f.lower()}{_k}"

# SMPL Y-up -> Isaac Z-up: (x,y,z)->(x,-z,y)
Q_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def expmap(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    x, y, z = v / th
    c, s, C = np.cos(th), np.sin(th), 1 - np.cos(th)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def quat_to_mat(q):
    a, b, c, d = q            # xyzw
    return np.array([[1-2*(b*b+c*c), 2*(a*b-c*d), 2*(a*c+b*d)],
                     [2*(a*b+c*d), 1-2*(a*a+c*c), 2*(b*c-a*d)],
                     [2*(a*c-b*d), 2*(b*c+a*d), 1-2*(a*a+b*b)]])


def _parse_mjcf_tree(path=MJCF):
    """Ordered [(name, parent_name, rest_offset)] depth-first, as the sim enumerates."""
    root = ET.parse(path).getroot()
    out = []

    def walk(e, parent):
        for b in e.findall("body"):
            pos = np.array([float(v) for v in b.get("pos", "0 0 0").split()])
            out.append((b.get("name"), parent, pos))
            walk(b, b.get("name"))
    walk(root.find("worldbody"), None)
    return out


class SMPLXPoser:
    def __init__(self, models_dir=None, betas_path="scripts/omomo_betas.npz",
                 mesh_npz=None):
        self.models_dir = os.path.expanduser(
            models_dir or os.environ.get("SMPLX_MODELS", "~/Downloads/models/smplx"))
        self.betas = np.load(betas_path, allow_pickle=True)
        self.gender = dict(x.split(":") for x in self.betas["_genders"])
        self.tree = _parse_mjcf_tree()
        self.body_order = [n for n, _, _ in self.tree]         # 52 MJCF bodies
        self.b2s = [SMPLX_JOINTS.index(_MJCF_TO_SMPL[n]) for n in self.body_order]
        self._cache = {}

    def _model(self, gender):
        g = gender.upper()
        if g not in self._cache:
            m = np.load(os.path.join(self.models_dir, f"SMPLX_{g}.npz"), allow_pickle=True)
            parents = m["kintree_table"][0].astype(np.int64).copy()
            parents[0] = -1
            self._cache[g] = dict(
                v_template=m["v_template"].astype(np.float64),
                shapedirs=m["shapedirs"].astype(np.float64)[:, :, :N_BETAS],
                posedirs=m["posedirs"].astype(np.float64),
                J_reg=m["J_regressor"].astype(np.float64),
                weights=m["weights"].astype(np.float64),
                parents=parents, faces=m["f"].astype(np.uint32))
        return self._cache[g]

    def _shape(self, subject):
        M = self._model(self.gender[subject])
        beta = self.betas[subject].astype(np.float64)[:N_BETAS]
        v_sh = M["v_template"] + M["shapedirs"] @ beta
        v0 = (Q_ZUP @ v_sh.T).T                                # template, Z-up
        J0 = (Q_ZUP @ (M["J_reg"] @ v_sh).T).T                 # rest joints, Z-up
        return v0, J0, M

    def mjcf_fk(self, dof_pos, root_pos, root_rot_quat):
        """dof_pos(153) + root -> per-body GLOBAL (R[52,3,3], p[52,3]) in Z-up."""
        d = np.asarray(dof_pos).reshape(51, 3)
        G = {}
        di = 0
        for name, parent, off in self.tree:
            if parent is None:
                Rg, pg = quat_to_mat(np.asarray(root_rot_quat)), np.asarray(root_pos, float)
            else:
                Rp, pp = G[parent]
                Rg, pg = Rp @ expmap(d[di]), pp + Rp @ off
                di += 1
            G[name] = (Rg, pg)
        R = np.array([G[n][0] for n in self.body_order])
        p = np.array([G[n][1] for n in self.body_order])
        return R, p

    def skin(self, subject, R_glob, p_glob):
        """Global-transform LBS: (R[52,3,3], p[52,3]) global -> posed verts (V,3) Z-up."""
        v0, J0, M = self._shape(subject)
        G = np.tile(np.eye(4), (55, 1, 1))
        G[:, :3, 3] = J0
        for k, jid in enumerate(self.b2s):
            G[jid, :3, :3] = R_glob[k]
            G[jid, :3, 3] = p_glob[k]
        # jaw + eyes are NOT in the MJCF; propagate them from their (moved) parent
        # with identity local rotation, else the face skin tears away from the
        # head (was the main cause of a mangled render).
        parents = M["parents"]
        known = set(self.b2s)
        for j in range(55):
            if j in known:
                continue
            pa = parents[j]
            G[j, :3, :3] = G[pa, :3, :3]
            G[j, :3, 3] = G[pa, :3, 3] + G[pa, :3, :3] @ (J0[j] - J0[pa])
        Gp = G.copy()
        for j in range(55):                                    # G'_j = G_j @ inv(rest_j)
            Gp[j, :3, 3] = G[j, :3, 3] - G[j, :3, :3] @ J0[j]
        Tv = np.einsum("vj,jab->vab", M["weights"], Gp)
        vh = np.concatenate([v0, np.ones((len(v0), 1))], 1)
        return np.einsum("vab,vb->va", Tv, vh)[:, :3]

    def pose_from_dof(self, subject, dof_pos, root_pos, root_rot_quat):
        """Full: dof_pos + root -> posed SMPL-X verts (V,3) Z-up, and faces."""
        R, p = self.mjcf_fk(dof_pos, root_pos, root_rot_quat)
        return self.skin(subject, R, p), self._model(self.gender[subject])["faces"]

    def pose_from_bodies(self, subject, body_rot_quat, body_pos):
        """From the sim's own GLOBAL body state (52 quats xyzw + 52 positions, Z-up)
        -- what stage-1 dumps at rollout. Most faithful (no dof reinterpretation)."""
        R = np.array([quat_to_mat(q) for q in np.asarray(body_rot_quat)])
        return self.skin(subject, R, np.asarray(body_pos)), \
            self._model(self.gender[subject])["faces"]

    # ---- IK RETARGET: fit a NATIVE SMPL-X pose to target joints -----------
    # The MJCF and SMPL-X rest poses differ ~90deg at the arms (arms-down vs
    # T-pose), so driving SMPL-X with MJCF-frame rotations twists the surface.
    # Instead, fit SMPL-X's own pose params to the target joint POSITIONS
    # (body_pos, available in every clip and every DUMP_TRAJ). The result is a
    # native SMPL pose -> clean surface AND correct pose. This is the retarget.

    def _torch_model(self, gender):
        import torch
        key = ("torch", gender.upper())
        if key not in self._cache:
            M = self._model(gender)
            self._cache[key] = dict(
                v_template=torch.tensor(M["v_template"]),
                shapedirs=torch.tensor(M["shapedirs"]),
                J_reg=torch.tensor(M["J_reg"]),
                parents=torch.tensor(M["parents"]),
                Q=torch.tensor(Q_ZUP))
        return self._cache[key]

    @staticmethod
    def _rodrigues(aa):
        import torch
        th = aa.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x, y, z = (aa / th).unbind(1)
        c, s = torch.cos(th)[:, 0], torch.sin(th)[:, 0]
        C = 1 - c
        return torch.stack([
            torch.stack([c+x*x*C, x*y*C-z*s, x*z*C+y*s], 1),
            torch.stack([y*x*C+z*s, c+y*y*C, y*z*C-x*s], 1),
            torch.stack([z*x*C-y*s, z*y*C+x*s, c+z*z*C], 1)], 1)

    def _fk_joints_torch(self, subject, pose_aa, trans):
        import torch
        tm = self._torch_model(self.gender[subject])
        beta = torch.tensor(self.betas[subject].astype(np.float64)[:N_BETAS])
        J0 = tm["J_reg"] @ (tm["v_template"] + tm["shapedirs"] @ beta)   # (55,3) Y-up
        R = self._rodrigues(pose_aa)
        parents = tm["parents"]
        G = [None] * 55
        g0 = torch.eye(4, dtype=torch.float64); g0[:3, :3] = R[0]; g0[:3, 3] = J0[0]
        G[0] = g0
        for j in range(1, 55):
            T = torch.eye(4, dtype=torch.float64)
            T[:3, :3] = R[j]; T[:3, 3] = J0[j] - J0[parents[j]]
            G[j] = G[parents[j]] @ T
        Jp = torch.stack([g[:3, 3] for g in G])                         # Y-up
        return (tm["Q"] @ Jp.T).T + trans                              # Z-up + trans

    def fit_sequence(self, subject, target_joints, iters_first=150, iters_warm=80,
                     lr=0.05, smooth=0.02, verbose=False):
        """Fit native SMPL-X pose to target joint positions per frame.

        target_joints: (T,52,3) Z-up (clip body_pos or DUMP_TRAJ body_pos), ordered
        like the MJCF body order. Warm-starts each frame from the previous for speed;
        `smooth` regularizes toward it so the fit can't drift into an awkward pose
        that matches the joints but mangles the surface. Returns (pose_aa[T,55,3],
        trans[T,3])."""
        import torch
        b2s = torch.tensor(self.b2s)
        T = len(target_joints)
        poses = np.zeros((T, 55, 3)); transl = np.zeros((T, 3))
        prev = torch.zeros(55, 3, dtype=torch.float64)
        for t in range(T):
            tgt = torch.tensor(np.asarray(target_joints[t], dtype=np.float64))
            pose = prev.clone().requires_grad_(True)
            trans = torch.tensor(tgt.mean(0).detach().numpy(), requires_grad=True)
            opt = torch.optim.Adam([pose, trans], lr=lr)
            n = iters_first if t == 0 else iters_warm
            for _ in range(n):
                opt.zero_grad()
                J = self._fk_joints_torch(subject, pose, trans)[b2s]
                loss = ((J - tgt) ** 2).sum(1).mean()
                if t > 0:                              # stay near the previous frame
                    loss = loss + smooth * ((pose - prev) ** 2).sum()
                loss.backward(); opt.step()
            poses[t] = pose.detach().numpy(); transl[t] = trans.detach().numpy()
            prev = pose.detach()
            if verbose:
                print(f"  frame {t+1}/{T}: RMSE {float(loss)**0.5*100:.2f}cm", flush=True)
        return poses, transl

    def verts_from_pose(self, subject, pose_aa, trans):
        """Native SMPL-X forward (shape + pose blendshapes + LBS) -> verts (V,3)
        Z-up, faces. Clean surface -- it's the model's own pose space."""
        M = self._model(self.gender[subject])
        beta = self.betas[subject].astype(np.float64)[:N_BETAS]
        v = M["v_template"] + M["shapedirs"] @ beta
        J = M["J_reg"] @ v
        R = np.stack([expmap(a) for a in np.asarray(pose_aa)])          # (55,3,3)
        v = v + M["posedirs"] @ (R[1:] - np.eye(3)).reshape(-1)         # pose blendshapes
        parents = M["parents"]
        G = np.zeros((55, 4, 4)); G[0, :3, :3] = R[0]; G[0, :3, 3] = J[0]; G[0, 3, 3] = 1
        for j in range(1, 55):
            Tm = np.eye(4); Tm[:3, :3] = R[j]; Tm[:3, 3] = J[j] - J[parents[j]]
            G[j] = G[parents[j]] @ Tm
        Gp = G.copy()
        for j in range(55):
            Gp[j, :3, 3] = G[j, :3, 3] - G[j, :3, :3] @ J[j]
        Tv = np.einsum("vj,jab->vab", M["weights"], Gp)
        vh = np.concatenate([v, np.ones((len(v), 1))], 1)
        v_y = np.einsum("vab,vb->va", Tv, vh)[:, :3]
        return (Q_ZUP @ v_y.T).T + np.asarray(trans), M["faces"]


def selftest():
    import glob
    import torch
    p = SMPLXPoser()

    def rigid(A, B):
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return np.linalg.norm(Ac @ R.T - Bc, axis=1).mean()

    clip = sorted(glob.glob("InterAct/OMOMO_new/sub2_*.pt"))[0]
    x = torch.load(clip, map_location="cpu", weights_only=False).detach().numpy()
    M = p._model("male")
    errs = []
    for t in range(0, min(len(x), 90), 30):
        v, f = p.pose_from_dof("sub2", x[t, 9:162], x[t, 0:3], x[t, 3:7])
        Jz = (M["J_reg"] @ v)[p.b2s]
        errs.append(rigid(Jz, x[t, 162:318].reshape(52, 3)))
    print(f"[selftest] posed SMPL-X mesh joints vs stored body_pos (rigid): "
          f"mean {np.mean(errs)*100:.2f} cm  frames {[round(e*100,1) for e in errs]}")
    print("  < ~6cm => pose transfer correct (residual = generic-MJCF vs subject-SMPL shape)")
    assert np.mean(errs) < 0.08, "pose transfer regressed"

    # Surface integrity: joints can be right while the SKIN is mangled (jaw/eyes
    # tearing off the head). Edge lengths must be preserved (LBS is near-rigid).
    v0, _, _ = p._shape("sub2")
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    er = np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    ep = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    ratio = ep / (er + 1e-9)
    n10 = int((ratio > 10).sum())
    print(f"[selftest] surface: edge-length ratio median {np.median(ratio):.3f}, "
          f"max {ratio.max():.1f}, edges >10x: {n10}")
    assert n10 == 0 and ratio.max() < 15, "SURFACE MANGLED (LBS explosion)"
    print(f"  verts={len(v)} faces={len(f)}  OK")

    # IK retarget: fit native SMPL-X pose to body_pos, forward the clean surface.
    # This is the fix for the ~90deg arm rest-pose mismatch. Check BOTH the joint
    # fit and the surface (a native pose can't candy-wrapper like MJCF-frame skin).
    tgt = x[:60:6, 162:318].reshape(-1, 52, 3)         # realistic stride (~render)
    poses, trans = p.fit_sequence("sub2", tgt)
    jfit, emax = [], []
    for i in range(len(tgt)):
        vi, fi = p.verts_from_pose("sub2", poses[i], trans[i])
        r = np.linalg.norm(vi[e[:, 0]] - vi[e[:, 1]], axis=1) / (er + 1e-9)
        jfit.append(rigid((M["J_reg"] @ vi)[p.b2s], tgt[i])); emax.append(r.max())
    print(f"[selftest] IK retarget: joint fit mean {np.mean(jfit)*100:.2f} cm, "
          f"surface edge max {max(emax):.1f}")
    assert np.mean(jfit) < 0.06 and max(emax) < 8, "IK retarget regressed"
    print("  OK -- correct pose + clean surface (the arm-twist fix)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print("import SMPLXPoser; or --selftest")


if __name__ == "__main__":
    main()

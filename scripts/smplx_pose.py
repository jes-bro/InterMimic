#!/usr/bin/env python3
"""Pose the SMPL-X SURFACE through a humanoid motion, so the mesh does what the
capsule humanoid did in sim -- makes subjects easy to tell apart.

The sim's humanoid IS the SMPL-X skeleton (the MJCFs are built from it, T-pose,
identity body frames), in a different axis order (Q_ZUP below). So a body's global
rotation in the sim is SMPL's global joint rotation, conjugated by Q_ZUP, and the
exact, jitter-free way to the surface is:
  1. per-body GLOBAL rotation + position in the sim frame: from a DUMP_TRAJ's
     body_rot/body_pos directly, or by MJCF forward kinematics from a clip's
     dof_pos (axis-angle per joint; validated exp-map vs euler: 4.8 cm vs 15 cm).
  2. native_pose_from_global: those -> SMPL's own (pose_aa, trans), exactly.
  3. verts_from_pose: SMPL's own forward (shape + pose blendshapes + LBS).
skin() does global-transform LBS straight from step 1 (same surface minus the
pose blendshapes); fit_sequence is the old per-frame IK, kept as an opt-in.

History: Q_ZUP used to be yawed 90 degrees against the MJCF (see its comment), which
made step-1 skinning twist the arms; the IK fitter was written to work around that
and brought its own problems (position-only fit: free twist about every bone, a
straightened back, sideways knees, per-frame jitter). Fixed 2026-10-21.

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

# SMPL-H is SMPL-X without the jaw and the two eyes, in the same order, so the
# MJCF-to-joint mapping is shared and only the indices differ. Built from the
# SMPL-X list rather than retyped, so the two cannot drift apart.
SMPLH_JOINTS = SMPLX_JOINTS[:22] + SMPLX_JOINTS[25:]


def _load_model_file(path):
    """Load a SMPL-family model from .npz or .pkl into plain numpy arrays.

    The SMPL-H release ships pickles holding chumpy arrays, which will not
    unpickle without chumpy installed. A stub is registered when it is missing:
    the arrays only need to survive as buffers, and every field is converted to
    numpy immediately afterwards.

    Raises:
        SystemExit: if the file cannot be read or lacks a field the fit needs.
    """
    import pickle
    import sys as _sys
    import types as _types

    if not os.path.isfile(path):
        raise SystemExit(f"no model file at {path}")

    if path.endswith(".npz"):
        raw = dict(np.load(path, allow_pickle=True))
    else:
        if "chumpy" not in _sys.modules:
            try:
                import chumpy  # noqa: F401
            except ImportError:
                stub = _types.ModuleType("chumpy")

                class Ch(np.ndarray):
                    """Stand-in for chumpy.Ch that unpickles as a plain array."""

                    def __array_finalize__(self, obj):
                        """Nothing to carry over; the array data is the payload."""
                        return

                stub.Ch = Ch
                stub.ch = stub
                _sys.modules["chumpy"] = stub
                _sys.modules["chumpy.ch"] = stub
        with open(path, "rb") as fh:
            raw = pickle.load(fh, encoding="latin1")

    def field(*names):
        """Return the first present key, or fail naming what was looked for."""
        for n in names:
            if n in raw:
                return raw[n]
        raise SystemExit(f"{os.path.basename(path)} has no {names[0]}; "
                         f"found {sorted(raw)[:12]}")

    reg = field("J_regressor")
    if hasattr(reg, "toarray"):
        reg = reg.toarray()

    out = dict(
        v_template=np.asarray(field("v_template"), dtype=np.float64),
        shapedirs=np.asarray(field("shapedirs"), dtype=np.float64),
        J_regressor=np.asarray(reg, dtype=np.float64),
        weights=np.asarray(field("weights"), dtype=np.float64),
        kintree_table=np.asarray(field("kintree_table")),
        faces=np.asarray(field("f", "faces"), dtype=np.uint32))

    # posedirs is OPTIONAL here, unlike the fields above: the rest-pose paths
    # (shape comparison, body galleries) never need it, and some stripped model
    # builds omit it. verts_from_pose warns and skins without it. But when the
    # file has it, it must survive this curation -- it did not, which is why
    # posing a body died on KeyError('posedirs').
    if "posedirs" in raw:
        pd = np.asarray(raw["posedirs"], dtype=np.float64)
        if pd.shape[:2] == (out["v_template"].shape[0], 3):
            out["posedirs"] = pd
        else:
            print(f"[smplx_pose] ignoring posedirs with unexpected shape "
                  f"{pd.shape}", flush=True)

    # Validate rather than trust. A model whose parts disagree produces a body
    # that renders without complaint and is wrong, which is the failure this
    # whole exercise is trying to escape.
    n_v = out["v_template"].shape[0]
    if out["v_template"].shape != (n_v, 3):
        raise SystemExit(f"v_template is {out['v_template'].shape}, expected (V, 3)")
    if out["shapedirs"].shape[:2] != (n_v, 3):
        raise SystemExit(f"shapedirs is {out['shapedirs'].shape}, expected (V, 3, B)")
    if out["weights"].shape[0] != n_v:
        raise SystemExit(f"weights has {out['weights'].shape[0]} rows for "
                         f"{n_v} vertices")
    if out["J_regressor"].shape[1] != n_v:
        raise SystemExit(f"J_regressor is {out['J_regressor'].shape}, expected "
                         f"(J, {n_v})")
    return out


# SMPL's template frame is x = the body's LEFT, y = UP, z = FORWARD. The MJCF
# humanoids (omomo.xml and every subject file) stand with x = FORWARD, y = LEFT,
# z = UP: Pelvis->L_Hip is +y, Pelvis->Torso is +z, L_Shoulder->L_Elbow is +y.
# So sim = Q_ZUP @ smpl with Q_ZUP = (z, x, y). Measured 2026-10-21 against
# omomo.xml's rest skeleton with its own subject's betas (sub9): 0.01 cm RMS over
# the 22 body joints. The previous matrix, (x, -z, y), kept y-up -> z-up but left
# the body yawed 90 degrees against the MJCF: 37 cm RMS, and the "arm twist" the
# IK path was written to work around -- skinning a template whose arms point +x
# with body rotations of a skeleton whose arms point +y.
# Both rest poses are the T-pose; there is NO rest-pose mismatch beyond this axis
# permutation, so the sim's body rotations are SMPL's own global joint rotations
# after conjugation: R_smpl = Q_ZUP^T @ R_sim @ Q_ZUP (native_pose_from_global).
Q_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)


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
                 mesh_npz=None, model_type=None, mjcf=None):
        """mjcf: the humanoid whose body order / rest offsets index dof_pos and
        body_pos (default: the generic omomo.xml). Pass the subject's own file for
        exact forward kinematics of that subject's clips."""
        self.models_dir = os.path.expanduser(
            models_dir or os.environ.get("SMPLX_MODELS", "~/Downloads/models/smplx"))
        self.betas = np.load(betas_path, allow_pickle=True)
        self.gender = dict(x.split(":") for x in self.betas["_genders"])
        # smplh renders the body the betas were actually fitted for; smplx
        # reads those same coefficients in a different shape basis, which
        # distorts build even when the pose is right.
        self.model_type = (model_type
                           or os.environ.get("SMPL_MODEL_TYPE", "smplx")).lower()
        if self.model_type not in ("smplx", "smplh"):
            raise SystemExit(f"model_type must be smplx or smplh, got "
                             f"{self.model_type!r}")
        self.joint_names = (SMPLH_JOINTS if self.model_type == "smplh"
                            else SMPLX_JOINTS)
        self.mjcf = mjcf or MJCF
        self.tree = _parse_mjcf_tree(self.mjcf)
        self.body_order = [n for n, _, _ in self.tree]         # 52 MJCF bodies
        self.b2s = [self.joint_names.index(_MJCF_TO_SMPL[n]) for n in self.body_order]
        self._cache = {}

    def _model(self, gender):
        g = gender.upper()
        if g in self._cache:
            return self._cache[g]

        prefix = "SMPLH" if self.model_type == "smplh" else "SMPLX"
        candidates = [f"{prefix}_{g}.npz", f"{prefix}_{g}.pkl",
                      f"{prefix}_{gender.lower()}.pkl"]
        path = next((os.path.join(self.models_dir, c) for c in candidates
                     if os.path.isfile(os.path.join(self.models_dir, c))), None)
        if path is None:
            raise SystemExit(
                f"no {prefix} model for {gender} in {self.models_dir}; looked "
                f"for {candidates}")
        m = _load_model_file(path)

        parents = m["kintree_table"][0].astype(np.int64).copy()
        parents[0] = -1

        # The joint count the model actually has must match the list this poser
        # indexes with, or every b2s lookup silently addresses the wrong joint.
        n_joints = m["weights"].shape[1]
        if n_joints != len(self.joint_names):
            raise SystemExit(
                f"{os.path.basename(path)} has {n_joints} joints but "
                f"{self.model_type} expects {len(self.joint_names)}. The model "
                f"file and --model-type disagree.")

        n_betas = min(N_BETAS, m["shapedirs"].shape[2])
        # posedirs: the pose-corrective blendshapes verts_from_pose applies.
        # It was missing from this dict, so verts_from_pose died on
        # KeyError('posedirs') the moment anyone skinned a POSED body -- the
        # rest-pose paths never touch it, which is why it survived.
        # Its last axis is (n_joints-1)*9, matching the flattened
        # (R[1:] - I) pose feature; slice it in case a model ships more joints
        # than this poser indexes.
        posedirs = m.get("posedirs")
        if posedirs is not None:
            want = (n_joints - 1) * 9
            if posedirs.shape[-1] < want:
                raise SystemExit(
                    f"{os.path.basename(path)}: posedirs has "
                    f"{posedirs.shape[-1]} pose components but {n_joints} "
                    f"joints need {want}. Model and --model-type disagree.")
            posedirs = posedirs[..., :want]
        self._cache[g] = dict(
            v_template=m["v_template"],
            shapedirs=m["shapedirs"][:, :, :n_betas],
            J_reg=m["J_regressor"],
            weights=m["weights"],
            posedirs=posedirs,
            parents=parents[:n_joints], faces=m["faces"], n_betas=n_betas)
        print(f"[smplx_pose] {os.path.basename(path)}: "
              f"{m['v_template'].shape[0]} verts, {n_joints} joints, "
              f"{n_betas} betas", flush=True)
        return self._cache[g]

    def _shape(self, subject):
        M = self._model(self.gender[subject])
        beta = self.betas[subject].astype(np.float64)[:M["shapedirs"].shape[2]]
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
        n_j = len(self.joint_names)   # 52 for SMPL-H, 55 for SMPL-X
        """Global-transform LBS: (R[52,3,3], p[52,3]) global -> posed verts (V,3) Z-up."""
        v0, J0, M = self._shape(subject)
        G = np.tile(np.eye(4), (n_j, 1, 1))
        G[:, :3, 3] = J0
        for k, jid in enumerate(self.b2s):
            G[jid, :3, :3] = R_glob[k]
            G[jid, :3, 3] = p_glob[k]
        # jaw + eyes are NOT in the MJCF; propagate them from their (moved) parent
        # with identity local rotation, else the face skin tears away from the
        # head (was the main cause of a mangled render).
        parents = M["parents"]
        known = set(self.b2s)
        for j in range(n_j):
            if j in known:
                continue
            pa = parents[j]
            G[j, :3, :3] = G[pa, :3, :3]
            G[j, :3, 3] = G[pa, :3, 3] + G[pa, :3, :3] @ (J0[j] - J0[pa])
        Gp = G.copy()
        for j in range(n_j):                                    # G'_j = G_j @ inv(rest_j)
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

    # ---- NATIVE POSE, exactly, from the sim's global body state ------------
    # The MJCF skeleton is SMPL's (T-pose, identity body frames) in the Q_ZUP axis
    # order, so each body's global rotation in the sim is SMPL's global joint
    # rotation conjugated by Q_ZUP, and the local pose is parent^T @ child. No
    # fitting, no free twist, no jitter: the surface moves exactly as the sim did.

    def native_pose_from_global(self, subject, R_glob, p_glob):
        """(R[52,3,3], p[52,3]) global in the sim frame, MJCF body order ->
        (pose_aa[n_j,3], trans[3]) in SMPL's own pose space, such that
        verts_from_pose(subject, pose_aa, trans) puts every joint at p_glob."""
        from scipy.spatial.transform import Rotation
        M = self._model(self.gender[subject])
        n_j, parents = len(self.joint_names), M["parents"]
        G = np.tile(np.eye(3), (n_j, 1, 1))
        known = set()
        for k, jid in enumerate(self.b2s):
            G[jid] = Q_ZUP.T @ np.asarray(R_glob[k]) @ Q_ZUP      # sim -> SMPL axes
            known.add(jid)
        for j in range(n_j):                                      # jaw, eyes: ride the head
            if j not in known:
                G[j] = G[parents[j]]
        local = np.empty((n_j, 3, 3))
        local[0] = G[0]
        for j in range(1, n_j):
            local[j] = G[parents[j]].T @ G[j]
        pose_aa = Rotation.from_matrix(local).as_rotvec()
        # verts_from_pose places the pelvis at Q_ZUP @ J0_y[0] + trans.
        beta = self.betas[subject].astype(np.float64)[:M["shapedirs"].shape[2]]
        j0_y = (M["J_reg"] @ (M["v_template"] + M["shapedirs"] @ beta))[0]
        pelvis = np.asarray(p_glob)[self.body_order.index("Pelvis")]
        return pose_aa, pelvis - Q_ZUP @ j0_y

    def fk_joints(self, subject, pose_aa, trans):
        """SMPL's own forward kinematics of (pose_aa, trans) -> joint positions
        (n_j, 3) in the SIM frame: what verts_from_pose's skeleton is, before
        skinning. The exact inverse of native_pose_from_global."""
        M = self._model(self.gender[subject])
        n_j, parents = len(self.joint_names), M["parents"]
        beta = self.betas[subject].astype(np.float64)[:M["shapedirs"].shape[2]]
        J = M["J_reg"] @ (M["v_template"] + M["shapedirs"] @ beta)
        R = np.stack([expmap(a) for a in np.asarray(pose_aa)])
        G_R, G_p = [R[0]], [J[0]]
        for j in range(1, n_j):
            pa = parents[j]
            G_R.append(G_R[pa] @ R[j])
            G_p.append(G_p[pa] + G_R[pa] @ (J[j] - J[pa]))
        return (Q_ZUP @ np.stack(G_p).T).T + np.asarray(trans)

    def native_from_bodies(self, subject, body_rot_quat, body_pos):
        """A DUMP_TRAJ frame (52 quats xyzw + 52 positions, sim frame) -> (pose_aa, trans)."""
        R = np.array([quat_to_mat(q) for q in np.asarray(body_rot_quat)])
        return self.native_pose_from_global(subject, R, np.asarray(body_pos))

    def native_from_dof(self, subject, dof_pos, root_pos, root_rot_quat):
        """A clip frame (dof_pos + root) -> (pose_aa, trans), via MJCF forward kinematics."""
        R, p = self.mjcf_fk(dof_pos, root_pos, root_rot_quat)
        return self.native_pose_from_global(subject, R, p)

    def native_sequence(self, subject, body_rot_quat, body_pos):
        """(T,52,4), (T,52,3) -> (pose_aa[T,n_j,3], trans[T,3]); the drop-in for fit_sequence."""
        out = [self.native_from_bodies(subject, q, p) for q, p in zip(body_rot_quat, body_pos)]
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    # ---- IK RETARGET (opt-in): fit a NATIVE SMPL-X pose to joint POSITIONS ---
    # Written when Q_ZUP was wrong and step-1 skinning twisted the arms. It fits
    # pose params to body_pos only, so rotation about any bone is unobservable
    # (see the notes inside); kept for callers that have positions but no
    # rotations. Anything with body_rot or dof_pos should use the native path.

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
        n_j = len(self.joint_names)   # 52 for SMPL-H, 55 for SMPL-X
        import torch
        tm = self._torch_model(self.gender[subject])
        beta = torch.tensor(self.betas[subject].astype(np.float64)[:tm["shapedirs"].shape[2]])
        J0 = tm["J_reg"] @ (tm["v_template"] + tm["shapedirs"] @ beta)   # (n_j,3) Y-up
        R = self._rodrigues(pose_aa)
        parents = tm["parents"]
        G = [None] * n_j
        g0 = torch.eye(4, dtype=torch.float64); g0[:3, :3] = R[0]; g0[:3, 3] = J0[0]
        G[0] = g0
        for j in range(1, n_j):
            T = torch.eye(4, dtype=torch.float64)
            T[:3, :3] = R[j]; T[:3, 3] = J0[j] - J0[parents[j]]
            G[j] = G[parents[j]] @ T
        Jp = torch.stack([g[:3, 3] for g in G])                         # Y-up
        return (tm["Q"] @ Jp.T).T + trans                              # Z-up + trans

    def fit_sequence(self, subject, target_joints, iters_first=150, iters_warm=80,
                     lr=0.05, smooth=0.02, verbose=False, pinned=None, reg=0.0,
                     reg_weights=None, target_weights=None):
        """Fit native SMPL-X pose to target joint positions per frame.

        target_joints: (T,52,3) Z-up (clip body_pos or DUMP_TRAJ body_pos), ordered
        like the MJCF body order. Warm-starts each frame from the previous for speed;
        `smooth` regularizes toward it so the fit can't drift into an awkward pose
        that matches the joints but mangles the surface. Returns (pose_aa[T,n_j,3],
        trans[T,3]).

        `pinned` is {joint index: (T,3) axis-angle} for joints this objective
        CANNOT see, held at the caller's values instead of being optimised.
        Rotation about a bone moves no joint, so along a near-collinear chain
        like the spine the axial twist is nearly unobservable and the fit puts
        an arbitrary value there -- a torso twisted off to one side. A leaf
        (head, toes) is the degenerate case: wholly unobservable, so it simply
        keeps its initial value. Pinning also shrinks the search space, so the
        remaining joints converge in fewer iterations. dump_to_interact.py fills
        these from the rollout's own body_rot.

        NOTE `smooth` damps every joint, not just the twitchy ones: measured on
        a 400-frame rollout, smooth=0.15 cut the knee's pose variation ~95x and
        tripled the joint residual. Prefer iterations (or `pinned`) over it.
        """
        import torch
        b2s = torch.tensor(self.b2s)
        T = len(target_joints)
        n_j = len(self.joint_names)   # 52 for SMPL-H, 55 for SMPL-X
        pinned = pinned or {}
        pin_idx = sorted(pinned)
        # Per-joint prior weight, (n_j, 1). A flat prior does not stop the fit
        # reaching a target through a joint that should barely move: measured
        # against a CARI4D reconstruction of the same person, the fitted collars
        # ran 1.8x larger (54 degrees, where a real collar moves a few) and the
        # hips and spine about 2x. Those are exactly the joints whose excess
        # shows up in the SURFACE -- an arched back, a distorted seat -- because
        # linear blend skinning has to spread the rotation across the trunk.
        # Weighting them up costs little, since the joints below them can reach
        # the same targets.
        # (n_j,) weights one joint uniformly; (n_j, 3) weights each AXIS, which
        # is what a hinge needs: a knee or elbow bends about one axis only, and
        # weighting the joint as a whole cannot tell flexion from the sideways
        # rotation that swings a shin out from under its thigh. Measured against
        # a CARI4D reconstruction of the same person, the fitted knee carried
        # 0.358 mean on its abduction axis against the reconstruction's 0.098,
        # peaking at 0.79 rad -- 45 degrees of sideways knee, which no knee does.
        reg_w = torch.ones(n_j, 3, dtype=torch.float64)
        if reg_weights is not None:
            w = np.asarray(reg_weights, dtype=np.float64)
            reg_w = torch.tensor(w.reshape(n_j, 1) if w.ndim == 1 else w.reshape(n_j, 3))
        # Per-TARGET weights, (n_b, 1), over the MJCF bodies being matched. The
        # loss treats every joint's centimetres alike, but a centimetre does not
        # cost the same everywhere: the trunk segments are ~13 cm, so the few
        # centimetres of residual left over there swing the trunk by 20-30
        # degrees, while the same error on a 38 cm thigh is invisible. Measured
        # against the rollout's own trunk angles, every prior setting -- flat
        # included -- left the fitted back 18-31 degrees straighter than the
        # motion it was fitting. Weighting the short segments up is a statement
        # about geometry, not about how a body ought to move.
        tgt_w = torch.ones(len(self.b2s), 1, dtype=torch.float64)
        if target_weights is not None:
            tgt_w = torch.tensor(
                np.asarray(target_weights, dtype=np.float64)).reshape(-1, 1)
        poses = np.zeros((T, n_j, 3)); transl = np.zeros((T, 3))
        prev = torch.zeros(n_j, 3, dtype=torch.float64)
        for t in range(T):
            tgt = torch.tensor(np.asarray(target_joints[t], dtype=np.float64))
            pose = prev.clone()
            for j in pin_idx:
                pose[j] = torch.tensor(np.asarray(pinned[j][t], dtype=np.float64))
            pose = pose.requires_grad_(True)
            trans = torch.tensor(tgt.mean(0).detach().numpy(), requires_grad=True)
            opt = torch.optim.Adam([pose, trans], lr=lr)
            n = iters_first if t == 0 else iters_warm
            for _ in range(n):
                opt.zero_grad()
                J = self._fk_joints_torch(subject, pose, trans)[b2s]
                loss = (((J - tgt) ** 2).sum(1, keepdim=True) * tgt_w).mean()
                if t > 0:                              # stay near the previous frame
                    loss = loss + smooth * ((pose - prev) ** 2).sum()
                # Keep the UNOBSERVABLE components small. A hinge's child sits ON
                # its bone axis -- the wrist on the forearm, the ankle on the
                # shin -- so roll about that axis moves no target and the fit can
                # put anything there; measured values reached 12-16 rad, which
                # linear blend skinning renders as forearms twisted around
                # themselves and a knee pinched to nothing. This costs the
                # observable components almost nothing (they are pinned down by
                # the targets) while sending the free ones to neutral instead of
                # wherever the optimiser drifted.
                if reg:
                    loss = loss + reg * ((pose ** 2) * reg_w).sum()
                loss.backward()
                if pin_idx:          # never let the optimiser move a pinned joint
                    pose.grad[pin_idx] = 0.0
                opt.step()
            poses[t] = pose.detach().numpy(); transl[t] = trans.detach().numpy()
            prev = pose.detach()
            if verbose:
                print(f"  frame {t+1}/{T}: RMSE {float(loss)**0.5*100:.2f}cm", flush=True)
        return poses, transl

    def verts_from_pose(self, subject, pose_aa, trans):
        n_j = len(self.joint_names)   # 52 for SMPL-H, 55 for SMPL-X
        """Native SMPL-X forward (shape + pose blendshapes + LBS) -> verts (V,3)
        Z-up, faces. Clean surface -- it's the model's own pose space."""
        M = self._model(self.gender[subject])
        beta = self.betas[subject].astype(np.float64)[:M["shapedirs"].shape[2]]
        v = M["v_template"] + M["shapedirs"] @ beta
        J = M["J_reg"] @ v
        R = np.stack([expmap(a) for a in np.asarray(pose_aa)])          # (n_j,3,3)
        if M.get("posedirs") is not None:
            v = v + M["posedirs"] @ (R[1:] - np.eye(3)).reshape(-1)   # pose blendshapes
        elif not getattr(self, "_warned_posedirs", False):
            # Not silent: without these the surface is still correctly shaped
            # and posed, but loses the pose-dependent corrections (bulging at
            # bent joints). Worth knowing before judging a render.
            print("[smplx_pose] WARNING: model has no posedirs; skinning "
                  "WITHOUT pose blendshapes (joints will look slightly "
                  "collapsed when bent)", flush=True)
            self._warned_posedirs = True
        parents = M["parents"]
        G = np.zeros((n_j, 4, 4)); G[0, :3, :3] = R[0]; G[0, :3, 3] = J[0]; G[0, 3, 3] = 1
        for j in range(1, n_j):
            Tm = np.eye(4); Tm[:3, :3] = R[j]; Tm[:3, 3] = J[j] - J[parents[j]]
            G[j] = G[parents[j]] @ Tm
        Gp = G.copy()
        for j in range(n_j):
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

    def translated(A, B):
        """Mean joint error after matching centroids ONLY. A rigid (rotating)
        alignment would absorb a wrong Q_ZUP -- it did, for months -- so the
        selftest must not rotate."""
        return np.linalg.norm((A - A.mean(0)) - (B - B.mean(0)), axis=1).mean()

    # Axis mapping: the generic MJCF (omomo.xml) IS sub9's skeleton, so its rest
    # joints must coincide with sub9's shaped template under Q_ZUP, to the mm.
    _, J0, _ = p._shape("sub9")
    _, p_rest = p.mjcf_fk(np.zeros(153), np.zeros(3), np.array([0, 0, 0, 1.0]))
    body = [k for k, n in enumerate(p.body_order)
            if not any(f in n for f in ("Index", "Middle", "Pinky", "Ring", "Thumb"))]
    ax_err = translated(J0[[p.b2s[k] for k in body]], p_rest[body])
    print(f"[selftest] Q_ZUP: sub9 template joints vs omomo.xml rest skeleton: {ax_err*100:.2f} cm")
    assert ax_err < 0.005, "Q_ZUP does not map the SMPL template onto the MJCF skeleton"

    clip = sorted(glob.glob("InterAct/OMOMO_new/sub2_*.pt"))[0]
    x = torch.load(clip, map_location="cpu", weights_only=False).detach().numpy()
    M = p._model("male")
    errs = []
    for t in range(0, min(len(x), 90), 30):
        v, f = p.pose_from_dof("sub2", x[t, 9:162], x[t, 0:3], x[t, 3:7])
        Jz = (M["J_reg"] @ v)[p.b2s]
        errs.append(translated(Jz, x[t, 162:318].reshape(52, 3)))
    print(f"[selftest] posed SMPL-X mesh joints vs stored body_pos (translation only): "
          f"mean {np.mean(errs)*100:.2f} cm  frames {[round(e*100,1) for e in errs]}")
    print("  < ~6cm => pose transfer correct (residual = generic-MJCF vs subject-SMPL shape)")
    assert np.mean(errs) < 0.08, "pose transfer regressed"

    # With the subject's OWN MJCF the sim's joints and SMPL's are one skeleton, so
    # the native pose (SMPL's own FK) and global LBS (joints pinned where the sim
    # put them) must agree, and both must land on the clip's stored body_pos.
    p2 = SMPLXPoser(mjcf=os.path.join(os.path.dirname(MJCF), "smplx_omomo_sub2.xml"))
    nat, own = [], []
    for t in range(0, min(len(x), 90), 30):
        v, _ = p2.pose_from_dof("sub2", x[t, 9:162], x[t, 0:3], x[t, 3:7])
        Jz = (M["J_reg"] @ v)[p2.b2s]
        own.append(np.linalg.norm(Jz - x[t, 162:318].reshape(52, 3), axis=1).mean())
        _, p_sim = p2.mjcf_fk(x[t, 9:162], x[t, 0:3], x[t, 3:7])
        pose_aa, trans = p2.native_from_dof("sub2", x[t, 9:162], x[t, 0:3], x[t, 3:7])
        nat.append(np.linalg.norm(p2.fk_joints("sub2", pose_aa, trans)[p2.b2s] - p_sim, axis=1))
    nat = np.array(nat)
    hands = [k for k in range(len(p2.body_order)) if k not in body]
    print(f"[selftest] sub2's own MJCF: regressed joints vs stored body_pos, no alignment: "
          f"mean {np.mean(own)*100:.2f} cm; native FK joints vs the sim's: body max "
          f"{nat[:, body].max()*100:.3f} cm, hands max {nat[:, hands].max()*100:.2f} cm "
          f"(the MJCF fingertip bodies sit ~3 cm off SMPL's last finger joints; expected)")
    assert np.mean(own) < 0.03, "pose transfer with the subject's own MJCF is off"
    assert nat[:, body].max() < 0.002, "native_pose_from_global does not invert the sim's body state"

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
    # Legacy path, opt-in only: 10 frames at 80 warm iterations fit to ~6 cm
    # (rigid-aligned) and a clean surface. Kept so --ik keeps working.
    assert np.mean(jfit) < 0.08 and max(emax) < 8, "IK retarget regressed"
    print("  OK -- the legacy position-only fit still runs (use the native path)")


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

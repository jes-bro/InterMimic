#!/usr/bin/env python3
"""Convert a CARI4D output bundle into InterAct's intermediate schema.

CARI4D (NVlabs CVPR 2026) produces per-clip .pth bundles with SMPL-H body
parameters and a 6-DoF object trajectory. InterAct's simulation/interact2mimic.py
converts InterAct-format clips into InterMimic's 591-channel .pt motion files
plus per-subject MJCFs.

This script is the missing adapter: it reads a CARI4D bundle and writes the two
.npz files (human + object) plus copies the Hunyuan3D mesh into the layout that
interact2mimic.py expects, using `dataset_name=behave_<tag>` so that
interact2mimic.py's BEHAVE branch (SMPL-H, num_betas=10, flat_hand_mean=False)
is selected — which exactly matches CARI4D's output schema.

Workflow:

    # 1. Local: run this adapter
    python scripts/cari4d_to_interact.py \\
        --bundle /path/to/<seq>.pth \\
        --mesh   /path/to/<seq>*_align.obj \\
        --interact-root /home/jess/interact/InterAct \\
        --gender male

    # 2. Cluster: run InterAct's converter against the new dataset
    cd /path/to/InterAct/simulation
    python interact2mimic.py --dataset_name behave_cari4d

    # 3. Cluster: replay in InterMimic
    cd /path/to/InterMimic
    sh isaacgym/scripts/data_replay_cari4d.sh
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


# Copied verbatim from CARI4D's lib_smpl/th_hand_prior.py to avoid taking a
# runtime dependency on the CARI4D module tree (which pulls in torchvision,
# smplx, human_body_prior, etc.).
GRAB_MEAN_HAND = np.array([
    0.13566974,  0.09491789, -0.28316078, -0.06223104, -0.0483653,
   -0.39977205,  0.13620542, -0.13199732, -0.3829936,  -0.21186522,
    0.07707776, -0.5384531,   0.10212211, -0.01378017, -0.49732804,
   -0.0471581,  -0.08448984, -0.1955775,  -0.58500576, -0.1548803,
   -0.47505018,  0.17948975, -0.13303751, -0.24022132, -0.3436518,
    0.11407528, -0.02665429, -0.23750143, -0.07435384, -0.4635036,
   -0.07951606, -0.07775243, -0.43911096, -0.19834545, -0.03837305,
   -0.22386047,  0.74066657,  0.3301243,  -0.11117966, -0.4979891,
    0.00626109,  0.1454768,   0.62785035, -0.01757009, -0.16062371,
    0.16868931, -0.12404376,  0.35450554, -0.04718762,  0.04999495,
    0.4440688,   0.13983883,  0.14151372,  0.37325338, -0.21371473,
   -0.14219724,  0.5842063,   0.11580209,  0.0260711,   0.55343974,
   -0.07212783,  0.09037765,  0.21028592, -0.6847437,  -0.00735493,
    0.5761462,   0.3632393,   0.18621148,  0.3402348,  -0.57334983,
   -0.13106765, -0.03578933, -0.291134,    0.003825,    0.5634436,
   -0.10148321,  0.09694234,  0.47672924, -0.22845045,  0.04699614,
    0.26392558,  0.8213351,  -0.2821158,   0.1008013,  -0.6013597,
   -0.02904042,  0.01898805,  0.733293,    0.08564732,  0.02389174,
], dtype=np.float64)


def pose72to156(pose72: np.ndarray) -> np.ndarray:
    """Upcast SMPL 72-dim axis-angle pose to SMPL-H 156-dim, filling hand joints
    with the GRAB mean-hand prior. Matches CARI4D's lib_smpl.pose72to156."""
    if pose72.ndim != 2 or pose72.shape[-1] != 72:
        raise ValueError(f"pose72 must be (T, 72), got {pose72.shape}")
    pose156 = np.zeros((pose72.shape[0], 156), dtype=np.float64)
    pose156[:, 66:] = GRAB_MEAN_HAND
    pose156[:, :69] = pose72[:, :69]
    pose156[:, 69 + 45:69 + 48] = pose72[:, 69:72]
    return pose156


class _PermissiveUnpickler(pickle.Unpickler):
    """Replaces unresolvable CARI4D-internal classes (e.g. TrainState) with a
    no-op stub so we can load the bundle without CARI4D on PYTHONPATH."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            class _Stub:
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            _Stub.__name__ = name
            _Stub.__module__ = module
            return _Stub


def _load_bundle(path: Path) -> dict:
    class _PickleModule:
        Unpickler = _PermissiveUnpickler

        @staticmethod
        def load(fh, **kw):
            return _PermissiveUnpickler(fh).load()

    with path.open("rb") as fh:
        return torch.load(fh, map_location="cpu", weights_only=False,
                          pickle_module=_PickleModule)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="CARI4D .pth bundle (with gt/pr/in sub-dicts)")
    parser.add_argument("--mesh", type=Path, required=True,
                        help="Path to the *_align.obj that pose_abs is calibrated to "
                             "(typically the first sorted _align.obj under "
                             "<hy3d_meshes_root>/<seq>*/).")
    parser.add_argument("--interact-root", type=Path, required=True,
                        help="Path to your InterAct clone (must contain simulation/interact2mimic.py).")
    parser.add_argument("--dataset-tag", default="behave_cari4d",
                        help="InterAct dataset_name folder. Must start with 'behave_' to "
                             "trigger interact2mimic.py's BEHAVE branch (SMPL-H, num_betas=10). "
                             "Default: behave_cari4d.")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], required=True,
                        help="SMPL-H gender used during CARI4D reconstruction.")
    parser.add_argument("--object-name", default=None,
                        help="Object slug (URDF + folder name). Default: token at "
                             "split('_')[-2] of the bundle filename stem.")
    parser.add_argument("--subject-id", type=int, default=99,
                        help="Integer subject id to embed in the output filename. "
                             "InterMimic's task code (intermimic.py:63) parses "
                             "split('_')[0][3:] as int, so the seq prefix must be "
                             "'sub<int>'. Default: 99.")
    parser.add_argument("--clip-idx", type=int, default=0,
                        help="Sequence index suffix for this clip (default: 0). "
                             "Increment when converting more clips of the same subject.")
    parser.add_argument("--seq-name", default=None,
                        help="Override the sequence directory name entirely. "
                             "Default: 'sub<subject_id>_<object_name>_<clip_idx:03d>'.")
    parser.add_argument("--bundle-key", default="pr", choices=["pr", "gt", "in"],
                        help="Which sub-dict to read. 'pr' (default) = optimized prediction "
                             "(what you almost always want).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Source video FPS (default 30).")
    parser.add_argument("--prerotate-x", type=float, default=0.0,
                        help="Degrees to pre-rotate the SMPL pose + object pose "
                             "around the X axis before writing. NOT RECOMMENDED — "
                             "interferes with interact2mimic.py's upright_start "
                             "correction and produces broken joint angles. Default "
                             "0 (disabled). Post-process via rotate_pt.py instead.")
    args = parser.parse_args()

    bundle_path = args.bundle.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    interact_root = args.interact_root.expanduser().resolve()

    if not bundle_path.is_file():
        print(f"[cari4d->interact] bundle not found: {bundle_path}", file=sys.stderr)
        return 2
    if not mesh_path.is_file():
        print(f"[cari4d->interact] mesh not found: {mesh_path}", file=sys.stderr)
        return 2
    if not (interact_root / "simulation" / "interact2mimic.py").is_file():
        print(f"[cari4d->interact] --interact-root does not look like an InterAct clone "
              f"(missing simulation/interact2mimic.py): {interact_root}", file=sys.stderr)
        return 2
    if not args.dataset_tag.lower().startswith("behave"):
        print(f"[cari4d->interact] WARNING: --dataset-tag '{args.dataset_tag}' does not start "
              f"with 'behave'; interact2mimic.py will NOT use the SMPL-H/num_betas=10 branch.",
              file=sys.stderr)

    # Derive object name from the bundle filename (Date03_Sub01_gas_wild002 → 'gas')
    # before composing the InterMimic-style seq name.
    bundle_stem_parts = bundle_path.stem.split("_")
    if args.object_name:
        object_name = args.object_name
    elif len(bundle_stem_parts) >= 2:
        object_name = bundle_stem_parts[-2]
    else:
        print(f"[cari4d->interact] cannot infer object from bundle '{bundle_path.name}'; "
              f"pass --object-name", file=sys.stderr)
        return 2

    # InterMimic expects 'sub<int>_<object>_<idx>.pt' (intermimic.py:63 does
    # int(split('_')[0][3:])). CARI4D's native sequence names like
    # 'Date03_Sub01_gas_wild002' don't fit. Compose a conforming name.
    seq_name = args.seq_name or f"sub{args.subject_id}_{object_name}_{args.clip_idx:03d}"

    print(f"[cari4d->interact] loading {bundle_path.name} (key={args.bundle_key})")
    bundle = _load_bundle(bundle_path)
    if args.bundle_key not in bundle:
        print(f"[cari4d->interact] bundle missing key '{args.bundle_key}'; "
              f"got {list(bundle)}", file=sys.stderr)
        return 2
    src = bundle[args.bundle_key]
    for k in ("smpl_pose", "smpl_t", "betas", "pose_abs"):
        if k not in src:
            print(f"[cari4d->interact] bundle['{args.bundle_key}'] missing '{k}'; "
                  f"got {list(src)}", file=sys.stderr)
            return 2

    smpl_pose = src["smpl_pose"].detach().cpu().numpy()
    smpl_t = src["smpl_t"].detach().cpu().numpy()
    betas_all = src["betas"].detach().cpu().numpy()
    pose_abs = src["pose_abs"].detach().cpu().numpy()

    T = smpl_pose.shape[0]
    if smpl_t.shape != (T, 3):
        raise SystemExit(f"smpl_t shape {smpl_t.shape} incompatible with T={T}")
    if pose_abs.shape != (T, 4, 4):
        raise SystemExit(f"pose_abs shape {pose_abs.shape} incompatible with T={T}")

    if smpl_pose.shape[1] == 72:
        poses = pose72to156(smpl_pose.astype(np.float64))
    elif smpl_pose.shape[1] == 156:
        poses = smpl_pose.astype(np.float64)
    else:
        raise SystemExit(f"unexpected smpl_pose width {smpl_pose.shape[1]}; want 72 or 156")

    beta = betas_all[0].astype(np.float64)
    if betas_all.shape[1] != 10:
        raise SystemExit(f"expected 10 betas (BEHAVE branch), got {betas_all.shape[1]}")
    if not np.allclose(betas_all, betas_all[0:1], atol=1e-6):
        drift = float(np.max(np.abs(betas_all - betas_all[0:1])))
        print(f"[cari4d->interact] note: betas not constant across frames "
              f"(max drift {drift:.4g}); using frame 0")

    obj_trans = pose_abs[:, :3, 3].astype(np.float64)
    obj_angles = sRot.from_matrix(pose_abs[:, :3, :3]).as_rotvec().astype(np.float64)

    # Pre-rotate around X to convert CARI4D's camera-frame convention to the
    # Y-up SMPL frame that interact2mimic.py expects. Applied identically to
    # human root + object so their relative geometry stays consistent.
    if args.prerotate_x != 0.0:
        R_pre = sRot.from_euler("x", args.prerotate_x, degrees=True)
        print(f"[cari4d->interact] pre-rotating {args.prerotate_x}° around X")

        # Human root orient (first 3 dims of poses) — rotate as rotvec
        root_orient = sRot.from_rotvec(poses[:, :3])
        poses[:, :3] = (R_pre * root_orient).as_rotvec()

        # Human translation
        smpl_t = R_pre.apply(smpl_t)

        # Object axis-angle
        obj_orient = sRot.from_rotvec(obj_angles)
        obj_angles = (R_pre * obj_orient).as_rotvec()

        # Object translation
        obj_trans = R_pre.apply(obj_trans)

    seq_dir = interact_root / "data" / args.dataset_tag / "sequences_canonical" / seq_name
    obj_dir = interact_root / "data" / args.dataset_tag / "objects" / object_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    human_npz = seq_dir / "human.npz"
    object_npz = seq_dir / "object.npz"
    mesh_dst = obj_dir / f"{object_name}.obj"

    np.savez(human_npz,
             poses=poses.astype(np.float32),
             trans=smpl_t.astype(np.float32),
             beta=beta.astype(np.float32),
             gender=np.array(args.gender),
             fps=np.float32(args.fps))
    np.savez(object_npz,
             angles=obj_angles.astype(np.float32),
             trans=obj_trans.astype(np.float32),
             name=np.array(object_name))

    if mesh_dst.resolve() != mesh_path:
        # Strip mtllib/usemtl references from the .obj as we copy. Hunyuan3D
        # meshes ship with a sibling .mtl that isn't present at Isaac Gym's
        # asset-resolution path, and Isaac Gym refuses to load a .obj whose
        # mtllib it can't find. We only need geometry, not materials.
        with mesh_path.open("r") as src, mesh_dst.open("w") as dst:
            for line in src:
                stripped = line.lstrip()
                if stripped.startswith(("mtllib ", "usemtl ")):
                    continue
                dst.write(line)

    print(f"[cari4d->interact] seq={seq_name} object={object_name} T={T}")
    print(f"  wrote {human_npz}")
    print(f"        poses {poses.shape}, trans {smpl_t.shape}, beta {beta.shape}, "
          f"gender={args.gender}, fps={args.fps}")
    print(f"  wrote {object_npz}")
    print(f"        angles {obj_angles.shape}, trans {obj_trans.shape}")
    print(f"  wrote {mesh_dst}")
    print()
    print("Next step (on the machine that can run InterAct + Isaac Gym):")
    print(f"  cd {interact_root}/simulation")
    print(f"  python interact2mimic.py --dataset_name {args.dataset_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write a simulator rollout as a CARI4D prediction file, for nvdiffrast figures.

render_mesh_replay.py draws with matplotlib: no z-buffer, no camera model, flat
shading. It runs anywhere and is fine for checking a retarget, and wrong for a
figure -- polygons paint in depth order, so a hand crossing the torso can sink
into it.

CARI4D already has the renderer worth using. tools/viz_pred.py rasterises with
nvdiffrast through the real camera intrinsics, composites over the source video,
and puts a novel view beside it. It just expects its own prediction format,
expressed in the filming camera's frame. This converts a rollout into that.

    python scripts/sim_to_cari4d_bundle.py \\
        --dump renders/sub100_bball_000_rollout.npz \\
        --bundle <cari4d>/output/opt/.../<seq>.pth \\
        --pt InterAct/behave_cari4d/sub100_bball_000.pt \\
        --betas scripts/cari4d_betas.npz --subject sub100 \\
        --out <cari4d>/output/sim/<seq>.pth

Then render it in the CARI4D repo and environment, which is where nvdiffrast is:

    python tools/viz_pred.py -pf <that file> --wild_video --kid 0 \\
        --video <aligned>.mp4 --out_root output/viz-sim

The frame change is the substance here. The rollout is in the simulator's world,
which retargeting, an upright flip and a floor re-fit have moved away from the
camera's. Rather than transform fitted SMPL parameters -- which needs the rest
pose's root offset and is easy to get subtly wrong -- the JOINT POSITIONS are
transformed first and the fit runs on those, so it produces camera-frame
parameters directly. The transform itself is recovered by aligning the two
descriptions of the same trajectory.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_module(name, path):
    """Import a sibling script by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rigid_fit(src, dst):
    """Return (R, t, rms) mapping src onto dst by rotation and translation only.

    Kabsch. Scale is deliberately not fitted: both frames are metric, so a scale
    away from 1 means something is wrong upstream and should surface in the
    residual rather than be absorbed here.
    """
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - R @ src_c
    rms = float(np.sqrt((((src @ R.T + t) - dst) ** 2).sum(axis=1).mean()))
    return R, t, rms


def quat_to_mat(q):
    """Convert (x, y, z, w) quaternions, shape (..., 4), to rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def load_bundle(path):
    """Load a CARI4D bundle, reusing cari4d_to_interact's permissive unpickler.

    Imported rather than reimplemented. The stub it substitutes for an
    unimportable class has to accept __setstate__; returning a bare dict looks
    equivalent and fails inside torch's unpickler, which is what a local copy of
    this got wrong.

    Raises:
        SystemExit: if the file is missing.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"no bundle at {path}")
    helper = load_module("cari4d_to_interact",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "cari4d_to_interact.py"))
    return helper._load_bundle(Path(path))


def parse_args():
    """Parse the rollout, the reconstruction it came from, and where to write."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, required=True,
                        help="rollout npz from DUMP_TRAJ")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="the CARI4D .pth the clip was built from; supplies "
                            "the camera frame, the betas and the gt/in panels")
    parser.add_argument("--pt", type=Path, required=True,
                        help="the installed motion tensor, to recover the frame "
                             "change between the bundle and the simulator")
    parser.add_argument("--betas", type=Path,
                        default=Path("scripts/cari4d_betas.npz"))
    parser.add_argument("--subject", default=None,
                        help="betas key (default: read from the dump)")
    parser.add_argument("--models", default=None,
                        help="SMPL model directory (default: $SMPLX_MODELS)")
    parser.add_argument("--model-type", default="smplh",
                        choices=["smplh", "smplx"])
    parser.add_argument("--ik-iters", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    """Fit the rollout in the camera's frame and write it as a prediction file."""
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    dump = np.load(args.dump, allow_pickle=True)
    body_pos = dump["body_pos"].astype(np.float64)          # (T, 52, 3), sim world
    obj_pos = dump["obj_pos"].astype(np.float64)            # (T, 3)
    obj_rot = dump["obj_rot"].astype(np.float64)            # (T, 4) xyzw
    print(f"rollout: {len(body_pos)} frames")

    bundle = load_bundle(str(args.bundle))
    if "pr" not in bundle:
        raise SystemExit(f"bundle has no 'pr'; got {list(bundle)}")
    ref_t = bundle["pr"]["smpl_t"].detach().cpu().numpy().astype(np.float64)

    pt = torch.load(str(args.pt), map_location="cpu", weights_only=False)
    sim_root = pt[:, 0:3].numpy().astype(np.float64)
    if len(sim_root) != len(ref_t):
        raise SystemExit(f"the bundle has {len(ref_t)} frames and the motion "
                         f"tensor {len(sim_root)}; these are not one clip")

    # Recover the frame change, then invert it: the fit wants targets in the
    # camera's frame, which is where the renderer's intrinsics apply.
    R_bs, t_bs, rms = rigid_fit(ref_t, sim_root)
    print(f"bundle -> sim alignment: RMS {rms * 100:.1f} cm over {len(ref_t)} frames")
    if rms > 0.15:
        raise SystemExit(
            f"{rms * 100:.0f} cm is too large for the same motion in two frames. "
            f"Refusing to write a bundle whose frame change is not trustworthy.")
    R_sb = R_bs.T
    t_sb = -R_sb @ t_bs

    if len(body_pos) != len(ref_t):
        n = min(len(body_pos), len(ref_t))
        print(f"rollout is {len(body_pos)} frames against {len(ref_t)}; "
              f"using the first {n}")
        body_pos, obj_pos, obj_rot = body_pos[:n], obj_pos[:n], obj_rot[:n]

    joints_cam = body_pos @ R_sb.T + t_sb

    poser_mod = load_module("smplx_pose", os.path.join(here, "smplx_pose.py"))
    subject = args.subject
    if subject is None:
        raw = str(dump["subject"])
        import re
        found = [c for c in re.findall(r"sub\d+", raw)]
        subject = found[-1] if found else raw
        print(f"subject from the dump: {raw!r} -> {subject!r}")

    P = poser_mod.SMPLXPoser(models_dir=args.models, betas_path=str(args.betas),
                             model_type=args.model_type)
    if subject not in P.betas.files:
        raise SystemExit(f"{subject!r} has no betas entry in {args.betas}")

    print(f"fitting {len(joints_cam)} frames in the camera's frame...")
    poses, trans = P.fit_sequence(subject, joints_cam, iters_warm=args.ik_iters,
                                  verbose=False)
    poses = np.asarray(poses, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    print(f"fitted poses {poses.shape}, trans {trans.shape}")

    # CARI4D stores smpl_pose flat and distinguishes 72 from 156 by width:
    # viz_pred.py:196 does pose72to156(x) if x.shape[1] == 72 else x. A (T, J, 3)
    # array passes that test unchanged and reaches the body model with the wrong
    # rank, so flatten here. SMPL-H's 52 joints give exactly the 156 expected.
    if poses.ndim == 3:
        poses = poses.reshape(len(poses), -1)
        print(f"flattened to {poses.shape} for the CARI4D format")
    if poses.shape[1] not in (72, 156):
        raise SystemExit(
            f"smpl_pose came out {poses.shape[1]} wide; CARI4D expects 72 or "
            f"156, and anything else is read as one of them")

    # The object: rotate and translate its pose into the same frame.
    pose_abs = np.tile(np.eye(4), (len(obj_pos), 1, 1))
    pose_abs[:, :3, :3] = R_sb @ quat_to_mat(obj_rot)
    pose_abs[:, :3, 3] = obj_pos @ R_sb.T + t_sb

    n = len(poses)

    def keep(value):
        """Whether a bundle field is plain data that can be written back out.

        The bundle also holds training bookkeeping -- optimizer, scheduler,
        train_state -- which the permissive loader reconstructs as local stub
        classes. Those cannot be pickled again, and nothing about a render wants
        them.
        """
        if isinstance(value, (torch.Tensor, np.ndarray, list, tuple, dict,
                              str, int, float, bool)) or value is None:
            return True
        return False

    def trim(block):
        """Cut a bundle sub-dict to the rollout's length, keeping plain data."""
        out, dropped = {}, []
        for key, value in block.items():
            if not keep(value):
                dropped.append(f"{key}({type(value).__name__})")
                continue
            out[key] = value[:n] if hasattr(value, "__len__") and len(value) >= n \
                else value
        if dropped:
            print(f"dropped non-data fields: {', '.join(dropped)}")
        return out

    # Start from the original prediction and replace only what the simulator
    # changed. viz_pred reads more than the four pose fields -- 'frames' among
    # them -- and enumerating those would leave the next one to be discovered by
    # a failed render.
    out_pr = trim(bundle["pr"])
    out_pr.update({
        "smpl_pose": torch.from_numpy(poses).float(),
        "smpl_t": torch.from_numpy(trans).float(),
        "pose_abs": torch.from_numpy(pose_abs).float(),
    })
    print(f"prediction keys: {sorted(out_pr)}")

    # gt and in are carried through so viz_pred's other panels still populate;
    # only the prediction is replaced with what the simulator did.
    result = {"pr": out_pr,
              "gt": trim(bundle.get("gt", bundle["pr"])),
              "in": trim(bundle.get("in", bundle["pr"]))}

    # viz_pred.py:212 decides which object mesh to load by looking for "hy3d"
    # in the prediction file's PATH: with it, the reconstructed mesh; without
    # it, a BEHAVE template under a path that does not exist here. Since a
    # reconstructed clip always uses the former, say so rather than let the
    # render fail on someone else's home directory.
    if "hy3d" not in str(args.out):
        print(f"WARNING: '{args.out}' has no 'hy3d' in its path, so viz_pred "
              f"will look for a BEHAVE object template instead of the "
              f"reconstructed mesh. Put 'hy3d' in the directory or filename.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(args.out))
    print(f"wrote {args.out}")
    print()
    print("Render it in the CARI4D repo, where nvdiffrast lives:")
    print(f"  python tools/viz_pred.py -pf {args.out} --wild_video --kid 0 \\")
    print(f"      --video <aligned>.0.color.mp4 --out_root output/viz-sim")
    return 0


if __name__ == "__main__":
    sys.exit(main())

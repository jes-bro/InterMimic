#!/usr/bin/env python3
"""Convert a DUMP_TRAJ rollout into the InterAct layout that renders a skinned
SMPL surface (InterAct/visualization/visualize.py, pyrender).

A rollout is the SIMULATOR's output -- per-joint global rotations and positions
-- while InterAct wants SMPL PARAMETERS. The gap is the IK fit in
smplx_pose.fit_sequence, the same one scripts/render_mesh_replay.py uses; this
script wraps it and writes the dataset layout instead of a matplotlib mp4.

    python3 scripts/dump_to_interact.py ~/Downloads/novice_demo_t018c/novice_demo_t018c.npz \\
        --betas_from_bundle ~/Downloads/novice_Sub204/Date170_Sub204_bball_t004a/Date170_Sub204_bball_t004a.pth \\
        --gender male \\
        --out_root ~/interact/InterAct/data/behave_cari4d \\
        --clip Sub204_sim_t018c \\
        --object_mesh ~/Downloads/novice_Sub204/Date170_Sub204_bball_t004a/Date170_Sub204_bball_t004a_067_align.obj

FRAMES. A rollout is Z-UP (measured on novice_demo_t018c: pelvis at 0.903 m,
travel along y), InterAct's renderer is Y-UP. GYM_TO_YUP is that change of
basis. As in CARI4D's tools/bundle_to_interact.py, the root translation is
rotated ABOUT THE PELVIS rather than the origin: SMPL applies global_orient at
the pelvis joint, so rotating orientation and translation independently
displaces the body by (I - R) @ J0 and lifts the object out of the hands.

BETAS. --betas is a subject-keyed npz like scripts/omomo_betas.npz.
--betas_from_bundle instead takes the shape straight out of a CARI4D result
bundle, which is how sub204 is available at all: it is a reconstructed EgoExo4D
novice and is in none of the packaged betas files.

OBJECT. The rollout carries the object's pose but not its mesh, so
--object_mesh says which mesh those poses drive. Rotations are (x, y, z, w),
Isaac Gym's order, matching render_mesh_replay.quat_to_mat.
"""
import argparse
import importlib.util
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.append(os.getcwd())

# Rollouts are in the sim frame (x forward, y left, z up); InterAct's renderer is
# SMPL's native frame (x left, y up, z forward). That is Q_ZUP^T: sim (x, y, z)
# -> (y, z, x). Corrected 2026-10-21 together with Q_ZUP; the old (x, z, -y)
# was the transpose of the old, 90-degree-yawed matrix.
GYM_TO_YUP = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0]])


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that stubs classes it cannot import (the CARI4D training stack)."""

    def find_class(self, module, name):
        """Resolve normally; on ImportError/AttributeError return a stub."""
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (object,), {'__reduce__': lambda self: (dict, ((),))})


class _TolerantPickleModule:
    """A pickle-module stand-in for torch.load."""

    Unpickler = _TolerantUnpickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)
    Pickler = pickle.Pickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def load_poser(models, betas_path, model_type='smplh'):
    """Build SMPLXPoser from scripts/smplx_pose.py by path, as render_mesh_replay does."""
    spec = importlib.util.spec_from_file_location(
        'smplx_pose', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smplx_pose.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SMPLXPoser(models_dir=models, betas_path=betas_path, model_type=model_type)


def betas_from_bundle(pth, subject, gender, out_npz, width=16):
    """Write a one-subject betas npz from a CARI4D bundle's averaged shape.

    SMPLXPoser reads a subject-keyed npz with a '_genders' entry, so the shape
    that CARI4D reconstructed for this person is repackaged into that form
    rather than the subject being added to a shipped betas file.

    CARI4D works in 10 betas; the SMPL+H release carries 16, and while _shape
    slices to the model width, the torch path multiplies shapedirs (n, 3, 16) by
    the raw vector and fails on a 10-wide one. Zero-padding to 16 is the same
    body -- the extra coefficients contribute nothing -- and is the width the
    shipped omomo_betas.npz uses. The returned 10-wide vector is what goes into
    the InterAct npz, whose renderer builds its model with num_betas=10.
    """
    import torch
    d = torch.load(os.path.expanduser(pth), map_location='cpu', weights_only=False,
                   pickle_module=_TolerantPickleModule)
    betas = d['pr']['betas'].detach().float().numpy().reshape(-1, 10).mean(0)
    padded = np.zeros(width, dtype=np.float32)
    padded[:len(betas)] = betas
    np.savez(out_npz, _genders=np.array([f'{subject}:{gender}']),
             **{subject: padded})
    return out_npz, betas


def quat_to_mat(q):
    """(T, 4) Isaac Gym (x, y, z, w) quaternions -> (T, 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def mjcf_body_order(mjcf):
    """The MJCF body names in document order, which indexes body_rot/body_pos."""
    import re as _re
    with open(mjcf) as f:
        return _re.findall(r'body name="([^"]+)"', f.read())


# SMPL-H joints the position-only IK sees poorly or not at all, taken from the
# rollout instead of being optimised. Each is (smpl joint, MJCF parent, MJCF body).
#
# LEAVES have nothing below them, so their rotation moves no joint at all and
# the fit leaves them at identity: on novice_demo_t018c the head and both toes
# came out with std exactly 0.000000 -- a frozen head and locked ankles.
#
# The SPINE chain is the partial case: Torso/Spine/Chest are near-collinear, so
# rotation ABOUT that axis barely moves anything downstream and the fit is free
# to choose it. That is the torso twisted off to one side. Bend is observable
# and would be fitted fine; the twist is what has to come from elsewhere.
#
# Safe because a rotation RELATIVE TO ITS PARENT is convention-free wherever the
# pair shares a rest orientation -- true up the trunk and at neck/head and
# ankle/toe. The documented MJCF-vs-SMPL rest mismatch is ~90 degrees at the
# ARMS, which is exactly why the arms are left to the fit.
LEAF_FILLS = [(15, 'Neck', 'Head'),
              (10, 'L_Ankle', 'L_Toe'),
              (11, 'R_Ankle', 'R_Toe')]
SPINE_FILLS = [(3, 'Pelvis', 'Torso'),
               (6, 'Torso', 'Spine'),
               (9, 'Spine', 'Chest'),
               (12, 'Chest', 'Neck')]


# SMPL's pose space is its own native Y-up; the rollout is Z-up. A relative
# rotation is free of the PARENT's orientation but is still written in world
# axes, so it has to be conjugated into the target frame or its components are
# permuted -- a head that nods comes out shaking instead, which is why the chin
# never moved. This is smplx_pose.Q_ZUP: sim = Q_ZUP @ smpl, SMPL's (left, up,
# forward) to the sim's (forward, left, up). Corrected 2026-10-21 (the old matrix
# was yawed 90 degrees; tests/test_smplx_axes.py pins the two copies equal).
Q_ZUP = np.array([[0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

# Per-joint weights on the pose prior, {smpl joint: weight}. A flat prior stops
# the fit drifting but not from reaching a target THROUGH a joint that should
# barely move, because the joint below it can serve the same target. Measured
# against a CARI4D reconstruction of this same person, the fitted collars ran
# 1.8x the reconstruction's range (54 degrees, where a real collar moves a few),
# the spine and hips about 2x. Those are the joints whose excess lands in the
# SURFACE rather than the skeleton -- an arched back, a distorted seat -- since
# linear blend skinning spreads a trunk rotation across the whole midsection.
# Weighting them brought collars and spine to at or below the reconstruction's
# range for 0.9 mm of extra joint residual.
REG_WEIGHTS = {13: 25.0, 14: 25.0,      # collars
               3: 8.0, 6: 8.0, 9: 8.0,  # spine1/2/3
               1: 12.0, 2: 12.0}        # hips

# HINGES: {smpl joint: the one axis it may turn on}. A knee or elbow bends about
# a single axis; the other two are anatomically near-zero, and a per-JOINT prior
# cannot tell them apart. Left free, the fit reaches an ankle or wrist target by
# swinging the shin or forearm sideways -- measured at 0.358 mean on the knee's
# abduction axis against a CARI4D reconstruction's 0.098, peaking at 0.79 rad,
# which is 45 degrees of sideways knee. Axis 0 is flexion, confirmed on that
# reconstruction: 0.57 mean there against 0.14 and 0.10 on the other two.
# The hinge axis DIFFERS between knees and elbows in SMPL's joint frames, so it
# is read off real data rather than assumed: in both the novice's and the
# expert's CARI4D reconstructions the knee's dominant axis is 0 (0.571 / 0.477
# mean, against 0.15 and 0.09 on the others) while the elbow's is 1 (1.584 /
# 0.813). Assuming axis 0 for elbows too put the penalty on the elbow's own
# bend and all but froze it: 0.131 mean against the reconstruction's 1.584.
HINGE_AXIS = {4: 0, 5: 0,        # knees  -> flexion on axis 0
              18: 1, 19: 1}      # elbows -> flexion on axis 1
HINGE_OFF_AXIS_WEIGHT = 40.0

# Global scale on every weight above, calibrated so the fitted pose MATCHES the
# CARI4D reconstruction of this same person rather than merely being smaller
# than it. Measured as a ratio to the reconstruction's per-axis mean (1.00 is
# the target), on 40 frames:
#
#   scale  residual  R_knee abd  L_knee abd  collar   hip    spine2
#   0.00    0.0261     3.82x       4.29x     2.21x   1.34x   1.23x
#   0.10    0.0270     1.42x       1.90x     1.25x   1.23x   1.22x
#   0.25    0.0300     0.86x       1.35x     0.81x   1.12x   1.20x   <- closest
#   0.50    0.0324     0.52x       0.81x     0.54x   1.01x   1.18x
#   1.00    0.0349     0.35x       0.46x     0.37x   0.84x   1.02x
#
# Full strength overshoots: it makes the knee three times STIFFER than the real
# person's, which is its own artefact. A weight of 1.0 is applied as
# 1 + (w - 1) * scale, so scale 0 is a flat prior.
REG_SCALE = 0.25

# Per-TARGET weights on the MJCF bodies, by name. The fit treats a centimetre
# the same everywhere, but the trunk segments are ~13 cm against a 38 cm thigh,
# so the residual that is invisible in a leg swings the back by tens of degrees.
# Measured against the ROLLOUT'S OWN trunk angles -- no other clip involved --
# the fitted back came out 14 to 31 degrees straighter than the motion it was
# fitting, at every prior setting including a flat one. Weighting the short
# segments up fixes that and costs nothing:
#
#   trunk_w  residual   Torso   Spine   Chest
#      1      0.0266   +14.4   +31.0    -9.5
#     10      0.0242    +3.9   +16.7    -5.7
#     40      0.0278    +3.4   +11.4    -3.1   <- used
TRUNK_BODIES = ['Pelvis', 'Torso', 'Spine', 'Chest', 'Neck', 'Head']
TRUNK_TARGET_WEIGHT = 40.0


def leaf_pose_from_dump(body_rot, order, parent, child, conjugate=True):
    """A leaf joint's local rotation as (T, 3) axis-angle, from the sim's state.

    The rollout records every body's global rotation, and the rotation of a
    body RELATIVE to its parent drops the parent's orientation -- so it carries
    over wherever the pair shares a rest orientation, true for neck/head and
    ankle/toe. It still needs conjugating by Q_ZUP to be expressed in SMPL's
    own axes. (The documented MJCF-vs-SMPL rest mismatch is ~90 degrees at the
    ARMS, which is why the body as a whole is fitted, not driven, from these.)
    """
    from scipy.spatial.transform import Rotation
    ip, ic = order.index(parent), order.index(child)
    Rp, Rc = quat_to_mat(body_rot[:, ip]), quat_to_mat(body_rot[:, ic])
    Rl = np.transpose(Rp, (0, 2, 1)) @ Rc
    if conjugate:
        Rl = Q_ZUP.T @ Rl @ Q_ZUP
    return Rotation.from_matrix(Rl).as_rotvec()


def pelvis_rest(models_dir, gender, betas):
    """The rest-pose pelvis for this shape, which the frame change rotates about."""
    import torch
    import smplx
    m = smplx.create(os.path.expanduser(models_dir), model_type='smplh',
                     gender=gender, use_pca=False, ext='pkl')
    out = m(betas=torch.from_numpy(np.asarray(betas, dtype=np.float32)[None, :]))
    return out.joints.detach().numpy()[0, 0].astype(np.float64)


def rotate_global_orient(R, rotvecs):
    """Compose a frame rotation onto (T, 3) global-orientation rotation vectors."""
    from scipy.spatial.transform import Rotation
    mats = Rotation.from_rotvec(rotvecs).as_matrix()
    return Rotation.from_matrix(R @ mats).as_rotvec()


def main():
    """Fit a rollout to SMPL-H and write it as an InterAct sequence."""
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', help='the DUMP_TRAJ .npz')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--clip', default=None, help='sequence name (default: the npz stem)')
    ap.add_argument('--betas', default=None, help='subject-keyed betas npz')
    ap.add_argument('--betas_from_bundle', default=None, help='a CARI4D result .pth')
    ap.add_argument('--gender', default='male', choices=['male', 'female'])
    ap.add_argument('--models', default='~/interact/InterAct/models')
    ap.add_argument('--smplh_models', default=None,
                    help='SMPL-H models for the poser (default: --models)')
    ap.add_argument('--object_mesh', default=None)
    ap.add_argument('--object_name', default=None)
    # Measured on novice_demo_t018c (40 frames, residual = fitted SMPL joints vs
    # the dump's own body_pos, whose bone lengths match SMPL-H exactly):
    #   smooth 0.15, 80 iters -> residual 0.132 m, L_knee pose std 0.020
    #   smooth 0.0,  80 iters -> residual 0.031 m, L_knee pose std 1.888
    #   smooth 0.0, 400 iters -> residual 0.007 m, L_knee pose std 1.617
    # Smoothing does not tame the spine, it damps EVERY joint: at 0.15 the knee
    # stops bending altogether. The targets are reachable, so the fix for a bad
    # fit is iterations, not regularisation.
    ap.add_argument('--iters_first', type=int, default=600)
    ap.add_argument('--iters_warm', type=int, default=300)
    ap.add_argument('--smooth', type=float, default=0.0,
                    help='fit_sequence temporal regulariser; see the note above -- '
                         'raising it freezes joints rather than calming the surface')
    ap.add_argument('--stride', type=int, default=1, help='take every Nth frame')
    ap.add_argument('--reg_scale', type=float, default=REG_SCALE,
                    help='scale on the per-joint/per-axis weights; calibrated so '
                         'the fit matches the reconstruction, not so it minimises')
    ap.add_argument('--flat_reg', action='store_true',
                    help='use one prior weight everywhere instead of REG_WEIGHTS')
    ap.add_argument('--no_head_fill', action='store_true',
                    help='leave the head at identity so it simply follows the neck; '
                         'its orientation is unobservable in the targets, so a '
                         'transferred value cannot be checked against anything')
    ap.add_argument('--no_leaf_fill', action='store_true',
                    help='leave head and toes at the IK result (frozen at identity)')
    # Pinning the spine to the rollout's relative rotations is OFF: measured,
    # the MJCF and SMPL rest frames differ enough that transferring rotations
    # between them leaves a 0.56 m joint residual even after conjugating into
    # SMPL's Y-up pose space, so those pins feed in mis-framed values. --reg
    # addresses the same twist honestly, by shrinking only what the targets do
    # not constrain.
    ap.add_argument('--spine_pin', action='store_true',
                    help='pin the spine chain to the rollout (mis-framed; see note)')
    ap.add_argument('--reg', type=float, default=1e-4,
                    help='L2 prior on pose, which bites only on components the '
                         'targets leave free -- hinge roll above all. Measured on '
                         '40 frames: reg 0 gave knee |max| 5.83 rad and elbow 8.44 '
                         '(twisted forearms, pinched knee); 1e-4 gave 1.65 and 2.98 '
                         'with a LOWER residual (0.0261 vs 0.0272); 1e-3 over-'
                         'constrains and the residual climbs to 0.0395.')
    ap.add_argument('--mjcf', default='isaacgym/src/intermimic/data/assets/smplx/omomo.xml',
                    help='MJCF whose body order indexes body_rot/body_pos')
    ap.add_argument('--fit_cache', default=None, help='where to cache the IK fit')
    ap.add_argument('--refit', action='store_true', help='ignore any cached fit')
    ap.add_argument('--no_flip', action='store_true')
    ap.add_argument('--ik', action='store_true',
                    help='fit the pose per frame from joint POSITIONS (the old fit_sequence '
                         'path, with its cache, pins and priors) instead of the exact native '
                         'pose from the dump\'s body_rot (smplx_pose.native_sequence). '
                         'Comparison only.')
    args = ap.parse_args()

    dump = os.path.abspath(os.path.expanduser(args.dump))
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    clip = args.clip or os.path.splitext(os.path.basename(dump))[0]
    d = np.load(dump, allow_pickle=True)
    for k in ('body_pos', 'obj_pos', 'obj_rot', 'subject'):
        if k not in d.files:
            raise SystemExit(f'{dump} has no {k!r}; keys are {d.files}')
    subject = str(d['subject'])
    body_pos = d['body_pos'][::args.stride]
    obj_pos = d['obj_pos'][::args.stride].astype(np.float64)
    obj_rot = d['obj_rot'][::args.stride]
    T = len(body_pos)
    print(f'[dump2interact] {clip}: subject {subject}, {T} frames '
          f'(stride {args.stride}), gender {args.gender}')

    if args.betas_from_bundle:
        betas_path = os.path.join(out_root, f'.betas_{subject}.npz')
        os.makedirs(out_root, exist_ok=True)
        betas_path, betas = betas_from_bundle(args.betas_from_bundle, subject,
                                              args.gender, betas_path)
        print(f'[dump2interact] betas from bundle: {np.round(betas, 3)}')
    elif args.betas:
        betas_path = os.path.expanduser(args.betas)
        betas = np.load(betas_path, allow_pickle=True)[subject]
    else:
        raise SystemExit('pass --betas or --betas_from_bundle')

    # The IK is minutes of CPU for a few hundred frames, and the frame handling
    # downstream of it is the part that gets iterated on, so the raw fit is
    # cached beside the dump and reused unless the frame count changes.
    # the smoothing weight changes the fit, so it belongs in the cache key --
    # otherwise raising --smooth silently reuses the pose it was meant to replace
    # pinning changes the fit as much as the smoothing weight does, so it is in
    # the key too -- otherwise a cached pre-pinning fit is silently reused
    cache = (args.fit_cache or os.path.splitext(dump)[0]
             + f'.fit{args.stride}_s{args.smooth:g}'
             + ('_pin' if args.spine_pin else '_nopin')
             + f'_r{args.reg:g}' + ('_flat' if args.flat_reg else f'_w{args.reg_scale:g}')
             + f'_t{TRUNK_TARGET_WEIGHT:g}_i{args.iters_warm}.npz')
    pose_aa = trans = None
    if not args.ik:
        # Exact: the dump's body rotations ARE SMPL's global joint rotations in
        # the sim's axes (smplx_pose.Q_ZUP), so the native pose follows without
        # any fit. No cache (it is instant), no pins or leaf fills (nothing is
        # unobservable), no priors.
        if 'body_rot' not in d.files:
            raise SystemExit(f'{dump} has no body_rot; the exact native pose needs the '
                             f'sim\'s body rotations. Pass --ik to fit from positions.')
        poser = load_poser(args.smplh_models or args.models, betas_path, 'smplh')
        print(f'[dump2interact] native pose from body_rot, {T} frames', flush=True)
        pose_aa, trans = poser.native_sequence(subject, d['body_rot'][::args.stride], body_pos)
    elif os.path.isfile(cache) and not args.refit:
        c = np.load(cache)
        if len(c['poses']) == T:
            pose_aa, trans = c['poses'], c['trans']
            print(f'[dump2interact] reusing fit from {cache}')
        else:
            print(f'[dump2interact] cache {cache} has {len(c["poses"])} frames, '
                  f'need {T}; refitting')
    if pose_aa is None:
        poser = load_poser(args.smplh_models or args.models, betas_path, 'smplh')
        order = mjcf_body_order(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', args.mjcf))
        br = d['body_rot'][::args.stride]
        fills = [f for f in LEAF_FILLS if not (args.no_head_fill and f[0] == 15)]
        fills += list(SPINE_FILLS) if args.spine_pin else []
        pinned = {}
        for j, parent, child in fills:
            pinned[j] = leaf_pose_from_dump(br, order, parent, child)
            print(f'[dump2interact] pin {child:8s} -> smpl joint {j:2d} '
                  f'(std {pinned[j].std(0).max():.3f})')
        rw = np.ones((52, 3))
        if not args.flat_reg:
            sc = args.reg_scale
            for j, w in REG_WEIGHTS.items():
                rw[j, :] = 1.0 + (w - 1.0) * sc
            for j, ax in HINGE_AXIS.items():
                off = [a for a in range(3) if a != ax]
                rw[j, off] = 1.0 + (HINGE_OFF_AXIS_WEIGHT - 1.0) * sc
        tw = np.ones(len(order))
        for n in TRUNK_BODIES:
            if n in order:
                tw[order.index(n)] = TRUNK_TARGET_WEIGHT
        print(f'[dump2interact] IK fitting {T} frames, {len(pinned)} joints pinned, '
              f'prior {"flat" if args.flat_reg else "per-joint"}...', flush=True)
        pose_aa, trans = poser.fit_sequence(subject, body_pos,
                                            iters_first=args.iters_first,
                                            iters_warm=args.iters_warm,
                                            smooth=args.smooth, pinned=pinned,
                                            reg=args.reg, reg_weights=rw,
                                            target_weights=tw)
        np.savez(cache, poses=np.asarray(pose_aa), trans=np.asarray(trans))
        print(f'[dump2interact] cached fit to {cache}')
    poses = np.asarray(pose_aa, dtype=np.float64).reshape(T, -1)
    if poses.shape[1] != 156:
        raise SystemExit(f'expected 156 pose channels for SMPL-H, got {poses.shape[1]}')
    trans = np.asarray(trans, dtype=np.float64)

    if args.ik and not args.no_leaf_fill:          # the native pose already has exact leaves
        order = mjcf_body_order(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', args.mjcf))
        br = d['body_rot'][::args.stride]
        for j, parent, child in [f for f in LEAF_FILLS
                                 if not (args.no_head_fill and f[0] == 15)]:
            aa = leaf_pose_from_dump(br, order, parent, child)
            poses[:, j * 3:j * 3 + 3] = aa
            print(f'[dump2interact] {child:8s} -> smpl joint {j:2d} from dump: '
                  f'|mean| {np.abs(aa).mean():.3f} std {aa.std(0).max():.3f} '
                  f'(IK left it at 0)')

    obj_mats = quat_to_mat(obj_rot)
    if not args.no_flip:
        # fit_sequence's poses are already in SMPL's NATIVE Y-up pose space:
        # verts_from_pose builds v_y natively, rotates it by Q_ZUP and only
        # THEN adds trans (smplx_pose.py:447), so the world is
        #   world_zup = Q_ZUP @ v_y(pose) + trans.
        # Viewing that same world Y-up is Q_ZUP^T applied throughout:
        #   world_yup = v_y(pose) + Q_ZUP^T @ trans
        # -- the POSES ARE UNCHANGED, and only the translations move. Rotating
        # global_orient here (as the CARI4D bundle path correctly does, since
        # that one really is in a camera frame) double-rotated the body against
        # the object and pushed the ball out of the hands.
        trans = trans @ GYM_TO_YUP.T
        obj_mats = GYM_TO_YUP @ obj_mats
        obj_pos = obj_pos @ GYM_TO_YUP.T

    from scipy.spatial.transform import Rotation
    obj_angles = Rotation.from_matrix(obj_mats).as_rotvec().astype(np.float32)
    obj = args.object_name or ''.join(c for c in clip.lower() if c.isalnum())

    seq_dir = os.path.join(out_root, 'sequences_canonical', clip)
    obj_dir = os.path.join(out_root, 'objects', obj)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)
    np.savez(os.path.join(seq_dir, 'human.npz'),
             poses=poses.astype(np.float32), betas=np.asarray(betas, dtype=np.float32),
             trans=trans.astype(np.float32), gender=args.gender)
    np.savez(os.path.join(seq_dir, 'object.npz'),
             angles=obj_angles, trans=obj_pos.astype(np.float32), name=obj)
    if args.object_mesh:
        shutil.copyfile(os.path.expanduser(args.object_mesh),
                        os.path.join(obj_dir, f'{obj}.obj'))
    else:
        print(f'[dump2interact] NOTE: no --object_mesh; put one at {obj_dir}/{obj}.obj')
    print(f'[dump2interact] wrote {seq_dir}/{{human,object}}.npz and objects/{obj}/')


if __name__ == '__main__':
    main()

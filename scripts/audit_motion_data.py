#!/usr/bin/env python3
"""Audit InterMimic motion clips for bad data, per subject AND per object.

Loads every sub<N>_<obj>_<idx>.pt in a motion dir, computes anomaly stats on the
591-channel reference, flags clips, and aggregates so bad SUBJECTS (body/motion)
and bad OBJECTS stand out -- no sim, no GPU, no video. Complements the kinematic
replay: numbers first, eyes second.

591-channel layout (from intermimic.py:_load_motion):
  0:3 root_pos | 9:162 dof_pos (51*3 joint angles) | 162:318 body_pos (52*3)
  318:321 obj_pos | ... (see rotate_pt.py for the rest)

Checks per clip (all axis-robust except where noted):
  nonfinite     any NaN/Inf frame                          -> corrupt
  pose_extent   max |body - root| (m): a human is <~1.5m   -> explosion if huge
  max_dof       max |joint angle| (rad): hinges bound +-pi -> broken retarget
  max_vel       max per-frame body displacement (m)        -> teleport/glitch
  min_body_obj  closest any body gets to the object (m)    -> no interaction if large

Usage:
  python scripts/audit_motion_data.py --motion-dir InterAct/OMOMO_new
  python scripts/audit_motion_data.py --subjects sub4 sub6
  python scripts/audit_motion_data.py --selftest      # PROVE the detector is correct
"""
import argparse
import glob
import os
import statistics
import sys
from collections import defaultdict

import torch

# Flag thresholds (deliberately loose -- catch gross corruption reliably; use the
# per-subject/object medians in the report to spot subtler outliers).
POSE_EXTENT_MAX = 3.0    # m   (human root->extremity is <~1.5m)
DOF_MAX         = 3.30   # rad (joint hinges are bounded +-pi ~= 3.14159)
VEL_MAX         = 0.50   # m per frame (at 30fps that's 15 m/s -- a gross teleport)
OBJ_DIST_MAX    = 1.00   # m   (if the closest body never gets within 1m, no interaction)
# Occlusion glitches snap a joint OUT and BACK over 1-2 frames: the per-frame
# velocity can stay UNDER VEL_MAX, but the acceleration (2nd difference of
# position) spikes. Smooth fast motion = high velocity, ~0 acceleration; a
# tracking glitch = the reverse. So acceleration is what catches the foot jitter
# you can see in the replay but velocity-thresholding misses.
ACC_MAX         = 0.15   # m/frame^2 -- a clip with any body this jerky is flagged 'jitter'
ACC_FLAG        = 0.10   # m/frame^2 -- per-(frame,body) accel above this counts as one jumpy sample


def clip_stats(x):
    """Return per-clip metrics + a list of flags for a (T, 591) tensor."""
    T = int(x.shape[0])
    if not bool(torch.isfinite(x).all()):
        return dict(T=T, finite=False, pose_extent=float('nan'), max_dof=float('nan'),
                    max_vel=float('nan'), min_body_obj=float('nan')), ['nonfinite']

    root = x[:, 0:3]                            # (T,3)
    dof  = x[:, 9:162]                          # (T,153) joint angles
    body = x[:, 162:318].reshape(T, 52, 3)      # (T,52,3)
    obj  = x[:, 318:321]                        # (T,3)

    rel = body - root.unsqueeze(1)                       # body relative to root
    pose_extent = float(rel.norm(dim=-1).max())          # furthest limb from root
    max_dof = float(dof.abs().max())
    max_vel = float((body[1:] - body[:-1]).norm(dim=-1).max()) if T > 1 else 0.0
    # acceleration = 2nd difference of body position; spikes on snap-and-return
    # glitches (occlusion) but stays ~0 for smooth motion, however fast.
    if T > 2:
        acc = (body[2:] - 2 * body[1:-1] + body[:-2]).norm(dim=-1)  # (T-2, 52)
        max_acc = float(acc.max())
        n_jitter = int((acc > ACC_FLAG).sum())   # how many (frame,body) samples snap
    else:
        max_acc, n_jitter = 0.0, 0
    min_body_obj = float((body - obj.unsqueeze(1)).norm(dim=-1).min())

    flags = []
    if pose_extent > POSE_EXTENT_MAX: flags.append('explosion')
    if max_dof > DOF_MAX:             flags.append('joint_limit')
    if max_vel > VEL_MAX:             flags.append('teleport')
    if max_acc > ACC_MAX:             flags.append('jitter')
    if min_body_obj > OBJ_DIST_MAX:   flags.append('no_interaction')
    return dict(T=T, finite=True, pose_extent=pose_extent, max_dof=max_dof,
                max_vel=max_vel, max_acc=max_acc, n_jitter=n_jitter,
                min_body_obj=min_body_obj), flags


def load_clip(path):
    x = torch.load(path, map_location='cpu')
    if isinstance(x, dict):
        x = x.get('hoi_data', next(iter(x.values())))
    return x.float()


def parse_name(fname):
    """sub<N>_<obj>_<idx>.pt -> (subject, object). Handles sub<src>to<tgt>_..."""
    base = os.path.basename(fname).rsplit('.', 1)[0]
    parts = base.split('_')
    return parts[0], parts[-2]          # subject token, object token (2nd-to-last)


def audit(motion_dir, subjects=None):
    files = sorted(glob.glob(os.path.join(motion_dir, '*.pt')))
    if subjects:
        keep = set(subjects)
        files = [f for f in files if parse_name(f)[0] in keep]
    if not files:
        print(f"no .pt clips found in {motion_dir}" + (f" for {subjects}" if subjects else ""))
        return 1

    per_subj = defaultdict(list)   # subject -> list of (stats, flags)
    per_obj  = defaultdict(list)
    flagged_clips = []
    for f in files:
        subj, obj = parse_name(f)
        try:
            st, fl = clip_stats(load_clip(f))
        except Exception as e:
            print(f"  ERROR loading {os.path.basename(f)}: {e}")
            continue
        per_subj[subj].append((st, fl)); per_obj[obj].append((st, fl))
        if fl:
            flagged_clips.append((os.path.basename(f), fl, st))

    def summarize(group, label):
        print(f"\n=== per {label} ({len(group)} {label}s) ===")
        print(f"{label:11} {'clips':>5} {'flag':>5} {'medext':>7} "
              f"{'maxacc':>7} {'jitfrm':>7} {'medobjd':>8} {'flags'}")
        rows = []
        for key, items in group.items():
            n = len(items)
            nflag = sum(1 for _, fl in items if fl)
            exts = [s['pose_extent'] for s, _ in items if s['finite']]
            accs = [s['max_acc'] for s, _ in items if s['finite']]
            jit  = sum(s['n_jitter'] for s, _ in items if s['finite'])
            objd = [s['min_body_obj'] for s, _ in items if s['finite']]
            allflags = sorted({f for _, fl in items for f in fl})
            rows.append((jit, nflag, key, n, exts, accs, jit, objd, allflags))
        # sort jitteriest-first so a glitchy subject/object floats to the top
        for _, nflag, key, n, exts, accs, jit, objd, allflags in sorted(rows, reverse=True):
            me = statistics.median(exts) if exts else float('nan')
            ma = max(accs) if accs else float('nan')
            mo = statistics.median(objd) if objd else float('nan')
            mark = ' <-- ' + ','.join(allflags) if allflags else ''
            print(f"{key:11} {n:>5} {nflag:>5} {me:>7.2f} {ma:>7.2f} {jit:>7} {mo:>8.2f}{mark}")

    summarize(per_subj, 'subject')
    summarize(per_obj, 'object')

    print(f"\n=== flagged clips ({len(flagged_clips)}) ===")
    for name, fl, st in flagged_clips[:40]:
        print(f"  {name:40} {','.join(fl):20} extent={st['pose_extent']:.1f} "
              f"dof={st['max_dof']:.2f} vel={st['max_vel']:.2f} objd={st['min_body_obj']:.2f}")
    if len(flagged_clips) > 40:
        print(f"  ... and {len(flagged_clips) - 40} more")
    print(f"\nTotal: {len(files)} clips, {len(flagged_clips)} flagged.")
    return 0


# --------------------------------------------------------------------------
# PROOF: inject known defects and assert the detector flags exactly them.
# --------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    T = 60

    def clean():
        x = torch.zeros(T, 591)
        # root drifts slowly; body within ~0.8m of root; small smooth motion
        t = torch.linspace(0, 1, T).unsqueeze(1)
        x[:, 0:3] = 0.5 * t                                  # root_pos slow drift
        offs = torch.randn(52, 3) * 0.3                      # body offsets from root (<~0.9m)
        x[:, 162:318] = (x[:, 0:3].unsqueeze(1) + offs.unsqueeze(0)
                         + 0.01 * torch.randn(T, 52, 3)).reshape(T, -1)
        x[:, 9:162] = 0.2 * torch.randn(T, 153)              # joint angles small, in-range
        # object sits 0.2m from body[0] (a hand is near it -> interaction present)
        x[:, 318:321] = x[:, 162:165] + 0.2
        return x

    cases = {}
    cases['clean'] = (clean(), [])                            # expect NO flags

    x = clean(); x[10, 5] = float('nan')
    cases['nan'] = (x, ['nonfinite'])

    # one body stuck 50m from root, CONSTANT over time: huge pose extent, but
    # normal velocity (drifts with root) and interaction intact -> explosion only
    x = clean(); x[:, 192:195] = x[:, 0:3] + 50.0             # body #10 (chans 192:195)
    cases['explosion'] = (x, ['explosion'])

    x = clean(); x[:, 9:162] = 5.0                            # joints past +-pi
    cases['joint_limit'] = (x, ['joint_limit'])

    # SUSTAINED unrealistic speed: rigid translation 0.4/axis -> velocity
    # 0.4*sqrt(3)=0.69 > VEL_MAX, but acceleration ~0 (linear) -> teleport only.
    x = clean()
    ramp = (torch.arange(T).float() * 0.4).unsqueeze(1)
    x[:, 0:3] += ramp; x[:, 162:318] += ramp; x[:, 318:321] += ramp
    cases['teleport'] = (x, ['teleport'])

    # smooth FAST move at 0.2/axis -> velocity 0.2*sqrt(3)=0.35 (< VEL_MAX),
    # acceleration ~0 -> NO flags. Same velocity as occlusion_snap below, but
    # smooth -> proves 'jitter' is acceleration, not velocity in disguise.
    x = clean()
    ramp = (torch.arange(T).float() * 0.2).unsqueeze(1)
    x[:, 0:3] += ramp; x[:, 162:318] += ramp; x[:, 318:321] += ramp
    cases['smooth_fast'] = (x, [])

    # OCCLUSION snap: one foot (body #7) jumps out for a SINGLE frame and back.
    # Same ~0.35 per-frame velocity as smooth_fast (< VEL_MAX, so the old velocity
    # check MISSES it) -- but the acceleration spikes -> caught as 'jitter'.
    x = clean(); x[30, 183:186] += 0.2
    cases['occlusion_snap'] = (x, ['jitter'])

    x = clean(); x[:, 318:321] = x[:, 162:165] + 5.0          # object 5m from every body
    cases['no_interaction'] = (x, ['no_interaction'])

    print("=== SELF-TEST: inject known defect -> expect exact flags ===")
    ok = True
    for name, (x, expected) in cases.items():
        _, got = clip_stats(x)
        passed = (sorted(got) == sorted(expected))
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:16} expected={expected!s:20} got={got}")
    print(f"\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'} -- "
          f"each defect is caught, and clean data raises no flags." if ok
          else "SOME CHECKS FAILED -- detector is wrong.")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--motion-dir', default='InterAct/OMOMO_new')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--selftest', action='store_true',
                   help='inject known defects and verify the detector flags exactly them')
    args = p.parse_args()
    return selftest() if args.selftest else audit(args.motion_dir, args.subjects)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""What does Isaac Gym / PhysX actually build from a per-subject MJCF?

Everything offline says smplx_omomo_sub4.xml is fine (structure identical to
sub2, sane geometry, clean FK), yet the simulator moves sub4's left leg far
beyond the state it is written each frame. The file's parsed content cannot
explain that, so this prints what the ENGINE derived from it, per rigid body
and per DOF, and compares a suspect subject against a control:

  rigid bodies : mass, centre of mass, inertia diagonal (as PhysX computed them
                 from the geoms), flags
  dofs         : limits, drive mode, stiffness/damping/armature/effort/velocity
  shapes       : count, friction, restitution, filter

Anything NaN/inf/non-positive, or a suspect/control ratio outside [0.5, 2] on
a mass or inertia entry, or ANY difference in a dof/shape property (those
should be identical across subjects), is flagged. Needs Isaac Gym, so run on
the cluster in intermimic-gym2, e.g.:

  srun --account=simurgh --partition=simurgh-interactive --qos=normal \
       --exclude=simurgh6,simurgh2 --cpus-per-task=4 --mem=16G --gres=gpu:1 \
       --time=00:10:00 python -u scripts/probe_asset_in_gym.py --suspect sub4 --control sub2
"""
import argparse
import math
import os
import sys

import numpy as np

ASSET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "isaacgym", "src", "intermimic", "data", "assets", "smplx")


def load(gym, sim, subject):
    """Load smplx_omomo_<subject>.xml exactly the way humanoid.py does."""
    from isaacgym import gymapi
    opts = gymapi.AssetOptions()
    opts.angular_damping = 0.01
    opts.max_angular_velocity = 100.0
    opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset = gym.load_asset(sim, ASSET_ROOT, f"smplx_omomo_{subject}.xml", opts)
    if asset is None:
        sys.exit(f"ERROR: load_asset returned None for {subject}")
    names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    rb = gym.get_asset_rigid_body_properties(asset)
    dofs = gym.get_asset_dof_properties(asset)
    shapes = gym.get_asset_rigid_shape_properties(asset)
    bodies = []
    for n, p in zip(names, rb):
        I = p.inertia
        bodies.append(dict(name=n, mass=float(p.mass),
                           com=(float(p.com.x), float(p.com.y), float(p.com.z)),
                           inertia=(float(I.x.x), float(I.y.y), float(I.z.z)),
                           offdiag=(float(I.x.y), float(I.x.z), float(I.y.z)),
                           flags=int(p.flags)))
    dof_rows = []
    for i, n in enumerate(dof_names):
        d = dofs[i]
        dof_rows.append(dict(name=n, hasLimits=bool(d["hasLimits"]), lower=float(d["lower"]),
                             upper=float(d["upper"]), driveMode=int(d["driveMode"]),
                             stiffness=float(d["stiffness"]), damping=float(d["damping"]),
                             armature=float(d["armature"]), effort=float(d["effort"]),
                             velocity=float(d["velocity"]), friction=float(d["friction"])))
    shape_rows = [dict(friction=float(s.friction), restitution=float(s.restitution),
                       filter=int(s.filter), rolling=float(s.rolling_friction),
                       torsion=float(s.torsion_friction)) for s in shapes]
    return dict(bodies=bodies, dofs=dof_rows, shapes=shape_rows,
                n_bodies=gym.get_asset_rigid_body_count(asset),
                n_dofs=gym.get_asset_dof_count(asset),
                n_shapes=gym.get_asset_rigid_shape_count(asset))


def finite_positive(v):
    return all(math.isfinite(x) for x in v) and all(x > 0 for x in v)


def compare(S, C, sus, ctl, ratio_lo=0.5, ratio_hi=2.0):
    bad = []
    print(f"\n{sus}: {S['n_bodies']} bodies, {S['n_dofs']} dofs, {S['n_shapes']} shapes   "
          f"{ctl}: {C['n_bodies']} bodies, {C['n_dofs']} dofs, {C['n_shapes']} shapes")
    if (S["n_bodies"], S["n_dofs"], S["n_shapes"]) != (C["n_bodies"], C["n_dofs"], C["n_shapes"]):
        bad.append("topology counts differ")

    print(f"\n{'body':12s} {'mass '+sus:>10s} {'mass '+ctl:>10s} {'ratio':>6s}   "
          f"{'Ixx/Iyy/Izz '+sus:>28s}   {'Ixx/Iyy/Izz '+ctl:>28s}   {'com '+sus}")
    for b, c in zip(S["bodies"], C["bodies"]):
        if b["name"] != c["name"]:
            bad.append(f"body order differs: {b['name']} vs {c['name']}")
        mr = b["mass"] / c["mass"] if c["mass"] > 0 else float("nan")
        ir = [x / y if y > 0 else float("nan") for x, y in zip(b["inertia"], c["inertia"])]
        flag = ""
        if not (math.isfinite(b["mass"]) and b["mass"] > 0):
            flag += " <<< bad mass"
        if not finite_positive(b["inertia"]):
            flag += " <<< bad inertia"
        if not all(math.isfinite(x) for x in b["com"]):
            flag += " <<< bad com"
        if math.isfinite(mr) and not (ratio_lo <= mr <= ratio_hi):
            flag += f" <<< mass ratio {mr:.2f}"
        if any(math.isfinite(r) and not (ratio_lo <= r <= ratio_hi) for r in ir):
            flag += " <<< inertia ratio " + "/".join(f"{r:.2f}" for r in ir)
        if b["flags"] != c["flags"]:
            flag += f" <<< flags {b['flags']} vs {c['flags']}"
        if flag:
            bad.append(f"{b['name']}:{flag}")
        print(f"{b['name']:12s} {b['mass']:10.4f} {c['mass']:10.4f} {mr:6.3f}   "
              f"{b['inertia'][0]:8.5f}/{b['inertia'][1]:8.5f}/{b['inertia'][2]:8.5f}   "
              f"{c['inertia'][0]:8.5f}/{c['inertia'][1]:8.5f}/{c['inertia'][2]:8.5f}   "
              f"({b['com'][0]:+.3f},{b['com'][1]:+.3f},{b['com'][2]:+.3f}){flag}")

    print("\nDOF properties (must be IDENTICAL across subjects):")
    diffs = 0
    for d, e in zip(S["dofs"], C["dofs"]):
        keys = [k for k in d if k != "name" and d[k] != e[k]]
        if d["name"] != e["name"] or keys:
            diffs += 1
            print(f"  {d['name']:14s} differs: " + ", ".join(f"{k} {d[k]} vs {e[k]}" for k in keys))
    if diffs:
        bad.append(f"{diffs} dof(s) differ")
    else:
        d0 = S["dofs"][0]
        print(f"  all {len(S['dofs'])} identical; e.g. {d0['name']}: limits [{d0['lower']:.3f}, "
              f"{d0['upper']:.3f}] stiffness {d0['stiffness']} damping {d0['damping']} "
              f"armature {d0['armature']} effort {d0['effort']}")
        nonfinite = [d["name"] for d in S["dofs"]
                     if not all(math.isfinite(d[k]) for k in ("lower", "upper", "stiffness", "damping", "armature", "effort", "velocity"))]
        if nonfinite:
            bad.append(f"non-finite dof props: {nonfinite}")

    print("\nShape properties (must be IDENTICAL across subjects):")
    sd = [i for i, (a, b) in enumerate(zip(S["shapes"], C["shapes"])) if a != b]
    if sd:
        bad.append(f"{len(sd)} shape(s) differ")
        for i in sd[:10]:
            print(f"  shape {i}: {S['shapes'][i]} vs {C['shapes'][i]}")
    else:
        print(f"  all {len(S['shapes'])} identical; e.g. {S['shapes'][0]}")

    tot_s = sum(b["mass"] for b in S["bodies"]); tot_c = sum(b["mass"] for b in C["bodies"])
    print(f"\ntotal mass as PhysX sees it: {sus} {tot_s:.2f} kg   {ctl} {tot_c:.2f} kg")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", default="sub4")
    ap.add_argument("--control", default="sub2")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more subjects to compare against the control")
    a = ap.parse_args()

    from isaacgym import gymapi   # after argparse so --help works without Isaac Gym
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.physx.use_gpu = False           # asset properties only; no stepping
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        sys.exit("ERROR: create_sim failed")

    C = load(gym, sim, a.control)
    verdict = {}
    for s in [a.suspect] + list(a.extra):
        S = load(gym, sim, s)
        print("=" * 100)
        print(f"ENGINE VIEW  {s} (suspect) vs {a.control} (control)")
        print("=" * 100)
        verdict[s] = compare(S, C, s, a.control)

    print("\n" + "=" * 100)
    for s, bad in verdict.items():
        if bad:
            print(f"VERDICT {s}: {len(bad)} anomaly(ies) in what PhysX built from the MJCF:")
            for line in bad:
                print(f"   - {line}")
        else:
            print(f"VERDICT {s}: PhysX built the same kind of body as {a.control} -- masses, "
                  f"inertias, coms, dof and shape properties all sane and comparable. The MJCF "
                  f"import is NOT where {s} goes wrong.")
    gym.destroy_sim(sim)
    sys.exit(1 if any(verdict.values()) else 0)


if __name__ == "__main__":
    main()

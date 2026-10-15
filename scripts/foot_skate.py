#!/usr/bin/env python3
"""Foot skating and root drift of a rollout against its reference, as numbers.

"The feet don't pick up, the humanoid slides" (Jess, 2026-10-15) is a claim
about two measurable things, so measure them instead of eyeballing a video:

  skating     horizontal speed of a toe WHILE it is on the floor. A planted foot
              has ~0; a sliding one moves at the body's speed. Per foot: the mean
              speed over contact frames (cm/s) and the share of contact frames
              above --skate-thresh (default 10 cm/s), for the rollout AND the
              reference, so the reference's own value is the floor.
  lift        share of frames each toe is OFF the floor -- "the feet don't pick
              up" = the rollout's lift share far below the reference's.
  root drift  horizontal distance between the rollout's and the reference's
              pelvis per frame: mean, last compared frame, max, and the slope
              over time (cm per second): a drift that GROWS along the episode is
              compounding error (a horizon problem); one that is large from the
              first frames is not.

Inputs: a DUMP_TRAJ-layout npz (a rollout), and the reference it tracked as a
dump npz (scripts/clip_to_dump.py) or a 591-channel .pt directly. The floor is
the 5th percentile of toe height over the REFERENCE (the mocap floor); a toe is
on the floor when within --contact-height of it (default 5 cm). Frames are
compared index by index from frame 0, so the dump must start at the clip's
frame 0 (stateInit Start, as every eval/render cfg does).

    python3 scripts/foot_skate.py --rollout renders/x/rollout.npz \\
        --reference InterAct/OMOMO_new/sub6_woodchair_000.pt [--out skate.json]
"""
import argparse
import os
import sys

import numpy as np

# InterMimic's 52-body order (scripts/motion_feedback.py BODY_NAMES)
FEET = {"L_Ankle": 3, "L_Toe": 4, "R_Ankle": 7, "R_Toe": 8}
PELVIS = 0
FPS = 30.0


def load_body_pos(path):
    """(T,52,3) from a dump npz or a 591-channel .pt."""
    if str(path).endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return d["body_pos"].astype(np.float64)
    import torch
    x = torch.load(path, map_location="cpu", weights_only=False).detach().numpy()
    if x.shape[-1] != 591:
        raise SystemExit(f"{path}: expected 591 channels, got {x.shape[-1]}")
    return x[:, 162:318].reshape(len(x), 52, 3).astype(np.float64)


def foot_metrics(body_pos, floor_z, contact_height, skate_thresh, fps=FPS):
    """{toe: {contact_share, lift_share, skate_mean_cms, skate_share, n_contact_steps}}."""
    out = {}
    for name in ("L_Toe", "R_Toe"):
        p = body_pos[:, FEET[name]]
        on_floor = p[:, 2] < floor_z + contact_height
        v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * fps          # m/s, frame t -> t+1
        planted = on_floor[:-1] & on_floor[1:]                                 # both ends on the floor
        contact_v = v[planted]
        out[name] = {
            "contact_share": float(on_floor.mean()),
            "lift_share": float(1.0 - on_floor.mean()),
            "skate_mean_cms": float(contact_v.mean() * 100.0) if contact_v.size else 0.0,
            "skate_share": float((contact_v > skate_thresh).mean()) if contact_v.size else 0.0,
            "n_contact_steps": int(contact_v.size),
        }
    return out


def root_drift(roll, ref, fps=FPS):
    """Horizontal pelvis distance per compared frame -> (per-frame m, mean, last, max, slope cm/s)."""
    n = min(len(roll), len(ref))
    d = np.linalg.norm(roll[:n, PELVIS, :2] - ref[:n, PELVIS, :2], axis=1)
    t = np.arange(n) / fps
    slope = float(np.polyfit(t, d, 1)[0] * 100.0) if n > 2 else 0.0
    return d, float(d.mean()), float(d[-1]), float(d.max()), slope, n


def compare(roll, ref, contact_height=0.05, skate_thresh=0.10):
    floor_z = float(np.percentile(np.concatenate([ref[:, FEET["L_Toe"], 2], ref[:, FEET["R_Toe"], 2]]), 5))
    return {"floor_z": floor_z, "frames": (len(roll), len(ref)),
            "rollout": foot_metrics(roll, floor_z, contact_height, skate_thresh),
            "reference": foot_metrics(ref, floor_z, contact_height, skate_thresh),
            "drift": root_drift(roll, ref)}


def report(r):
    d, mean_d, last_d, max_d, slope, n = r["drift"]
    lines = [f"frames: rollout {r['frames'][0]}, reference {r['frames'][1]}, compared {n}; "
             f"mocap floor z = {r['floor_z']:.3f} m",
             f"{'toe':6s} {'':10s} {'on floor':>9s} {'lifted':>8s} {'skate cm/s':>11s} {'skating':>8s}"]
    for foot in ("L_Toe", "R_Toe"):
        for who in ("reference", "rollout"):
            m = r[who][foot]
            lines.append(f"{foot:6s} {who:10s} {m['contact_share']:9.0%} {m['lift_share']:8.0%} "
                         f"{m['skate_mean_cms']:11.1f} {m['skate_share']:8.0%}")
    lines.append(f"root drift vs reference: mean {mean_d * 100:.1f} cm, last {last_d * 100:.1f} cm, "
                 f"max {max_d * 100:.1f} cm, slope {slope:+.1f} cm/s over {n / FPS:.1f} s")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollout", required=True, help="DUMP_TRAJ npz")
    ap.add_argument("--reference", required=True, help="the reference it tracked: dump npz or 591-channel .pt")
    ap.add_argument("--contact-height", type=float, default=0.05, help="toe within this of the floor = on it (m)")
    ap.add_argument("--skate-thresh", type=float, default=0.10, help="contact-frame speed above this = skating (m/s)")
    ap.add_argument("--out", default=None, help="also write the numbers as JSON")
    a = ap.parse_args(argv)
    r = compare(load_body_pos(a.rollout), load_body_pos(a.reference), a.contact_height, a.skate_thresh)
    print(f"rollout   {a.rollout}\nreference {a.reference}")
    print(report(r))
    if a.out:
        import json
        d, mean_d, last_d, max_d, slope, n = r["drift"]
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump({"rollout": a.rollout, "reference": a.reference, "floor_z": r["floor_z"], "frames_compared": n,
                   "feet": {"rollout": r["rollout"], "reference": r["reference"]},
                   "root_drift_cm": {"mean": mean_d * 100, "last": last_d * 100, "max": max_d * 100,
                                     "slope_cm_per_s": slope, "per_frame": (d * 100).round(2).tolist()}},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

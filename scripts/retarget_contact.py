#!/usr/bin/env python3
"""Contact-aware reference retargeting.

InterMimic drives EVERY target body with the SOURCE subject's joint angles
(dof_pos). Because bone lengths differ, the target body's hands land somewhere
else than the source's did -- measured 0.6 cm (sub4) to 12 cm (sub1/sub10) on
OMOMO. The reference's interaction geometry (`ig`) and contact flags still
describe the SOURCE's hand-object relationship, so the policy is handed a target
it cannot exactly satisfy and has to burn samples absorbing the discrepancy.

This script fixes the reference: it re-solves dof_pos so the TARGET body
reproduces the source's world-space body positions -- with extra weight on the
bodies that are in contact -- then recomputes body_pos from that solution.
Output is a clip in the identical [T, 591] layout, so training loads it normally.

NOT recomputed: `ig` (interaction geometry) and the contact flags. That is sound
for this solve because the target is made to reproduce the SOURCE's world body
positions, so the human-object relationship those fields encode is preserved to
within the solve residual (sub16 2.7 -> 0.13 cm).

Two modes (the object side is a scale factor, so both are the same solve):
  --object-scale 1.0        body-only retarget      (original object)
  --object-scale <sx sy sz> body + scaled object    (augmented object)
When the object is scaled, the contact targets move with the surface, so the
solve pulls the hands onto the NEW surface.

Layout of a clip tensor [T, 591] (see intermimic.py _load_motion):
  0:3 root_pos | 3:7 root_rot QUATERNION (x,y,z,w) | 7:9 zero pad | 9:162 dof_pos(51*3)
  162:318 body_pos(52*3, WORLD frame) | 318:321 obj_pos | 321:325 obj_rot
  ... | 330:331 contact_obj | 331:383 contact_human

VERIFIED against OMOMO_new: cols 3:7 have unit norm (1 - 3e-9) and cols 7:9 are
exactly 0; decoding 3:7 as an (x,y,z,w) quaternion and running the FK below
reproduces the stored body_pos to 0.10 mm mean / 0.27 mm max (sub2/sub6/sub9).
An earlier version of this header called 3:9 a 6D rotation -- it is NOT, and
building a matrix from those 6 numbers yields a non-orthonormal garbage frame
(|RR^T - I| = 0.84). tests/test_retarget_fk.py pins this.

  python3 scripts/retarget_contact.py --selftest
  python3 scripts/retarget_contact.py --clip InterAct/OMOMO_new/sub2_largetable_000.pt \
      --source sub2 --target sub16 --out-dir InterAct/OMOMO_retarget_contact
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_pose import _parse_mjcf_tree                      # noqa: E402

MJCF = "isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_%s.xml"
DOF, NB = 153, 52
I_ROOTP, I_ROOTQ = slice(0, 3), slice(3, 7)      # root_pos | root quat (x,y,z,w)
I_DOF, I_BODY = slice(9, 162), slice(162, 318)
I_OBJP, I_OBJR = slice(318, 321), slice(321, 325)
I_CONTACT_H = slice(331, 383)
# FK-vs-data agreement gate. Measured 0.10 mm mean / 0.27 mm max on OMOMO_new
# (sub2/sub6/sub9); 5 mm is ~50x headroom yet still catches any frame or
# convention error, which show up as centimetres-to-metres.
FK_TOL_M = 0.005


def quat_xyzw_to_mat(q):
    """(T,4) unit quaternion in (x,y,z,w) order -> (T,3,3) rotation matrices.

    The clip's root rotation uses x,y,z,w (verified: this decoding makes the FK
    reproduce the stored body_pos to 0.1 mm; the w,x,y,z reading is off by 1.57).
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


# ---------------------------------------------------------------- differentiable FK
class MJCFChain:
    """Torch forward kinematics for one subject's MJCF, matching smplx_pose.mjcf_fk:
    child_R = parent_R @ expmap(dof_j), child_p = parent_p + parent_R @ rest_offset."""

    def __init__(self, subject, device="cpu"):
        tree = _parse_mjcf_tree(MJCF % subject)
        self.names = [n for n, _, _ in tree]
        idx = {n: i for i, n in enumerate(self.names)}
        self.parent = [(-1 if p is None else idx[p]) for _, p, _ in tree]
        self.offset = torch.tensor(np.array([o for _, _, o in tree]), dtype=torch.float64,
                                   device=device)
        assert len(self.names) == NB, f"{subject}: expected {NB} bodies, got {len(self.names)}"

    def fk(self, dof, root_pos=None, root_rot=None):
        """dof (T,153) -> body positions (T, 52, 3).

        Pass root_pos/root_rot to get WORLD-frame positions -- the frame the clip's
        body_pos field is in, and the only frame safe to write out. Omitting them
        yields root-local, root-rotation-cancelled coordinates: fine for comparing
        two chains under identical dof (the shared root cancels), catastrophic if
        written to a clip, which is exactly the bug that produced ~0 reward.
        """
        T = dof.shape[0]
        d = dof.view(T, 51, 3)
        R = _expmap(d)                                   # (T,51,3,3)
        Rg = [None] * NB
        pg = [None] * NB
        di = 0
        for b in range(NB):
            par = self.parent[b]
            if par < 0:
                Rg[b] = (torch.eye(3, dtype=dof.dtype, device=dof.device).expand(T, 3, 3)
                         if root_rot is None else root_rot)
                pg[b] = (torch.zeros(T, 3, dtype=dof.dtype, device=dof.device)
                         if root_pos is None else root_pos)
            else:
                Rg[b] = Rg[par] @ R[:, di]
                pg[b] = pg[par] + (Rg[par] @ self.offset[b].to(dof.dtype)).squeeze(-1) \
                    if False else pg[par] + torch.einsum('tij,j->ti', Rg[par], self.offset[b].to(dof.dtype))
                di += 1
        return torch.stack(pg, dim=1)


def _expmap(v):
    """Rodrigues, batched: (...,3) axis-angle -> (...,3,3). Differentiable."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th.unsqueeze(-1)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


# ---------------------------------------------------------------------- the solve
def retarget(clip, source, target, object_scale=(1., 1., 1.), iters=300, lr=0.05,
             w_contact=10.0, w_pose=1.0, w_reg=0.1, device="cpu", verbose=True):
    """Re-solve dof_pos so `target`'s body reproduces `source`'s world body positions,
    weighting bodies that are in contact. Returns (new_clip, stats)."""
    # .detach(): some OMOMO_new clips were saved with requires_grad=True, which
    # makes every slice non-leaf and Adam refuse the dof ("can't optimize a
    # non-leaf Tensor"). We never want autograd history from the loaded data.
    clip = clip.detach().to(torch.float64)
    dof_src = clip[:, I_DOF].clone().to(device)
    contact_h = clip[:, I_CONTACT_H].to(device)                 # (T,52) binary
    src_chain, tgt_chain = MJCFChain(source, device), MJCFChain(target, device)

    # WORLD frame throughout. The solve itself is invariant to this choice (the
    # loss is a distance, and both sides get the same rigid root transform), but
    # the OUTPUT body_pos field is world-frame, and the object-scale branch below
    # mixes in world-space obj_pos -- so root-local FK here silently corrupts both.
    root_p = clip[:, I_ROOTP].to(device)
    root_R = quat_xyzw_to_mat(clip[:, I_ROOTQ].to(device))

    # Targets = where the SOURCE body actually was (its FK, same frame convention).
    with torch.no_grad():
        p_src = src_chain.fk(dof_src, root_pos=root_p, root_rot=root_R)   # (T,52,3) world

        # GATE: for a clip of subject S, FK(S, dof, root) MUST reproduce the clip's
        # own stored body_pos. This is the check whose absence let a root-local
        # write (1.34 m off, world-vs-local) ship and silently kill training --
        # --selftest compares FK to FK, so shared convention errors cancel there.
        # Fail loudly rather than emit references no policy can follow.
        fk_err = (p_src - clip[:, I_BODY].to(device).reshape(-1, NB, 3)).norm(dim=-1)
        if fk_err.mean() > FK_TOL_M:
            raise RuntimeError(
                f"FK does not reproduce the clip's stored body_pos for source "
                f"'{source}': mean {fk_err.mean()*1000:.2f} mm, max "
                f"{fk_err.max()*1000:.2f} mm (tolerance {FK_TOL_M*1000:.1f} mm). "
                f"Expected ~0.1 mm. Either the clip is not this subject's, or a "
                f"layout/convention assumption is wrong -- refusing to write a "
                f"retarget built on a broken forward model.")
    # An anisotropically scaled object moves its surface; contact targets follow it.
    s = torch.tensor(object_scale, dtype=torch.float64, device=device)
    if not torch.allclose(s, torch.ones(3, dtype=torch.float64, device=device)):
        obj_p = clip[:, I_OBJP].to(device).unsqueeze(1)          # (T,1,3) object origin
        p_tgt_pts = obj_p + (p_src - obj_p) * s                  # scale about the object
        p_goal = torch.where(contact_h.unsqueeze(-1) > 0.5, p_tgt_pts, p_src)
    else:
        p_goal = p_src

    # Per-body weight: contact bodies dominate, everything else keeps the pose honest.
    w = w_pose + w_contact * (contact_h > 0.5).to(torch.float64)   # (T,52)

    dof = dof_src.clone().requires_grad_(True)
    opt = torch.optim.Adam([dof], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        p = tgt_chain.fk(dof, root_pos=root_p, root_rot=root_R)
        err = ((p - p_goal) ** 2).sum(-1)                        # (T,52)
        loss = (w * err).mean() + w_reg * ((dof - dof_src) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"    it {it:4d}  loss {loss.item():.6f}")

    with torch.no_grad():
        p_new = tgt_chain.fk(dof, root_pos=root_p, root_rot=root_R)
        p_before = tgt_chain.fk(dof_src, root_pos=root_p, root_rot=root_R)
        cm = (contact_h > 0.5)
        def _e(p):
            d = (p - p_goal).norm(dim=-1)
            return (d[cm].mean().item() if cm.any() else float('nan'), d.mean().item())
        (cb, ab), (ca, aa) = _e(p_before), _e(p_new)
        stats = dict(contact_before_cm=cb * 100, contact_after_cm=ca * 100,
                     all_before_cm=ab * 100, all_after_cm=aa * 100,
                     contact_frac=cm.to(torch.float64).mean().item())
        out = clip.clone()
        out[:, I_DOF] = dof.detach().cpu()
        out[:, I_BODY] = p_new.reshape(p_new.shape[0], -1).cpu()
    return out.to(torch.float32), stats


# ------------------------------------------------------------------------ selftest
def selftest():
    """Identity retarget (sub2 -> sub2) must be a near no-op: if the FK or the target
    construction is wrong, this is where it shows up."""
    clip = torch.load("InterAct/OMOMO_new/sub2_largetable_000.pt", map_location="cpu",
                      weights_only=False).detach()
    print("[selftest] identity sub2 -> sub2 (expect ~0 contact error, dof unchanged)")
    out, st = retarget(clip, "sub2", "sub2", iters=60, verbose=False)
    dd = (out[:, I_DOF] - clip[:, I_DOF]).abs().max().item()
    print(f"  contact err before {st['contact_before_cm']:.3f} cm -> after {st['contact_after_cm']:.3f} cm")
    print(f"  max |dof change| = {dd:.5f} rad   (contact frames {st['contact_frac']*100:.1f}%)")
    ok = st["contact_before_cm"] < 0.5 and dd < 0.05
    print("  PASS" if ok else "  FAIL — identity retarget is not a no-op")
    return 0 if ok else 1


# --------------------------------------------------------------------- batch mode
def _one(job):
    """Worker: retarget a single (clip, target) pair -> <out>/<target>/<clip>.pt."""
    # One thread per worker: torch grabs ~n_cores threads per process by default, so
    # a pool of W workers oversubscribes W*n_cores threads and thrashes to a standstill.
    torch.set_num_threads(1)
    clip_path, source, target, scale, iters, out_dir = job
    dst = os.path.join(out_dir, target, os.path.basename(clip_path))
    if os.path.exists(dst):                       # resume: never redo finished work
        return (target, os.path.basename(clip_path), None, None, "skip")
    try:
        clip = torch.load(clip_path, map_location="cpu", weights_only=False).detach()
        out, st = retarget(clip, source, target, scale, iters=iters, verbose=False)
        # An under-converged solve can end up WORSE than not retargeting at all
        # (measured: 25 iters took sub16 from 2.72cm to 4.86cm). Never write that --
        # it would silently hand training a reference worse than the original.
        if st["contact_after_cm"] > st["contact_before_cm"]:
            return (target, os.path.basename(clip_path),
                    st["contact_before_cm"], st["contact_after_cm"],
                    f"WORSE {st['contact_before_cm']:.2f}->{st['contact_after_cm']:.2f}cm "
                    f"(raise --iters)")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(out, dst)
        return (target, os.path.basename(clip_path),
                st["contact_before_cm"], st["contact_after_cm"], "ok")
    except Exception as e:                        # never let one clip kill the sweep
        return (target, os.path.basename(clip_path), None, None, f"ERROR {e!r}")


def batch(motion_dir, source, targets, out_dir, scale, iters, workers, limit=None):
    """Retarget EVERY clip of `source` onto EVERY target body. This is the
    preprocessing step that makes the retargeted reference usable for training:
    one file per (target_body, clip), written to <out_dir>/<body>/<clip>.pt."""
    import json
    import multiprocessing as mp
    from collections import defaultdict

    clips = sorted(f for f in os.listdir(motion_dir)
                   if f.endswith(".pt") and f.startswith(source + "_"))
    if limit:
        clips = clips[:limit]
    if not clips:
        raise SystemExit(f"no clips for source '{source}' in {motion_dir}")
    jobs = [(os.path.join(motion_dir, c), source, t, scale, iters, out_dir)
            for t in targets for c in clips]
    print(f"[batch] {len(clips)} clips x {len(targets)} bodies = {len(jobs)} pairs, "
          f"{workers} workers -> {out_dir}")

    agg, errs, skipped = defaultdict(list), [], 0
    with mp.Pool(workers) as pool:
        for i, (tgt, clip, before, after, status) in enumerate(pool.imap_unordered(_one, jobs), 1):
            if status == "ok":
                agg[tgt].append((before, after))
            elif status == "skip":
                skipped += 1
            else:
                errs.append((tgt, clip, status))
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  (skipped {skipped}, errors {len(errs)})", flush=True)

    # Per-body summary: this is the evidence the retarget actually did something.
    # A clip whose contact flags are all zero has no contact error to report, so
    # retarget() returns nan for it (e.g. sub6 has 3 such clips out of 278, sub9
    # 3 of 293, sub2 none). Those must be EXCLUDED from the mean, not allowed to
    # poison it -- a plain mean turns every body's average into nan and the whole
    # generation looks failed when the data is fine. Counted and reported, not hidden.
    print(f"\n[batch] per-body contact error (cm), mean over clips:")
    summary = {}
    for t in targets:
        if not agg[t]:
            continue
        pairs = [(x[0], x[1]) for x in agg[t]]
        live = [(b, a) for b, a in pairs if not (np.isnan(b) or np.isnan(a))]
        n_nc = len(pairs) - len(live)
        if not live:
            summary[t] = dict(before_cm=float("nan"), after_cm=float("nan"),
                              n=0, n_no_contact=n_nc)
            print(f"    {t:>8}:   no clip with any contact frame ({n_nc} clips)")
            continue
        b = float(np.mean([x[0] for x in live])); a = float(np.mean([x[1] for x in live]))
        summary[t] = dict(before_cm=b, after_cm=a, n=len(live), n_no_contact=n_nc)
        note = f"   [{n_nc} clip(s) had no contact frames, excluded]" if n_nc else ""
        print(f"    {t:>8}: {b:6.2f} -> {a:6.2f}   ({len(live)} clips){note}")
    if errs:
        print(f"\n[batch] {len(errs)} FAILURES (not silently dropped):")
        for t, c, s in errs[:10]:
            print(f"    {t}/{c}: {s}")
    with open(os.path.join(out_dir, "retarget_summary.json"), "w") as f:
        json.dump(dict(source=source, object_scale=list(scale), iters=iters,
                       summary=summary, errors=[list(e) for e in errs]), f, indent=2)
    print(f"\n[batch] done -> {out_dir} (summary in retarget_summary.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--batch", action="store_true",
                    help="retarget all clips of --source onto all --targets")
    ap.add_argument("--motion-dir", default="InterAct/OMOMO_new")
    ap.add_argument("--targets", nargs="*", help="target bodies, e.g. sub1 sub16 sub100")
    ap.add_argument("--targets-from", help="env yaml to read subjectBodies from")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    ap.add_argument("--clip")
    ap.add_argument("--source", default="sub2")
    ap.add_argument("--target", required=False)
    ap.add_argument("--object-scale", nargs=3, type=float, default=[1., 1., 1.])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out-dir", default="InterAct/OMOMO_retarget_contact")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.batch:
        targets = a.targets
        if a.targets_from:                        # read subjectBodies straight from a cfg
            import yaml
            targets = yaml.safe_load(open(a.targets_from))["env"]["subjectBodies"]
        if not targets:
            ap.error("--batch needs --targets or --targets-from")
        batch(a.motion_dir, a.source, targets, a.out_dir, tuple(a.object_scale),
              a.iters, a.workers, a.limit)
        return
    if not (a.clip and a.target):
        ap.error("--clip and --target required (or --selftest / --batch)")
    clip = torch.load(a.clip, map_location="cpu", weights_only=False).detach()
    print(f"[retarget] {os.path.basename(a.clip)}  {a.source} -> {a.target}  "
          f"object_scale={a.object_scale}")
    out, st = retarget(clip, a.source, a.target, tuple(a.object_scale), iters=a.iters)
    print(f"  contact err {st['contact_before_cm']:.2f} -> {st['contact_after_cm']:.2f} cm | "
          f"all-body {st['all_before_cm']:.2f} -> {st['all_after_cm']:.2f} cm")
    os.makedirs(a.out_dir, exist_ok=True)
    dst = os.path.join(a.out_dir, os.path.basename(a.clip))
    torch.save(out, dst)
    print(f"  wrote {dst}")


if __name__ == "__main__":
    main()

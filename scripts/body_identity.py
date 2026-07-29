#!/usr/bin/env python3
"""Read which BODY a motion clip's reference is expressed on, straight from the data.

The whole method rests on one fact: a bone length -- the distance from a joint to
its parent -- is a property of the SKELETON, not of the motion. For a rigid body it
is identical in every frame of every clip. So the body a reference was authored for
is recoverable from `body_pos` alone, with no metadata and no trust in filenames:

    measure the 51 bone lengths in a clip  ->  compare against every candidate MJCF
                                           ->  the one that matches IS the body

Two questions this answers:
  * "is every subject's motion on one shared body?"   -> do all subjects measure the same?
  * "which body is that?"                             -> which MJCF matches?

Used by tests/test_body_identity.py (assertions) and scripts/plot_body_identity.py
(figures). Nothing here imports Isaac Gym, so it runs on a laptop.
"""
from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_DIR = os.path.join(REPO, "isaacgym/src/intermimic/data/assets/smplx")

# Column layout of a clip tensor [T, 591] -- see intermimic.py _load_motion.
I_BODY = slice(162, 162 + 52 * 3)
N_BODIES = 52

# Candidate motion directories. OMOMO_new is what training actually reads
# (motion_file: InterAct/OMOMO_new in 281 of the cfgs).
FULL_OMOMO_NEW = os.path.expanduser("~/new_one/OMOMO_new")
REPO_OMOMO = os.path.join(REPO, "InterAct/OMOMO")
REPO_OMOMO_RETARGET = os.path.join(REPO, "InterAct/OMOMO_retarget")


def parse_tree(path):
    """Ordered [(name, parent_name, rest_offset)], depth-first -- the order the sim enumerates."""
    root = ET.parse(path).getroot()
    out = []

    def walk(elem, parent):
        for b in elem.findall("body"):
            pos = np.array([float(v) for v in b.get("pos", "0 0 0").split()])
            out.append((b.get("name"), parent, pos))
            walk(b, b.get("name"))

    walk(root.find("worldbody"), None)
    return out


def _topology(path=None):
    """(names, parent_index, child_ids, parent_ids) for the shared 52-body skeleton."""
    tree = parse_tree(path or mjcf_path("sub2"))
    names = [n for n, _, _ in tree]
    idx = {n: i for i, n in enumerate(names)}
    parent = [(-1 if p is None else idx[p]) for _, p, _ in tree]
    child = [i for i in range(len(names)) if parent[i] >= 0]
    return names, parent, child, [parent[i] for i in child]


def mjcf_path(sub):
    """'sub9' -> per-subject MJCF; 'omomo' / 'omomo_isaaclab' -> the canonical assets."""
    if sub.startswith("sub"):
        return os.path.join(MJCF_DIR, f"smplx_omomo_{sub}.xml")
    return os.path.join(MJCF_DIR, f"{sub}.xml")


@lru_cache(maxsize=None)
def mjcf_bone_lengths(sub):
    """Rest bone lengths (51,) taken from the MJCF's own <body pos=...> offsets.

    This is the body as Isaac Gym will build it -- no simulation needed.
    """
    names, _, child, _ = _topology()
    tree = parse_tree(mjcf_path(sub))
    got = [n for n, _, _ in tree]
    if got != names:
        raise ValueError(f"{sub}: body order differs from the reference skeleton")
    off = np.array([o for _, _, o in tree])
    return np.linalg.norm(off[child], axis=1)


def available_mjcf_subjects(real_only=True):
    """Subject ids that have a per-subject MJCF. real_only drops synthetic sub100+."""
    subs = []
    for p in sorted(glob.glob(os.path.join(MJCF_DIR, "smplx_omomo_sub*.xml"))):
        s = re.search(r"(sub\d+)\.xml$", p).group(1)
        if real_only and int(s[3:]) > 17:
            continue
        subs.append(s)
    return sorted(subs, key=lambda s: int(s[3:]))


def clip_bone_lengths(path):
    """(mean, std) bone lengths over the frames of one clip, in metres.

    std is the rigidity check: for a real rigid skeleton it is ~0. If it is large,
    `body_pos` is not a rigid body (e.g. SMPL-X regressed joints, which move with
    pose blendshapes) and any identity claim from the mean is weak -- callers must
    check it rather than assume.
    """
    names, parent, child, pa = _topology()
    data = torch.load(path, map_location="cpu")
    bp = data.detach()[:, I_BODY].reshape(-1, N_BODIES, 3).double().numpy()
    L = np.linalg.norm(bp[:, child] - bp[:, pa], axis=2)
    return L.mean(0), L.std(0)


def subject_bone_lengths(data_dir, sub, max_clips=3):
    """(mean, max_std, n_clips) pooled over up to `max_clips` of one subject's clips."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{sub}_*.pt")))[:max_clips]
    if not files:
        return None, None, 0
    means, stds = zip(*(clip_bone_lengths(f) for f in files))
    return np.mean(means, axis=0), float(np.max(stds)), len(files)


def subjects_in(data_dir):
    """Subject ids present in a motion directory, numerically sorted."""
    subs = {re.match(r"(sub\d+)_", os.path.basename(p)).group(1)
            for p in glob.glob(os.path.join(data_dir, "*.pt"))}
    return sorted(subs, key=lambda s: int(s[3:]))


def match_errors(measured, candidates):
    """{candidate: mean |bone length| error in MILLIMETRES} for one measured vector."""
    return {c: float(np.abs(measured - mjcf_bone_lengths(c)).mean() * 1000)
            for c in candidates}


def best_match(measured, candidates):
    """(best_candidate, its error in mm) -- the body this reference is expressed on."""
    errs = match_errors(measured, candidates)
    best = min(errs, key=errs.get)
    return best, errs[best]


def identity_table(data_dir, candidates=None, max_clips=3):
    """{subject: dict(measured, max_std_mm, n_clips, errors, best, best_err)} for a dataset."""
    cands = candidates or available_mjcf_subjects()
    out = {}
    for sub in subjects_in(data_dir):
        m, sd, n = subject_bone_lengths(data_dir, sub, max_clips)
        if m is None:
            continue
        errs = match_errors(m, cands)
        best = min(errs, key=errs.get)
        out[sub] = dict(measured=m, max_std_mm=sd * 1000, n_clips=n,
                        errors=errs, best=best, best_err=errs[best])
    return out

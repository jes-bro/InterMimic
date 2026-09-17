"""Pure helpers for Arm A of the g3 student (body wire + contrastive twins).
No torch / Isaac Gym imports so tests run locally.

BODY FEATURES. The "body wire" gives the policy its embodiment explicitly: the
per-body offsets read from the subject's MJCF (each <body pos="x y z"> is the
child's origin in its parent's frame, i.e. the bone vector). 52 bodies x 3 =
156 numbers, model-agnostic (no SMPL betas frame problem -- OMOMO betas are
per-gender fits and the CARI4D people's are SMPL-H, so a betas vector would not
be one space; a rig's bone vectors are the same kind of number for every body).

TWIN ENVS. The contrastive term needs positive pairs: the SAME clip at the SAME
frame on DIFFERENT bodies. Two facts of the task decide who can be twins:
  * env e physically owns object e % n_objects (the sampler only offers it
    clips of that object), so twins must share an object bucket;
  * env e's body is e % n_bodies (round-robin over subjectBodies).
Pairing e with e + n_objects satisfies the first; the second holds whenever
n_objects is not a multiple of n_bodies (13 objects / 43 bodies; 132 / 43),
and `twin_partners` refuses otherwise rather than silently pairing an env with
its own body.
"""
import xml.etree.ElementTree as ET


def mjcf_body_offsets(path):
    """[(body_name, (x, y, z)), ...] in document order (= the task's body order)."""
    root = ET.parse(path).getroot()
    out = []
    for b in root.iter("body"):
        pos = b.get("pos", "0 0 0").split()
        if len(pos) != 3:
            raise ValueError(f"{path}: body {b.get('name')!r} has a non-3D pos {b.get('pos')!r}")
        out.append((b.get("name"), tuple(float(v) for v in pos)))
    if not out:
        raise ValueError(f"{path}: no <body> elements")
    return out


def body_feature_matrix(paths, expected_bodies=52):
    """Rows = bodies in `paths` order, columns = the 3*expected_bodies flattened
    offsets. Every rig must have the same body names in the same order, or the
    columns would mean different things for different bodies."""
    rows, names0 = [], None
    for p in paths:
        offs = mjcf_body_offsets(p)
        names = [n for n, _ in offs]
        if len(names) != expected_bodies:
            raise ValueError(f"{p}: {len(names)} bodies, expected {expected_bodies}")
        if names0 is None:
            names0 = names
        elif names != names0:
            raise ValueError(f"{p}: body order differs from {paths[0]}")
        rows.append([v for _, xyz in offs for v in xyz])
    return rows


def twin_partners(num_envs, n_objects, n_bodies):
    """partner[e] = the env paired with e, or -1. Pairs are (e, e + n_objects)
    for e in even blocks of n_objects, so both envs own the same object; each
    env is in at most one pair. Refuses if any pair would share a body."""
    if n_objects <= 0 or n_bodies <= 0:
        raise ValueError("n_objects and n_bodies must be positive")
    partner = [-1] * num_envs
    for e in range(num_envs):
        block = e // n_objects
        if block % 2 == 0 and e + n_objects < num_envs:
            p = e + n_objects
            if e % n_bodies == p % n_bodies:
                raise ValueError(
                    f"[twins] envs {e} and {p} share body {e % n_bodies}: n_objects={n_objects} "
                    f"is a multiple of n_bodies={n_bodies}; twin pairing needs them coprime-ish")
            partner[e], partner[p] = p, e
    return partner


def twin_pairs(partner):
    """[(a, b), ...] with a < b, one entry per pair."""
    return [(e, p) for e, p in enumerate(partner) if p > e]

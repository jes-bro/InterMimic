"""Pure-Python helpers for g4 distillation: teachers routed by the clip's OBJECT.

WHY A SECOND MANIFEST KIND. The g3 student picks each env's teacher by its
clip's SOURCE subject (utils/distill_g3.py): the per-source teachers own one
source each, the activity teachers own their people. The gen4 OBJECT-GROUP
teachers (boxes / stands / chairs / tables, each trained on ALL 13 OMOMO
sources) cannot be told apart that way -- a sub6 clip belongs to whichever
teacher owns its object. So a g4 manifest names OBJECTS, and the task
(env/tasks/intermimic_distill_g4.py) routes through the clip's object index.

    teachers:
      - file: chairs.pth
        objects: [whitechair, woodchair]
        from: checkpoints/smplx_teacher_gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0/nn/mimic_00044000.pth
        epoch: 44000
      - file: boxes.pth
        objects: [largebox, smallbox, plasticbox, suitcase, trashcan, monitor]
        ...

Written by scripts/collect_g3_teachers.py --gen4-objects. `from` / `epoch` are
provenance: the student inherits whatever epoch each teacher had reached
(latest, Jess 2026-10-15), and this records it. No torch / Isaac Gym imports,
so tests/test_distill_g4.py runs anywhere.
"""
import os

import yaml

MANIFEST_NAME = "teachers.yaml"


class ObjectTeacherEntry:
    __slots__ = ("file", "objects", "origin", "epoch")

    def __init__(self, file, objects, origin=None, epoch=None):
        self.file = file
        self.objects = list(objects)
        self.origin = origin
        self.epoch = epoch

    def __repr__(self):
        return f"ObjectTeacherEntry({self.file}, objects={self.objects}, epoch={self.epoch})"


def load_object_manifest(teacher_dir):
    """Read and validate <teacher_dir>/teachers.yaml as an OBJECT-keyed manifest.

    Refuses: missing manifest, empty list, an entry without 'file' or 'objects',
    an entry carrying 'sources' (that is a g3 manifest -- the wrong task would
    route by source and silently mis-assign every env), a listed file that is
    not on disk, an object claimed by two teachers, a non-string or '_'-bearing
    object name (the task reads the object as one '_' token of the filename).
    Every refusal names the offender."""
    path = os.path.join(teacher_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[distill-g4] no {MANIFEST_NAME} in teacherPolicy dir {teacher_dir}; "
            f"write it with scripts/collect_g3_teachers.py --gen4-objects ...")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    raw = doc.get("teachers")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"[distill-g4] {path}: 'teachers' must be a non-empty list")

    entries, owner = [], {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "file" not in item:
            raise ValueError(f"[distill-g4] {path}: entry {i} needs 'file': {item!r}")
        if "sources" in item:
            raise ValueError(
                f"[distill-g4] {path}: entry {i} ({item['file']}) is keyed by 'sources' -- that is a g3 "
                f"(per-source) manifest. InterMimicDistillG4 routes by OBJECT; use --task "
                f"InterMimicDistillG3 for it, or collect the gen4 object teachers with --gen4-objects")
        if "objects" not in item:
            raise ValueError(f"[distill-g4] {path}: entry {i} ({item['file']}) needs 'objects'")
        fpath = os.path.join(teacher_dir, item["file"])
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"[distill-g4] {path}: entry {i} file not found: {fpath}")
        objs = item["objects"]
        if (not isinstance(objs, list) or not objs
                or any(not isinstance(o, str) or not o or "_" in o for o in objs)):
            raise ValueError(f"[distill-g4] {path}: entry {i} 'objects' must be a non-empty list of "
                             f"object names (non-empty strings, no '_'), got {objs!r}")
        for o in objs:
            if o in owner:
                raise ValueError(f"[distill-g4] {path}: object {o} claimed by both {owner[o]} and {item['file']}")
            owner[o] = item["file"]
        entries.append(ObjectTeacherEntry(item["file"], objs, item.get("from"), item.get("epoch")))
    return entries


def build_object_lookup(entries, object_names):
    """Map object index (position in the task's sorted `object_name` list) ->
    teacher index (position in `entries`).

    Returns (lookup, unused): lookup is aligned with object_names; every loaded
    object must have a teacher, so an object with none raises here, at startup,
    with the list, rather than at the first reset. unused lists teachers no
    loaded object needs (reported, not an error: a 4-teacher set serving a
    chairs-only data filter is legitimate)."""
    obj_to_idx = {}
    for i, e in enumerate(entries):
        for o in e.objects:
            obj_to_idx[o] = i
    missing = [o for o in object_names if o not in obj_to_idx]
    if missing:
        raise ValueError(
            f"[distill-g4] {len(missing)} object(s) in the data have no teacher: {missing}. "
            f"Add them to teachers.yaml or drop their clips (dataObjects).")
    lookup = [obj_to_idx[o] for o in object_names]
    used = set(lookup)
    unused = [entries[i].file for i in range(len(entries)) if i not in used]
    return lookup, unused

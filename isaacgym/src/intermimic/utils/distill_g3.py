"""Pure-Python helpers for g3 distillation (no Isaac Gym / torch imports, so
tests/test_distill_g3.py can exercise them locally).

WHY A NEW PATH. InterMimic_All was written for the paper's data model: the
teacher is queried on the SOURCE mocap (it overwrites hoi_data with the source
clips) and the student reads a flat OMOMO_retarget dir with hardcoded horizons
(MLP [1,16] / transformer [0,1,4,16]). Every g3 teacher was trained on PER-BODY
contact-retargeted references (intermimic.py retargetedMotionDir) with
obsHorizons [1,4,7,10,13,16], and its routing key is the SOURCE subject while
the parent's `dataset_index` is aliased to the TARGET body. Three silent
mismatches -- wrong teacher obs, wrong student obs width, wrong teacher per env.
InterMimicDistillG3 (env/tasks/intermimic_distill_g3.py) keeps the parent's
load and observation code and only ADDS the student obs + teacher query; the
arithmetic and the manifest rules live here so they can be pinned by tests.

TEACHER MANIFEST. The teacherPolicy dir must hold `teachers.yaml`:

    teachers:
      - file: sub2.pth
        sources: [2]
        from: checkpoints/smplx_teacher_g3_omomo_geoall__f0/nn/mimic_00012000.pth
        epoch: 12000
      - file: bball7.pth
        sources: [401, 402, 404, 409, 411, 412, 458]
        from: ...
        epoch: ...

One checkpoint may serve several sources (an activity teacher trained on 7
people), which the old `sub{S}.pth` filename convention could not express
without loading 7 copies. `from` / `epoch` are provenance only -- the student
inherits whatever epoch each teacher was at, and this records it.
"""
import os

import yaml


class TeacherEntry:
    __slots__ = ("file", "sources", "origin", "epoch")

    def __init__(self, file, sources, origin=None, epoch=None):
        self.file = file
        self.sources = list(sources)
        self.origin = origin
        self.epoch = epoch

    def __repr__(self):
        return f"TeacherEntry({self.file}, sources={self.sources}, epoch={self.epoch})"


MANIFEST_NAME = "teachers.yaml"


def load_teacher_manifest(teacher_dir):
    """Read and validate <teacher_dir>/teachers.yaml. Returns [TeacherEntry].

    Refuses: missing manifest, empty list, a listed file that is not on disk,
    a source claimed by two teachers, non-integer sources. Every refusal names
    the offender -- a bad manifest must not become a wrong-teacher run."""
    path = os.path.join(teacher_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[distill-g3] no {MANIFEST_NAME} in teacherPolicy dir {teacher_dir}; "
            f"write it with scripts/collect_g3_teachers.py")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    raw = doc.get("teachers")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"[distill-g3] {path}: 'teachers' must be a non-empty list")

    entries, owner = [], {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "file" not in item or "sources" not in item:
            raise ValueError(f"[distill-g3] {path}: entry {i} needs 'file' and 'sources': {item!r}")
        fpath = os.path.join(teacher_dir, item["file"])
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"[distill-g3] {path}: entry {i} file not found: {fpath}")
        srcs = item["sources"]
        if not isinstance(srcs, list) or not srcs or any(
                not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in srcs):
            raise ValueError(f"[distill-g3] {path}: entry {i} 'sources' must be a non-empty "
                             f"list of non-negative ints, got {srcs!r}")
        for s in srcs:
            if s in owner:
                raise ValueError(f"[distill-g3] {path}: source sub{s} claimed by both "
                                 f"{owner[s]} and {item['file']}")
            owner[s] = item["file"]
        entries.append(TeacherEntry(item["file"], srcs, item.get("from"), item.get("epoch")))
    return entries


def build_source_lookup(entries, present_sources):
    """Map source subject id -> teacher index (position in `entries`).

    Returns (lookup, unused) where lookup is a list of length max(all ids)+1
    with -1 for ids no teacher serves, and unused lists teachers none of the
    present sources need. Raises if any PRESENT source has no teacher: the old
    code would KeyError at the first reset; better at startup, with the list."""
    present = sorted({int(s) for s in present_sources})
    src_to_idx = {}
    for i, e in enumerate(entries):
        for s in e.sources:
            src_to_idx[s] = i
    missing = [s for s in present if s not in src_to_idx]
    if missing:
        raise ValueError(
            f"[distill-g3] {len(missing)} source(s) in the data have no teacher: "
            f"{['sub%d' % s for s in missing]}. Add them to teachers.yaml or drop their clips.")
    n = max(list(src_to_idx) + present) + 1
    lookup = [-1] * n
    for s, i in src_to_idx.items():
        lookup[s] = i
    used = {src_to_idx[s] for s in present}
    unused = [entries[i].file for i in range(len(entries)) if i not in used]
    return lookup, unused


def validate_horizons(h, name):
    if (not isinstance(h, (list, tuple)) or not h
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in h)
            or len(set(h)) != len(h)):
        raise ValueError(f"[distill-g3] {name}={h!r}: expected a non-empty list of "
                         f"distinct non-negative ints (delta_t frames)")
    return list(h)


def student_obs_width(teacher_num_obs, teacher_horizons, student_horizons, uses_betas):
    """numObsRetarget the student MUST declare, from the teacher's obs layout.

    The parent stacks one `width`-wide block per horizon (obs_buf = width x
    len(obsHorizons)); the student stacks the same block over ITS horizons. With
    betas the width arithmetic differs per policy type and no g3 arm uses them,
    so that case is refused rather than guessed."""
    if uses_betas:
        raise ValueError("[distill-g3] betas conditioning is not supported on the g3 "
                         "distill path (no g3 teacher uses it); remove betas_file")
    th = validate_horizons(teacher_horizons, "obsHorizons")
    sh = validate_horizons(student_horizons, "studentObsHorizons")
    if teacher_num_obs % len(th) != 0:
        raise ValueError(f"[distill-g3] numObs {teacher_num_obs} is not a multiple of "
                         f"len(obsHorizons)={len(th)}; the per-horizon width is undefined")
    width = teacher_num_obs // len(th)
    return width * len(sh)


def token_layout(input_shape, num_tokens, readout_token):
    """Transformer obs layout: (obs_per_token, num_tokens, readout_token).

    The builder historically hardcoded 4 tokens and read the encoder output at
    index 1 (the delta_t=1 token of [0,1,4,16]). Both are now parameters so a
    6-horizon student ([1,4,7,10,13,16], readout 0 = its delta_t=1 token) can
    exist; defaults reproduce the old network exactly."""
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 1:
        raise ValueError(f"[transformer] num_tokens must be a positive int, got {num_tokens!r}")
    if not isinstance(readout_token, int) or isinstance(readout_token, bool) \
            or not (0 <= readout_token < num_tokens):
        raise ValueError(f"[transformer] readout_token {readout_token!r} out of range "
                         f"for num_tokens {num_tokens}")
    if input_shape % num_tokens != 0:
        raise ValueError(f"[transformer] obs size {input_shape} is not divisible by "
                         f"num_tokens {num_tokens}")
    return input_shape // num_tokens, num_tokens, readout_token

"""Expert retrieval over CARI4D 4D reconstructions, stored in an on-disk Qdrant database.

PULLED INTO InterMimic 2026-10-16 (Jess: retrieval + inference in one repo) from
jes-bro/Qwen3-VL-Embedding, branch recon-retrieval, commit 57fa48f, with the
database it was built with (scripts/recon_retrieval_db/, 9 MB: 33 Reverse Layup
/ 17 Mikan / 23 CPR / 62 soccer expert clips + 801 expert takes as payload).
Three changes from that commit, nothing else: DEFAULT_DB points at the copied
database; load_takes() tolerates a missing takes.json (a collaborator querying
on a machine without EgoExo4D: participant exclusion and camera views then
degrade with a note, retrieval itself does not need it); --rated_only restricts
hits to participants rated Early / Intermediate / Late Expert (Jess's rule:
'unlabelled' is not an expert). Needs `pip install qdrant-client` (plus numpy,
torch) in the env that runs it. scripts/novice_rollout.py calls `query` to
pick the expert clip for a novice.

Given a learner's reconstruction, find expert reconstructions of the same drill
with similar MOTION, similar HUMAN SIZE and a similar OBJECT SIZE; line the two
motions up over the longest stretch they share; pick the expert CAMERA whose
view of the body best matches the learner's camera; and cut the matched
stretch out of both videos, plus a side-by-side, so the match can be watched.

    # build (or rebuild) the database from exported reconstruction folders
    python recon_retrieval.py build

    # retrieve for one or more learner clips exported in the same folder layout
    python recon_retrieval.py query --clip_dir <clip folder> [<clip folder> ...] --task "Mikan Layup" --top 5

    # leave-one-person-out sanity checks on the expert clips themselves
    python recon_retrieval.py selftest

    # drill names and counts, for spelling --task
    python recon_retrieval.py list

On the Stanford cluster every path defaults to the shared locations (videos and
calibrations under /vision/group/egoexo4d/takes, exports under
/simurgh2/projects/ret-hoi/cari4d_exports); see slurm_recon_retrieval.sh.

WHAT GOES IN. Two Qdrant collections plus one small file per clip:

  experts  one point per tarred expert take (every take of every basketball,
           soccer and CPR subject in CARI4D's handoff/MANIFEST.tsv), payload
           only: scenario, drill, take, participant, participant tier, the
           take's own proficiency label, gender, cameras, has_recon. Takes not
           reconstructed yet live here so the database lists every expert.
  clips    one point per reconstructed clip, with named vectors
             body    the SMPL-H shape parameters (10-d)
             object  the metric object mesh's bounding-box extents, sorted
                     largest first (3-d, metres)
           and a payload with the same labels plus the clip's take frames.
  frames/<clip id>.npz
           the clip's per-frame motion features (standardised) and, when the
           take's calibration was found, every camera's direction from the body
           on every frame. Whole-sequence motion does not fit a fixed-size
           vector, so it lives beside the database rather than inside it.

Two collections rather than one because Qdrant's local mode fails any vector
query once a collection holds a point without vectors, even when a filter
excludes it.

MOTION, per frame: 21 body-joint rotations as 6D, root speed and root turn
rate, standardised over all expert frames. None of it depends on where the
camera stood, so experts filmed from different cameras still match.

HOW A QUERY SCORES. Candidates are every reconstructed clip of the drill
(minus exclusions). For each, the whole shorter clip slides along the longer
one at the same speed, and the offset with the highest mean per-frame cosine
similarity wins: the match is as long as the shorter clip. Body and object
ranks come from Qdrant nearest-neighbour search over the same candidates. The
three ranks are fused with weighted reciprocal-rank fusion (--w_motion,
--w_body, --w_object), and matches covering overlapping frames of the same
take are collapsed so the results are different moments.

CAMERA VIEW. For every frame and camera, the camera's direction from the pelvis
in a body-heading frame: azimuth (0 = straight in front of the person, +90 = on
their left, 180 = behind) and elevation. Expert cameras are ranked by the mean
angle to the learner camera's direction over the aligned frames. Checked
against footage on two CPR takes (every camera's picture matched its
computed azimuth, left/right included). Needs gopro_calibs.csv for both takes;
without it the result falls back to the take's best_exo and says so.

VIDEO OUTPUT, per query under --out_dir/<learner clip>/:
  learner_<clip>_<cam>_f<first>-<last>.mp4       the learner's matched stretch
  rankNN_<expert clip>_<cam>_f<first>-<last>.mp4  the expert's matched stretch
  rankNN_side_by_side.mp4                         learner left, expert right
  matches.json                                    everything printed, as data
Frames are cut by index with ffmpeg's select filter, as CARI4D's
slurm_trim_clips.sh does: take frame N is frame N of
frame_aligned_videos/<cam>.mp4. The downscaled/448 copies are used when present
(same frame count as the 4K originals, verified), and every output is
frame-counted afterwards.

CONVENTIONS, checked against the data rather than assumed: a bundle's pr dict
is in the rectified pipeline camera's frame, and rectification uses R=I, so
gopro_calibs.csv's camera-to-world pose for pipe_cam maps it to world
unchanged (CARI4D prep/static_prop.py). World is z-up and SMPL body forward is
+z. Bundle frame i is take frame lo + i.

OBJECT SIZE comes from the exported mesh, not meta.json's largest_m. They agree
exactly for basketball and soccer, but the CPR manikin meshes are 1.4-1.7x
largest_m (0.64 m against 0.39-0.44 m), and the mesh is what the final solve
used. largest_m is the fallback when a clip has no mesh.

Needs numpy, torch and qdrant-client (plus ffmpeg/ffprobe for videos). No SMPL
model files.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import os.path as osp
import pickle
import re
import shutil
import subprocess
import sys
import types
import uuid

import numpy as np
import torch
from qdrant_client import QdrantClient, models

CLUSTER_TAKES = "/vision/group/egoexo4d/takes"
CLUSTER_PROJECT = "/simurgh2/projects/ret-hoi"
ON_CLUSTER = osp.isdir(CLUSTER_TAKES)
HOME = osp.expanduser("~")
HERE = osp.dirname(osp.abspath(__file__))
ROOT_NAMES = ("cari4d_bball7", "cari4d_cpr", "cari4d_soccer")

if ON_CLUSTER:
    DEFAULT_ROOTS = [osp.join(CLUSTER_PROJECT, "cari4d_exports", d) for d in ROOT_NAMES]
    DEFAULT_TAKES = "/vision/group/egoexo4d/takes.json"
    DEFAULT_EGOEXO = CLUSTER_TAKES
    DEFAULT_MANIFEST = osp.join(CLUSTER_PROJECT, "CARI4D", "handoff", "MANIFEST.tsv")
    DEFAULT_ANNOT = osp.join(CLUSTER_PROJECT, "annotations-egoexo4d")
else:
    DEFAULT_ROOTS = [osp.join(HOME, d) for d in ROOT_NAMES]
    DEFAULT_TAKES = osp.join(HOME, "egoexo4d", "takes.json")
    DEFAULT_EGOEXO = osp.join(HOME, "egoexo4d")
    DEFAULT_MANIFEST = osp.join(HOME, "CARI4D", "handoff", "MANIFEST.tsv")
    DEFAULT_ANNOT = osp.join(HOME, "Downloads", "annotations-egoexo4d")
DEFAULT_DB = osp.join(HERE, "recon_retrieval_db")
DEFAULT_OUT = osp.join(HERE, "retrieval_out")

EXPERTS = "experts"
CLIPS = "clips"
RATED_TIERS = ("Early Expert", "Intermediate Expert", "Late Expert")   # the only tiers that count as experts
FPS = 30.0
BODY_JOINTS = list(range(1, 22))      # SMPL joints 1..21: everything but the pelvis and hand roots
ROOT_WEIGHT = 4.0                     # root speed / turn rate are 2 of 128 dims; lift them to count
RRF_K = 60                            # reciprocal-rank fusion constant
VIDEO_HEIGHT = 448

# The manifest's scenario name -> which takes.json takes belong to it.
SCENARIOS = {
    "basketball": lambda t: t.get("parent_task_name") == "Basketball",
    "soccer": lambda t: t.get("parent_task_name") == "Soccer",
    "cpr": lambda t: t.get("task_name") == "First Aid - CPR",
}
NAMESPACE = uuid.UUID("6f0b2f0e-5a3c-4f51-9a52-2f1c1f6b7a10")


# --------------------------------------------------------------------------- loading


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that stubs classes it cannot import.

    Bundles carry training-state objects whose classes pull in CARI4D's whole
    training stack; only plain tensors, strings and lists are read here.
    Mirrors CARI4D's prep/compare_ball_stages.TolerantUnpickler.
    """

    def find_class(self, module, name):
        """Resolve normally; on ImportError/AttributeError return an inert stub class."""
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {"__init__": lambda self, *a, **k: None,
                                   "__setstate__": lambda self, s: None})


TOLERANT_PICKLE = types.ModuleType("tolerant_pickle")
TOLERANT_PICKLE.Unpickler = _TolerantUnpickler
TOLERANT_PICKLE.load = pickle.load


def load_bundle(pth):
    """The 'pr' dict of a CARI4D result bundle as numpy: pose (T, 72), trans (T, 3), betas (10,)."""
    d = torch.load(pth, map_location="cpu", weights_only=False, pickle_module=TOLERANT_PICKLE)
    pr = d["pr"]
    return {
        "pose": pr["smpl_pose"].detach().float().numpy()[:, :72],
        "trans": pr["smpl_t"].detach().float().numpy(),
        "betas": pr["betas"].detach().float().numpy().reshape(-1, 10).mean(0),
    }


def load_takes(path):
    """takes.json as {take_name: take record}; {} with a note when the file is
    absent (a machine without EgoExo4D). Retrieval reads the expert takes from
    the database payload, so a query still works; only --exclude_participant
    self and the camera-view ranking need the file."""
    if not osp.isfile(path):
        print(f"note: no takes.json at {path}; participant exclusion by 'self' and camera views are unavailable")
        return {}
    with open(path) as f:
        return {t["take_name"]: t for t in json.load(f)}


def load_take_labels(annot_dir):
    """{take_uid: proficiency label} from proficiency_demonstrator_{train,val}.json, or None if absent."""
    out, found = {}, False
    for split in ("train", "val"):
        p = osp.join(annot_dir, f"proficiency_demonstrator_{split}.json")
        if osp.isfile(p):
            found = True
            with open(p) as f:
                for a in json.load(f)["annotations"]:
                    out[a["take_uid"]] = a["proficiency_score"]
    return out if found else None


def load_manifest(path):
    """Tarred expert subjects for basketball, soccer and CPR from CARI4D's handoff/MANIFEST.tsv."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["scenario"] in SCENARIOS]
    if not rows:
        raise SystemExit(f"no basketball/soccer/cpr rows in {path}")
    return rows


def read_extrinsics(path):
    """gopro_calibs.csv -> {cam_uid: (R_wc, t_wc)}, camera-to-world (same columns CARI4D reads)."""
    cams = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            x, y, z, w = (float(row[k]) for k in ("qx_world_cam", "qy_world_cam", "qz_world_cam", "qw_world_cam"))
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            t = np.array([float(row[k]) for k in ("tx_world_cam", "ty_world_cam", "tz_world_cam")])
            cams[row["cam_uid"]] = (R, t)
    return cams


def calib_path(egoexo_root, take):
    """Where a take's gopro_calibs.csv lives: <root>/<take>/trajectory/gopro_calibs.csv."""
    return osp.join(egoexo_root, take, "trajectory", "gopro_calibs.csv")


def exo_cams(take_rec):
    """The take's exocentric camera names (cam01.., or gp01.. on upenn rigs), sorted."""
    keys = (take_rec or {}).get("frame_aligned_videos") or {}
    return sorted(k for k in keys if k.startswith(("cam", "gp")))


def drill_of(task_name):
    """Short drill name used for filtering: 'Mikan Layup', 'Penalty Kick', 'CPR', ..."""
    if task_name == "First Aid - CPR":
        return "CPR"
    return re.sub(r"^(Basketball|Soccer) Drills - ", "", task_name or "")


def drill_key(name):
    """Case- and space-insensitive filter key for a drill name."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def point_id(kind, name):
    """Stable UUID for a point, so rebuilds overwrite rather than duplicate."""
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{name}"))


# --------------------------------------------------------------------------- clip folders


def find_clip_dirs(roots):
    """Every exported clip folder (holding meta.json and a .pth) under the given roots."""
    out = []
    for root in roots:
        for m in sorted(glob.glob(osp.join(osp.expanduser(root), "*", "meta.json"))):
            d = osp.dirname(m)
            if glob.glob(osp.join(d, "*.pth")):
                out.append(d)
    return out


def read_clip_meta(clip_dir):
    """Normalised metadata for one clip folder, tolerating the older export format.

    Returns clip, take, pipe_cam, gender, lo (take frame of bundle frame 0),
    section_of, largest_m, pth and obj paths.
    """
    with open(osp.join(clip_dir, "meta.json")) as f:
        m = json.load(f)
    lo = m.get("lo")
    if lo is None:
        # Older exports keep the window only as text: "... frames 1162-1362".
        hit = re.search(r"frames\s+(\d+)-(\d+)", json.dumps(m.get("window", {})))
        lo = int(hit.group(1)) if hit else None
    scale = ((m.get("stages") or {}).get("solve") or {}).get("scale") or {}
    pths = sorted(glob.glob(osp.join(clip_dir, "*.pth")))
    objs = sorted(glob.glob(osp.join(clip_dir, "*.obj")))
    return {
        "clip": m.get("clip") or m.get("seq") or osp.basename(clip_dir),
        "take": m["take"],
        "pipe_cam": m.get("pipe_cam"),
        "gender": m.get("gender"),
        "lo": lo,
        "section_of": m.get("section_of"),
        "largest_m": scale.get("largest_m"),
        "pth": pths[0],
        "obj": objs[0] if objs else None,
        "clip_dir": clip_dir,
    }


def object_extents(meta):
    """Metric object size as bounding-box extents sorted largest first, from the mesh, else largest_m."""
    if meta["obj"]:
        v = []
        with open(meta["obj"]) as f:
            for line in f:
                if line.startswith("v "):
                    v.append(line.split()[1:4])
        if v:
            v = np.asarray(v, dtype=np.float32)
            return np.sort(v.max(0) - v.min(0))[::-1].astype(float)
    if meta["largest_m"]:
        s = float(meta["largest_m"])
        return np.array([s, s, s])
    return None


# --------------------------------------------------------------------------- features


def rotvec_to_matrix(r):
    """Axis-angle (..., 3) to rotation matrices (..., 3, 3) by Rodrigues' formula."""
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(theta, 1e-8)
    theta = theta[..., None]
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def frame_features(pose, trans):
    """Per-frame, camera-independent motion features, shape (T, 128).

    21 body-joint rotations as 6D (first two matrix columns), then root speed
    (m/s) and root turn rate (rad/s), both frame-invariant magnitudes.
    """
    T = len(pose)
    rots = rotvec_to_matrix(pose.reshape(T, 24, 3)[:, BODY_JOINTS])
    six = rots[..., :, :2].reshape(T, -1)
    vel = np.zeros(T)
    turn = np.zeros(T)
    if T > 1:
        vel[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=1) * FPS
        g = rotvec_to_matrix(pose[:, :3])
        rel = np.einsum("tij,tik->tjk", g[:-1], g[1:])
        cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
        turn[1:] = np.arccos(cos) * FPS
        vel[0], turn[0] = vel[1], turn[1]
    return np.concatenate([six, vel[:, None], turn[:, None]], axis=1)


def standardise(feats, norm):
    """Standardise per-frame features with the database's mean/std and up-weight the root terms."""
    x = (feats - norm["mean"]) / norm["std"]
    x[:, -2:] *= ROOT_WEIGHT
    return x.astype(np.float32)


def view_directions(pose, trans, R_wc, t_wc, cams):
    """Every camera's direction from the pelvis on every frame, in a body-heading frame.

    Returns (cam names, (C, T, 2) array of [azimuth_deg, elevation_deg]):
    azimuth 0 is straight in front of the person, +90 on their left, +-180 behind.
    """
    g = R_wc @ rotvec_to_matrix(pose[:, :3])                  # body-to-world per frame
    root = trans @ R_wc.T + t_wc
    fwd = g[:, :, 2]
    heading = fwd[:, :2] / np.maximum(np.linalg.norm(fwd[:, :2], axis=1, keepdims=True), 1e-8)
    names = sorted(cams)
    out = np.zeros((len(names), len(pose), 2), dtype=np.float32)
    for c, cam in enumerate(names):
        d = cams[cam][1][None] - root
        dxy = d[:, :2]
        out[c, :, 0] = np.degrees(np.arctan2(heading[:, 0] * dxy[:, 1] - heading[:, 1] * dxy[:, 0],
                                             (heading * dxy).sum(1)))
        out[c, :, 1] = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(dxy, axis=1)))
    return names, out


def unit(ae):
    """[azimuth_deg, elevation_deg] rows to unit direction vectors."""
    ae = np.radians(np.asarray(ae, dtype=float))
    return np.stack([np.cos(ae[..., 1]) * np.cos(ae[..., 0]),
                     np.cos(ae[..., 1]) * np.sin(ae[..., 0]),
                     np.sin(ae[..., 1])], axis=-1)


def clip_record(meta, takes, egoexo_root):
    """Everything retrieval needs from one clip folder: raw motion features, sizes and camera views."""
    b = load_bundle(meta["pth"])
    T = len(b["pose"])
    rec = {"meta": meta, "take_rec": takes.get(meta["take"], {}), "T": T,
           "feats": frame_features(b["pose"], b["trans"]), "betas": b["betas"],
           "object": object_extents(meta), "cams": [], "views": None, "view_note": ""}
    cpath = calib_path(egoexo_root, meta["take"])
    if not osp.isfile(cpath):
        rec["view_note"] = f"no calibration at {cpath}"
    elif meta["pipe_cam"] is None:
        rec["view_note"] = "meta.json has no pipe_cam"
    else:
        cams = read_extrinsics(cpath)
        if meta["pipe_cam"] not in cams:
            rec["view_note"] = f"{meta['pipe_cam']} not in {cpath}"
        else:
            R_wc, t_wc = cams[meta["pipe_cam"]]
            rec["cams"], rec["views"] = view_directions(b["pose"], b["trans"], R_wc, t_wc, cams)
    return rec


# --------------------------------------------------------------------------- build


def expert_takes(manifest, takes):
    """(take record, manifest row) for every non-dropped take of every tarred subject in its scenario."""
    by_pid = collections.defaultdict(list)
    for t in takes.values():
        if t.get("participant_uid") is not None and not t.get("is_dropped"):
            by_pid[t["participant_uid"]].append(t)
    out = []
    for row in manifest:
        mine = [t for t in by_pid[int(row["participant"])] if SCENARIOS[row["scenario"]](t)]
        out += [(t, row) for t in sorted(mine, key=lambda t: t["take_idx"])]
    return out


def base_payload(take, row, labels):
    """Labels shared by the experts and clips collections."""
    return {
        "take": take["take_name"],
        "take_uid": take["take_uid"],
        "scenario": row["scenario"],
        "task": take["task_name"],
        "drill": drill_of(take["task_name"]),
        "drill_key": drill_key(drill_of(take["task_name"])),
        "participant": int(row["participant"]),
        "participant_tier": row["proficiency"],
        "take_label": "n/a" if labels is None else labels.get(take["take_uid"], "unlabelled"),
        "gender": row.get("gender"),
        "university": take.get("university_name"),
        "cameras": exo_cams(take),
        "best_exo": take.get("best_exo"),
        "archive": row.get("archive"),
    }


def cmd_build(args):
    """Rebuild the collections and per-clip frame files from scratch in the on-disk database."""
    takes = load_takes(args.takes_json)
    manifest = load_manifest(args.manifest)
    labels = load_take_labels(args.annot_dir)
    if labels is None:
        print(f"note: no proficiency_demonstrator_*.json under {args.annot_dir}; take labels will read 'n/a' "
              f"(participant tiers still come from the manifest)")
    experts = expert_takes(manifest, takes)
    row_of_take = {t["take_name"]: row for t, row in experts}

    metas, skipped = [], []
    for d in find_clip_dirs(args.recon_roots):
        meta = read_clip_meta(d)
        (metas if meta["take"] in row_of_take else skipped).append(meta)
    if skipped:
        print(f"skipping {len(skipped)} clip(s) whose take is not a tarred expert take: "
              f"{', '.join(m['clip'] for m in skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    if not metas:
        raise SystemExit("no reconstructed expert clips found under " + " ".join(args.recon_roots))

    records = []
    for i, meta in enumerate(metas, 1):
        records.append(clip_record(meta, takes, args.egoexo_root))
        if i % 10 == 0 or i == len(metas):
            print(f"  read {i}/{len(metas)} clips", flush=True)

    allf = np.concatenate([r["feats"] for r in records])
    norm = {"mean": allf.mean(0), "std": np.maximum(allf.std(0), 1e-3)}

    if osp.isdir(args.db):
        shutil.rmtree(args.db)
    os.makedirs(osp.join(args.db, "frames"))
    np.savez(osp.join(args.db, "motion_norm.npz"), mean=norm["mean"], std=norm["std"], root_weight=ROOT_WEIGHT)
    client = QdrantClient(path=args.db)

    recon_takes = {r["meta"]["take"] for r in records}
    client.create_collection(EXPERTS, vectors_config={})
    client.upsert(EXPERTS, [
        models.PointStruct(id=point_id("take", t["take_name"]), vector={},
                           payload={**base_payload(t, row, labels), "has_recon": t["take_name"] in recon_takes})
        for t, row in experts])

    client.create_collection(CLIPS, vectors_config={
        "body": models.VectorParams(size=10, distance=models.Distance.EUCLID),
        "object": models.VectorParams(size=3, distance=models.Distance.EUCLID),
    })
    points, no_view = [], 0
    for r in records:
        meta = r["meta"]
        pid = point_id("clip", meta["clip"])
        np.savez_compressed(osp.join(args.db, "frames", f"{pid}.npz"), feats=standardise(r["feats"], norm),
                            cams=np.array(r["cams"]),
                            views=r["views"] if r["views"] is not None else np.zeros((0, r["T"], 2), np.float32))
        no_view += r["views"] is None
        payload = {
            **base_payload(r["take_rec"], row_of_take[meta["take"]], labels),
            "clip": meta["clip"], "clip_dir": meta["clip_dir"], "pipe_cam": meta["pipe_cam"],
            "section_of": meta["section_of"], "n_frames": r["T"], "lo": meta["lo"],
            "hi": None if meta["lo"] is None else meta["lo"] + r["T"] - 1,
            "object_known": r["object"] is not None,
            "object_extents_m": None if r["object"] is None else [round(float(x), 4) for x in r["object"]],
            "betas": [round(float(x), 4) for x in r["betas"]],
            "has_views": r["views"] is not None, "view_note": r["view_note"],
        }
        if meta["gender"]:
            payload["gender"] = meta["gender"]
        vectors = {"body": r["betas"].tolist(),
                   # Qdrant local mode needs every named vector present; unknown sizes rank last at query time.
                   "object": (r["object"] if r["object"] is not None else np.zeros(3)).tolist()}
        points.append(models.PointStruct(id=pid, vector=vectors, payload=payload))
    client.upsert(CLIPS, points)
    client.close()

    per = collections.Counter(p.payload["scenario"] for p in points)
    exp = collections.Counter(row["scenario"] for _, row in experts)
    print(f"database: {args.db}")
    print(f"  experts: {len(experts)} tarred takes ({dict(exp)}), {len(recon_takes)} reconstructed")
    print(f"  clips:   {len(points)} reconstructed clips ({dict(per)})")
    if no_view:
        print(f"  {no_view} clip(s) have no camera views (missing calibration under {args.egoexo_root})")


# --------------------------------------------------------------------------- query


def open_db(db):
    """Open the on-disk database and its motion normalisation."""
    if not osp.isfile(osp.join(db, "motion_norm.npz")):
        raise SystemExit(f"no database at {db}; run `build` first")
    n = np.load(osp.join(db, "motion_norm.npz"))
    return QdrantClient(path=db), {"mean": n["mean"], "std": n["std"]}


def load_frames(db, pid):
    """A clip's standardised per-frame features and per-frame camera views from its frames file."""
    z = np.load(osp.join(db, "frames", f"{pid}.npz"))
    return {"feats": z["feats"], "cams": [str(c) for c in z["cams"]], "views": z["views"] if len(z["views"]) else None}


def build_filter(args, exclude_participant=None):
    """Qdrant filter for the drill / scenario / gender / participant exclusion requested."""
    must, must_not = [], []
    if args.task:
        must.append(models.FieldCondition(key="drill_key", match=models.MatchValue(value=drill_key(args.task))))
    if args.scenario:
        must.append(models.FieldCondition(key="scenario", match=models.MatchValue(value=args.scenario)))
    if getattr(args, "gender", None):
        must.append(models.FieldCondition(key="gender", match=models.MatchValue(value=args.gender)))
    if getattr(args, "rated_only", False):
        must.append(models.FieldCondition(key="participant_tier", match=models.MatchAny(any=list(RATED_TIERS))))
    if exclude_participant is not None:
        must_not.append(models.FieldCondition(key="participant", match=models.MatchValue(value=int(exclude_participant))))
    return models.Filter(must=must or None, must_not=must_not or None)


def scroll_all(client, flt):
    """Every clips point matching the filter, with payload."""
    out, offset = [], None
    while True:
        pts, offset = client.scroll(CLIPS, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
        out += pts
        if offset is None:
            return out


def qdrant_ranks(client, using, vector, flt, n):
    """{point id: rank} (0 = nearest) from a Qdrant nearest-neighbour search over the filtered clips."""
    res = client.query_points(CLIPS, query=list(map(float, vector)), using=using, query_filter=flt,
                              limit=max(n, 1), with_payload=False)
    return {p.id: i for i, p in enumerate(res.points)}


def longest_overlap(q, e):
    """Best same-speed alignment of two per-frame feature sequences over the longest shared length.

    The shorter sequence slides along the longer one; the offset with the
    highest mean per-frame cosine similarity wins. Returns sim, length and the
    start frame in each sequence.
    """
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    en = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    S = qn @ en.T
    Tq, Te = S.shape
    if Tq <= Te:
        sims = np.array([S.diagonal(k).mean() for k in range(Te - Tq + 1)])
        k = int(sims.argmax())
        return {"sim": float(sims[k]), "length": Tq, "q_start": 0, "e_start": k}
    sims = np.array([S.diagonal(-k).mean() for k in range(Tq - Te + 1)])
    k = int(sims.argmax())
    return {"sim": float(sims[k]), "length": Te, "q_start": k, "e_start": 0}


def rank_cameras(q_dirs, e_frames, e_start, length):
    """Expert cameras ranked by mean angle (deg) to the learner camera's direction over the aligned frames."""
    qd = unit(q_dirs)
    ranked = []
    for c, cam in enumerate(e_frames["cams"]):
        ed = unit(e_frames["views"][c, e_start:e_start + length])
        ranked.append((cam, float(np.degrees(np.arccos(np.clip((qd * ed).sum(1), -1, 1))).mean())))
    return sorted(ranked, key=lambda x: x[1])


def segment(payload_or_meta, start, length):
    """(first, last) take frames of a matched stretch, or None when the clip's take frames are unknown."""
    lo = payload_or_meta["lo"]
    return None if lo is None else (lo + start, lo + start + length - 1)


def retrieve(client, db, norm, q, args, flt, top):
    """Rank every candidate clip by fused motion/body/object rank and choose cameras for the top matches."""
    cands = scroll_all(client, flt)
    if not cands:
        return [], None
    q_std = standardise(q["feats"], norm)
    motion = {p.id: longest_overlap(q_std, load_frames(db, p.id)["feats"]) for p in cands}
    motion_rank = {pid: i for i, pid in enumerate(sorted(motion, key=lambda k: -motion[k]["sim"]))}
    body_rank = qdrant_ranks(client, "body", q["betas"], flt, len(cands))
    obj_rank = qdrant_ranks(client, "object", q["object"], flt, len(cands)) if q["object"] is not None else None

    def fused(p):
        """Weighted reciprocal-rank fusion score of one candidate."""
        n = len(cands)
        s = args.w_motion / (RRF_K + motion_rank[p.id]) + args.w_body / (RRF_K + body_rank.get(p.id, n))
        if obj_rank is not None:
            r = obj_rank.get(p.id, n) if p.payload.get("object_known", True) else n
            s += args.w_object / (RRF_K + r)
        return s

    q_cam = args.query_cam or q["meta"]["pipe_cam"]
    q_views_ok = q["views"] is not None and q_cam in q["cams"]
    hits, kept_segments = [], []
    for p in sorted(cands, key=fused, reverse=True):
        m = motion[p.id]
        seg = segment(p.payload, m["e_start"], m["length"])
        if seg and any(t == p.payload["take"] and seg[0] <= b and a <= seg[1] for t, (a, b) in kept_segments):
            continue
        if seg:
            kept_segments.append((p.payload["take"], seg))
        e_frames = load_frames(db, p.id)
        hit = {"score": fused(p), "payload": p.payload, "motion": m, "expert_frames": seg,
               "learner_frames": segment(q["meta"], m["q_start"], m["length"]),
               "body_dist": float(np.linalg.norm(np.array(p.payload["betas"]) - q["betas"])),
               "cameras": None, "camera_note": ""}
        if not q_views_ok:
            hit["camera_note"] = (f"learner has no camera views ({q['view_note']})" if q["views"] is None
                                  else f"learner camera {q_cam} not in its calibration")
        elif e_frames["views"] is None:
            hit["camera_note"] = f"expert has no camera views ({p.payload.get('view_note')})"
        else:
            qd = q["views"][q["cams"].index(q_cam), m["q_start"]:m["q_start"] + m["length"]]
            hit["cameras"] = rank_cameras(qd, e_frames, m["e_start"], m["length"])
        hit["expert_cam"] = hit["cameras"][0][0] if hit["cameras"] else (p.payload.get("best_exo") or p.payload["pipe_cam"])
        hits.append(hit)
        if len(hits) == top:
            break
    return hits, q_cam


# --------------------------------------------------------------------------- video


def video_source(egoexo_root, take, cam):
    """The take's frame-aligned video for a camera, preferring the 448p copy (same frame count as 4K)."""
    base = osp.join(egoexo_root, take, "frame_aligned_videos")
    for p in (osp.join(base, "downscaled", "448", f"{cam}.mp4"), osp.join(base, f"{cam}.mp4")):
        if osp.isfile(p):
            return p
    return None


_FFMPEG = {}


def ffmpeg_flags():
    """(frame-rate flag, encoder args) for this machine's ffmpeg, detected once.

    -fps_mode replaced -vsync in ffmpeg 5 and older builds reject it (as
    CARI4D's slurm_trim_clips.sh found on ikura); libx264 is preferred and
    mpeg4 is the fallback on builds without it.
    """
    if not _FFMPEG:
        ok = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc",
                             "-frames:v", "1", "-fps_mode", "vfr", "-f", "null", "-"], capture_output=True).returncode == 0
        _FFMPEG["fps"] = ["-fps_mode", "vfr"] if ok else ["-vsync", "vfr"]
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        _FFMPEG["enc"] = (["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"] if " libx264 " in enc
                          else ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"])
    return _FFMPEG["fps"], _FFMPEG["enc"]


def count_frames(path):
    """Number of video frames in a file, by counting packets."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def cut_clip(src, first, last, out):
    """Cut take frames first..last (inclusive) out of a video by frame index, at 448p. Returns a note."""
    fps, enc = ffmpeg_flags()
    n = last - first + 1
    vf = f"select='between(n\\,{first}\\,{last})',setpts=N/FRAME_RATE/TB,scale=-2:{VIDEO_HEIGHT}"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vf", vf, *fps,
                        "-frames:v", str(n), *enc, "-an", out], capture_output=True, text=True)
    if r.returncode != 0:
        return f"ffmpeg failed: {r.stderr.strip()[-300:]}"
    got = count_frames(out)
    return "" if got == n else f"WARNING: {out} has {got} frames, expected {n}"


def side_by_side(left, right, out):
    """Learner (left) and expert (right) clips stacked horizontally. Returns a note."""
    _, enc = ffmpeg_flags()
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", left, "-i", right,
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2", *enc, "-an", out],
                       capture_output=True, text=True)
    return "" if r.returncode == 0 else f"ffmpeg failed: {r.stderr.strip()[-300:]}"


def write_videos(hits, q, q_cam, args, out_dir):
    """Cut the learner's and each expert's matched stretch, plus a side-by-side per match."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("   (no ffmpeg/ffprobe on PATH; skipping videos)")
        return
    os.makedirs(out_dir, exist_ok=True)
    qm = q["meta"]
    q_src = video_source(args.egoexo_root, qm["take"], q_cam)
    for rank, h in enumerate(hits, 1):
        p = h["payload"]
        files = {}
        if h["learner_frames"] and q_src:
            a, b = h["learner_frames"]
            files["learner"] = osp.join(out_dir, f"rank{rank:02d}_learner_{qm['clip']}_{q_cam}_f{a}-{b}.mp4")
            h["learner_note"] = cut_clip(q_src, a, b, files["learner"])
        else:
            h["learner_note"] = "no learner video: " + ("unknown take frames" if not h["learner_frames"]
                                                        else f"{q_cam} video not found for {qm['take']}")
        e_src = video_source(args.egoexo_root, p["take"], h["expert_cam"])
        if h["expert_frames"] and e_src:
            a, b = h["expert_frames"]
            files["expert"] = osp.join(out_dir, f"rank{rank:02d}_expert_{p['clip']}_{h['expert_cam']}_f{a}-{b}.mp4")
            h["expert_note"] = cut_clip(e_src, a, b, files["expert"])
        else:
            h["expert_note"] = "no expert video: " + ("unknown take frames" if not h["expert_frames"]
                                                      else f"{h['expert_cam']} video not found for {p['take']}")
        if "learner" in files and "expert" in files and not h["learner_note"] and not h["expert_note"]:
            files["side_by_side"] = osp.join(out_dir, f"rank{rank:02d}_side_by_side.mp4")
            note = side_by_side(files["learner"], files["expert"], files["side_by_side"])
            if note:
                files.pop("side_by_side")
                h["expert_note"] = (h["expert_note"] + "; " if h["expert_note"] else "") + note
        h["videos"] = files


def print_hits(hits, q, q_cam):
    """Human-readable result list."""
    m = q["meta"]
    print(f"query: {m['clip']} ({m['take']}, {q['T']} frames, camera {q_cam}); "
          f"object {None if q['object'] is None else np.round(q['object'], 3).tolist()} m")
    for rank, h in enumerate(hits, 1):
        p, mo = h["payload"], h["motion"]
        print(f"{rank}. {p['clip']}  [{p['drill']}]  participant {p['participant']} ({p['participant_tier']}; "
              f"take label: {p['take_label']}), {p['gender']}, {p['university']}")
        print(f"   take {p['take']}  fused {h['score']:.4f}  motion sim {mo['sim']:.2f} over {mo['length']} frames "
              f"({mo['length'] / FPS:.1f} s)  body dist {h['body_dist']:.2f}  object {p['object_extents_m']} m")
        if h["expert_frames"]:
            a, b = h["expert_frames"]
            la = h["learner_frames"]
            print(f"   expert take frames {a}-{b} (~{a / FPS:.1f}-{(b + 1) / FPS:.1f} s)"
                  + (f"  <->  learner take frames {la[0]}-{la[1]}" if la else ""))
        if h["cameras"]:
            cam, ang = h["cameras"][0]
            rest = ", ".join(f"{c} {a:.0f}deg" for c, a in h["cameras"][1:])
            print(f"   camera: {cam} ({ang:.0f} deg from the learner's view){'; then ' + rest if rest else ''}")
        else:
            print(f"   camera: {h['expert_cam']} (best_exo fallback; {h['camera_note']})")
        for key in ("learner", "expert", "side_by_side"):
            if key in h.get("videos", {}):
                print(f"   {key}: {h['videos'][key]}")
        for key in ("learner_note", "expert_note"):
            if h.get(key):
                print(f"   {h[key]}")


def hits_json(hits, q, q_cam):
    """matches.json content: the query and every hit, as plain data."""
    m = q["meta"]
    return {
        "query": {"clip": m["clip"], "take": m["take"], "camera": q_cam, "n_frames": q["T"], "lo": m["lo"],
                  "object_extents_m": None if q["object"] is None else [float(x) for x in q["object"]]},
        "hits": [{
            "rank": i, "clip": h["payload"]["clip"], "take": h["payload"]["take"], "drill": h["payload"]["drill"],
            "participant": h["payload"]["participant"], "participant_tier": h["payload"]["participant_tier"],
            "take_label": h["payload"]["take_label"], "gender": h["payload"]["gender"],
            "fused_score": h["score"], "motion_sim": h["motion"]["sim"], "match_frames": h["motion"]["length"],
            "expert_take_frames": h["expert_frames"], "learner_take_frames": h["learner_frames"],
            "body_dist": h["body_dist"], "object_extents_m": h["payload"]["object_extents_m"],
            "expert_camera": h["expert_cam"], "camera_ranking": h["cameras"], "camera_note": h["camera_note"],
            "videos": h.get("videos", {}), "notes": [n for n in (h.get("learner_note"), h.get("expert_note")) if n],
        } for i, h in enumerate(hits, 1)],
    }


def query_records(args, takes):
    """Learner clips as clip_records, from --clip_dir folders or from --pth plus explicit metadata."""
    if args.clip_dir:
        return [clip_record(read_clip_meta(osp.expanduser(d)), takes, args.egoexo_root) for d in args.clip_dir]
    if not (args.pth and args.take):
        raise SystemExit("give --clip_dir, or --pth with --take (plus --pipe_cam and --lo for views and videos)")
    meta = {"clip": osp.splitext(osp.basename(args.pth))[0], "take": args.take, "pipe_cam": args.pipe_cam,
            "gender": None, "lo": args.lo, "section_of": None, "largest_m": args.object_size,
            "pth": osp.expanduser(args.pth), "obj": args.obj, "clip_dir": osp.dirname(osp.expanduser(args.pth))}
    return [clip_record(meta, takes, args.egoexo_root)]


def cmd_query(args):
    """Retrieve experts for each learner reconstruction, printing results and writing videos + matches.json."""
    takes = load_takes(args.takes_json)
    client, norm = open_db(args.db)
    for q in query_records(args, takes):
        exclude = args.exclude_participant
        if exclude == "self":
            exclude = takes.get(q["meta"]["take"], {}).get("participant_uid")
        hits, q_cam = retrieve(client, args.db, norm, q, args, build_filter(args, exclude_participant=exclude), args.top)
        if not hits:
            print(f"{q['meta']['clip']}: no matches; check --task spelling (`list` shows the drills)")
            continue
        out_dir = osp.join(args.out_dir, q["meta"]["clip"])
        if not args.no_videos:
            write_videos(hits, q, q_cam, args, out_dir)
        print_hits(hits, q, q_cam)
        os.makedirs(out_dir, exist_ok=True)
        with open(osp.join(out_dir, "matches.json"), "w") as f:
            json.dump(hits_json(hits, q, q_cam), f, indent=1)
        print(f"   matches.json: {osp.join(out_dir, 'matches.json')}\n")
    client.close()


def cmd_list(args):
    """Drill names and counts in the database, for spelling --task."""
    client, _ = open_db(args.db)
    for coll in (EXPERTS, CLIPS):
        counts, offset = collections.Counter(), None
        while True:
            pts, offset = client.scroll(coll, limit=500, offset=offset, with_payload=["scenario", "drill"])
            counts.update((p.payload["scenario"], p.payload["drill"]) for p in pts)
            if offset is None:
                break
        print(f"{coll}:")
        for (sc, dr), n in sorted(counts.items()):
            print(f"  {sc:<11} {dr:<26} {n}")
    client.close()


# --------------------------------------------------------------------------- calibrations and self-test


def cmd_missing_calibs(args):
    """Write an rclone filter for gopro_calibs.csv files the reconstructed clips need but lack."""
    missing = sorted({t for t in (read_clip_meta(d)["take"] for d in find_clip_dirs(args.recon_roots))
                      if not osp.isfile(calib_path(args.egoexo_root, t))})
    if not missing:
        print("every reconstructed clip has its calibration")
        return
    with open(args.out, "w") as f:
        f.write("# rclone filter: gopro_calibs.csv for reconstructed takes, relative to /vision/group/egoexo4d/takes\n")
        for t in missing:
            f.write(f"+ /{t}/trajectory/gopro_calibs.csv\n")
        f.write("- **\n")
    print(f"{len(missing)} take(s) missing calibration; filter written to {args.out}")
    print(f"rclone copy scdt:/vision/group/egoexo4d/takes {args.egoexo_root} --filter-from {args.out} -P")


def cmd_selftest(args):
    """Leave-one-person-out checks per drill: filter and exclusion, perfect self-alignment, own camera at 0 deg."""
    takes = load_takes(args.takes_json)
    client, norm = open_db(args.db)
    by_drill = collections.defaultdict(list)
    for p in scroll_all(client, None):
        by_drill[p.payload["drill"]].append(p)
    failures = 0
    for drill, group in sorted(by_drill.items()):
        probe = max(group, key=lambda p: (p.payload["has_views"], p.payload["n_frames"]))
        ns = argparse.Namespace(**{**vars(args), "task": drill, "scenario": None, "gender": None, "query_cam": None})
        q = clip_record(read_clip_meta(probe.payload["clip_dir"]), takes, args.egoexo_root)

        # 1. Person excluded: every hit is this drill and somebody else.
        hits, _ = retrieve(client, args.db, norm, q, ns, build_filter(ns, exclude_participant=probe.payload["participant"]), 5)
        ok1 = all(h["payload"]["drill"] == drill and h["payload"]["participant"] != probe.payload["participant"] for h in hits)
        # 2. Nothing excluded: the clip finds itself first, aligned frame for frame.
        own, _ = retrieve(client, args.db, norm, q, ns, build_filter(ns), 1)
        ok2 = (bool(own) and own[0]["payload"]["clip"] == probe.payload["clip"] and own[0]["motion"]["sim"] > 0.999
               and own[0]["motion"]["e_start"] == 0 and own[0]["motion"]["length"] == q["T"])
        # 3. ... and picks its own pipeline camera at ~0 degrees.
        cam_ok = None
        if ok2 and own[0]["cameras"]:
            cam_ok = own[0]["cameras"][0][0] == q["meta"]["pipe_cam"] and own[0]["cameras"][0][1] < 1.0
        failures += (not ok1) + (not ok2) + (cam_ok is False)
        print(f"{drill:<24} clips={len(group):<3} probe={probe.payload['clip']:<32} ({q['T']} frames)  "
              f"others-only={'ok' if ok1 else 'FAIL'}  self-aligned={'ok' if ok2 else 'FAIL'}  "
              f"own-camera={'ok' if cam_ok else ('FAIL' if cam_ok is False else 'n/a (no calib)')}")
        for h in hits[:3]:
            p = h["payload"]
            cam = f"{h['cameras'][0][0]} ({h['cameras'][0][1]:.0f} deg)" if h["cameras"] else f"{h['expert_cam']} (fallback)"
            print(f"    -> {p['clip']:<32} participant {p['participant']:<4} motion {h['motion']['sim']:.2f} "
                  f"over {h['motion']['length']} frames, camera {cam}")
    client.close()
    print("selftest:", "all checks passed" if not failures else f"{failures} failure(s)")
    if failures:
        sys.exit(1)


# --------------------------------------------------------------------------- CLI


def add_common(p):
    """Paths shared by every subcommand."""
    p.add_argument("--db", default=DEFAULT_DB, help="on-disk Qdrant folder")
    p.add_argument("--takes_json", default=DEFAULT_TAKES)
    p.add_argument("--egoexo_root", default=DEFAULT_EGOEXO,
                   help="folder holding <take>/trajectory/gopro_calibs.csv and <take>/frame_aligned_videos/")


def add_search(p):
    """Search and fusion options shared by query and selftest."""
    p.add_argument("--task", default=None, help="drill to restrict to, e.g. 'Mikan Layup', 'Penalty Kick', 'CPR'")
    p.add_argument("--scenario", default=None, choices=sorted(SCENARIOS))
    p.add_argument("--gender", default=None, help="only experts recorded with this gender")
    p.add_argument("--rated_only", action="store_true",
                   help="only experts whose participant is rated Early / Intermediate / Late Expert "
                        "(EgoExo4D proficiency_demonstrator); 'unlabelled' is NOT an expert")
    p.add_argument("--w_motion", type=float, default=2.0)
    p.add_argument("--w_body", type=float, default=1.0)
    p.add_argument("--w_object", type=float, default=1.0)


def main():
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="(re)build the database")
    add_common(b)
    b.add_argument("--recon_roots", nargs="+", default=DEFAULT_ROOTS)
    b.add_argument("--manifest", default=DEFAULT_MANIFEST)
    b.add_argument("--annot_dir", default=DEFAULT_ANNOT, help="EgoExo4D annotations, for per-take labels (optional)")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="retrieve experts for learner reconstructions")
    add_common(q)
    add_search(q)
    q.add_argument("--clip_dir", nargs="+", default=None, help="exported clip folder(s): meta.json + .pth + .obj")
    q.add_argument("--pth", default=None, help="or: a bundle .pth ...")
    q.add_argument("--take", default=None, help="... with its EgoExo take name")
    q.add_argument("--pipe_cam", default=None, help="... the camera the bundle is in")
    q.add_argument("--lo", type=int, default=None, help="... the take frame of the bundle's first frame")
    q.add_argument("--obj", default=None, help="... the metric object mesh")
    q.add_argument("--object_size", type=float, default=None, help="... or the object's largest extent in metres")
    q.add_argument("--query_cam", default=None, help="learner camera to match views against (default: its pipe_cam)")
    q.add_argument("--exclude_participant", default=None, help="participant id to leave out, or 'self'")
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--out_dir", default=DEFAULT_OUT, help="videos and matches.json go in <out_dir>/<learner clip>/")
    q.add_argument("--no_videos", action="store_true", help="print and write matches.json only")
    q.set_defaults(func=cmd_query)

    ls = sub.add_parser("list", help="drill names and counts")
    add_common(ls)
    ls.set_defaults(func=cmd_list)

    mc = sub.add_parser("missing-calibs", help="rclone filter for calibrations missing locally")
    add_common(mc)
    mc.add_argument("--recon_roots", nargs="+", default=DEFAULT_ROOTS)
    mc.add_argument("--out", default="missing_calibs_filter.txt")
    mc.set_defaults(func=cmd_missing_calibs)

    st = sub.add_parser("selftest", help="leave-one-person-out checks on the expert clips")
    add_common(st)
    add_search(st)
    st.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Feedback for a novice from two InterMimic rollouts, written by Qwen from the raw geometric differences.

PULLED INTO InterMimic 2026-10-16 unchanged from jes-bro/Qwen3-VL-Embedding
recon-retrieval @42b31e6 (Jess: retrieval, inference and feedback in one repo).
scripts/novice_rollout.py runs it with --dry_run as the JOINT DIFFERENCING stage:
the three expert-vs-novice tables (joint angles, body-to-object relations,
speeds over the lined-up movement) are written to JSON without loading a
language model; the Qwen feedback is the optional last step.

    # the data tables and the prompt, no model loaded
    python motion_feedback.py --novice novice.npz --expert expert.npz \
        --task "Mikan layup" --obj ball.obj --dry_run

    # Qwen reads them and writes the feedback
    python motion_feedback.py --novice novice.npz --expert expert.npz \
        --task "Mikan layup" --obj ball.obj --out feedback.json

    # finer time resolution (better suited to a larger model): a row every 10 frames
    python motion_feedback.py ... --time keyframes --every 10 --model Qwen/Qwen2.5-14B-Instruct

INPUTS. The .npz InterMimic's player writes when DUMP_TRAJ is set
(isaacgym/src/intermimic/learning/intermimic_players.py): per frame, body_rot
and body_pos for 52 bodies, obj_pos, obj_rot and the subject. A 591-channel
InterMimic motion .pt is read too (body_pos 162:318, obj_pos 318:321, obj_rot
321:325, body_rot 383:591). Checked on data: quaternions are xyzw and the world
is z-up. Hand-to-object distances with the mesh match the dataset's contact
labels (about 1 cm in contact, 30-50 cm otherwise).

WHAT THE MODEL IS GIVEN, and nothing more:
  the task, which rollout is the expert and which the novice, and three tables
  with expert and novice values side by side for every row --
    joint angles   one number per joint for 20 joints (trunk, neck, head, and
                   each side's hip, knee, ankle, collarbone, shoulder, elbow,
                   wrist): the angle between the body segment entering the joint
                   and the one leaving it, with what 0 means for that joint
    object         for 15 body parts, the distance to the object (its surface
                   when --obj gives the mesh, else its centre) and where the
                   object's centre sits from that part in the person's own
                   facing frame: forward, left, up
    motion         speed of each of those body parts, and of the object
Nothing is thresholded, ranked or pre-selected, and no cue is written for it.
The prompt asks it to first work out what the novice does differently from the
expert that most needs to change, then give the novice feedback that explains
those differences in the context of the task, grounded in the numbers given.

TIME. The two rollouts are lined up first: the shorter slides along the longer
at the same speed and the offset where the joint angles agree best wins
(--align overlap; --align resample stretches both to one length instead). Then
  --time phases     (default) each value is the average over --phases equal
                    parts of the aligned movement (3: start, middle, end)
  --time keyframes  each value is the average over consecutive --every-frame
                    blocks, so a 150-frame match at --every 10 gives 15 rows per
                    joint -- much longer input, meant for a larger model

Nothing in a DUMP_TRAJ file names the object, so pass its mesh with --obj
(or --novice_obj / --expert_obj when they differ).

The model defaults to Qwen/Qwen2.5-3B-Instruct (in the local Hugging Face
cache, fits a 12 GB GPU); --model takes any chat model transformers can load.
Needs numpy and torch; transformers for the feedback.
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

FPS = 30.0

# InterMimic's 52-body SMPL-X humanoid, in rigid-body order
# (isaacgym/src/intermimic/data/assets/smplx/*.xml).
BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",
]
IDX = {name: i for i, name in enumerate(BODY_NAMES)}
UP = "UP"


def joint_definitions():
    """(name, incoming segment, outgoing segment, what the angle means) for the 20 non-finger joints.

    A segment is a (from body, to body) pair, or UP for the world vertical.
    The angle is between the two segment directions, so 0 means the body
    continues straight through the joint.
    """
    joints = [
        ("pelvis", UP, ("Pelvis", "Torso"), "tilt of the lower trunk from vertical; 0 = upright"),
        ("lower back", ("Pelvis", "Torso"), ("Torso", "Spine"), "bend in the lower spine; 0 = straight"),
        ("mid back", ("Torso", "Spine"), ("Spine", "Chest"), "bend in the middle spine; 0 = straight"),
        ("upper back", ("Spine", "Chest"), ("Chest", "Neck"), "bend in the upper spine; 0 = straight"),
        ("neck", ("Chest", "Neck"), ("Neck", "Head"), "bend of the neck; 0 = head in line with the upper spine"),
        ("head", UP, ("Neck", "Head"), "tilt of the head from vertical; 0 = upright"),
    ]
    for side, s in (("left", "L"), ("right", "R")):
        joints += [
            (f"{side} hip", ("Torso", "Pelvis"), (f"{s}_Hip", f"{s}_Knee"),
             "angle between the downward trunk line and the thigh; 0 = thigh straight down in line with the trunk"),
            (f"{side} knee", (f"{s}_Hip", f"{s}_Knee"), (f"{s}_Knee", f"{s}_Ankle"), "0 = straight leg, larger = more bent"),
            (f"{side} ankle", (f"{s}_Knee", f"{s}_Ankle"), (f"{s}_Ankle", f"{s}_Toe"),
             "angle between shin and foot; about 90 = neutral foot"),
            (f"{side} collarbone", ("Chest", f"{s}_Thorax"), (f"{s}_Thorax", f"{s}_Shoulder"), "bend at the collarbone"),
            (f"{side} shoulder", ("Chest", "Pelvis"), (f"{s}_Shoulder", f"{s}_Elbow"),
             "angle between the downward trunk line and the upper arm; 0 = arm hanging down, 90 = level, 180 = overhead"),
            (f"{side} elbow", (f"{s}_Shoulder", f"{s}_Elbow"), (f"{s}_Elbow", f"{s}_Wrist"), "0 = straight arm, larger = more bent"),
            (f"{side} wrist", (f"{s}_Elbow", f"{s}_Wrist"), (f"{s}_Wrist", f"{s}_Middle1"),
             "bend at the wrist; 0 = hand in line with the forearm"),
        ]
    return joints


JOINTS = joint_definitions()

# Body parts measured against the object and for speed; the hand is its middle knuckle, the foot its toe.
BODY_PARTS = [("pelvis", "Pelvis"), ("chest", "Chest"), ("head", "Head")]
for _side, _s in (("left", "L"), ("right", "R")):
    BODY_PARTS += [(f"{_side} shoulder", f"{_s}_Shoulder"), (f"{_side} elbow", f"{_s}_Elbow"),
                   (f"{_side} hand", f"{_s}_Middle1"), (f"{_side} hip", f"{_s}_Hip"),
                   (f"{_side} knee", f"{_s}_Knee"), (f"{_side} foot", f"{_s}_Toe")]


# --------------------------------------------------------------------------- loading


def load_motion(path):
    """A rollout as dict(body_pos (T,52,3), body_rot (T,52,4 xyzw), obj_pos (T,3), obj_rot (T,4 xyzw), subject)."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        missing = {"body_rot", "body_pos", "obj_pos", "obj_rot"} - set(z.files)
        if missing:
            raise SystemExit(f"{path} is missing {sorted(missing)}; expected an InterMimic DUMP_TRAJ .npz")
        out = {k: np.asarray(z[k], dtype=np.float64) for k in ("body_pos", "body_rot", "obj_pos", "obj_rot")}
        out["subject"] = str(z["subject"]) if "subject" in z.files else "unknown"
    elif path.endswith((".pt", ".pth")):
        import torch
        x = torch.load(path, map_location="cpu", weights_only=False)
        d = x.get("hoi_data") if isinstance(x, dict) else x
        d = torch.as_tensor(d).detach().double().numpy()
        if d.ndim != 2 or d.shape[1] != 591:
            raise SystemExit(f"{path}: expected a (T, 591) InterMimic motion, got {d.shape}")
        T = len(d)
        out = {"body_pos": d[:, 162:318].reshape(T, 52, 3), "obj_pos": d[:, 318:321], "obj_rot": d[:, 321:325],
               "body_rot": d[:, 383:591].reshape(T, 52, 4), "subject": osp.splitext(osp.basename(path))[0]}
    else:
        raise SystemExit(f"{path}: expected a DUMP_TRAJ .npz or a 591-channel motion .pt")
    if out["body_pos"].shape[1:] != (52, 3) or len(out["body_pos"]) < 2:
        raise SystemExit(f"{path}: expected at least 2 frames of 52 bodies, got {out['body_pos'].shape}")
    return out


def load_mesh_points(path, max_points=3000, seed=0):
    """Object vertices in the object's own frame from an .obj (or anything trimesh reads), subsampled."""
    if path.endswith(".obj"):
        v = []
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    v.append(line.split()[1:4])
        pts = np.asarray(v, dtype=np.float64)
    else:
        import trimesh
        pts = np.asarray(trimesh.load(path, force="mesh").vertices, dtype=np.float64)
    if len(pts) == 0:
        raise SystemExit(f"{path}: no vertices")
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


# --------------------------------------------------------------------------- measurements


def quat_to_matrix(q):
    """xyzw quaternions (..., 4) to rotation matrices (..., 3, 3)."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def angle_between(u, v):
    """Angle in degrees between two arrays of vectors (..., 3)."""
    cos = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def segment(pos, seg):
    """Direction vectors (T, 3) of a (from, to) body segment, or the world vertical for UP."""
    if seg == UP:
        return np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(pos), 3))
    return pos[:, IDX[seg[1]]] - pos[:, IDX[seg[0]]]


def joint_angles(m):
    """{joint name: (T,) angle in degrees} for every joint in JOINTS."""
    p = m["body_pos"]
    return {name: angle_between(segment(p, a), segment(p, b)) for name, a, b, _ in JOINTS}


def facing_frame(pos):
    """Per-frame forward, left and up unit vectors of the person, from the hips (independent of scene heading)."""
    left = pos[:, IDX["L_Hip"]] - pos[:, IDX["R_Hip"]]
    left[:, 2] = 0.0
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-9)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), left.shape)
    return np.cross(left, up), left, up


def object_relations(m, mesh_points):
    """{body part: dict(distance, forward, left, up)} in cm, each (T,): distance to the object and the object's centre offset."""
    p = m["body_pos"]
    if mesh_points is not None:
        world = np.einsum("tij,nj->tni", quat_to_matrix(m["obj_rot"]), mesh_points) + m["obj_pos"][:, None]
    else:
        world = m["obj_pos"][:, None]
    fwd, left, up = facing_frame(p)
    out = {}
    for label, body in BODY_PARTS:
        part = p[:, IDX[body]]
        rel = m["obj_pos"] - part
        out[label] = {
            "distance": 100 * np.linalg.norm(world - part[:, None], axis=2).min(1),
            "forward": 100 * (rel * fwd).sum(1),
            "left": 100 * (rel * left).sum(1),
            "up": 100 * (rel * up).sum(1),
        }
    return out


def speeds(m):
    """{body part or 'object': (T,) speed in m/s}."""
    out = {}
    for label, body in BODY_PARTS:
        out[label] = np.linalg.norm(np.gradient(m["body_pos"][:, IDX[body]], axis=0), axis=1) * FPS
    out["object"] = np.linalg.norm(np.gradient(m["obj_pos"], axis=0), axis=1) * FPS
    return out


# --------------------------------------------------------------------------- time


def align(novice_angles, expert_angles, mode):
    """Paired frame indices (novice, expert) and a sentence describing the pairing.

    overlap:  the shorter rollout slides along the longer at the same speed; the
              offset with the smallest mean standardised joint-angle difference wins.
    resample: both are stretched to the longer length.
    """
    fn = np.stack([novice_angles[n] for n, *_ in JOINTS], axis=1)
    fe = np.stack([expert_angles[n] for n, *_ in JOINTS], axis=1)
    Tn, Te = len(fn), len(fe)
    if mode == "resample":
        n = max(Tn, Te)
        return (np.round(np.linspace(0, Tn - 1, n)).astype(int), np.round(np.linspace(0, Te - 1, n)).astype(int),
                f"both rollouts stretched to {n} frames (novice {Tn}, expert {Te})")
    allf = np.concatenate([fn, fe])
    mu, sd = allf.mean(0), np.maximum(allf.std(0), 1.0)
    zn, ze = (fn - mu) / sd, (fe - mu) / sd
    L = min(Tn, Te)
    costs = [np.abs((zn[k:k + L] - ze) if Tn > Te else (zn - ze[k:k + L])).mean() for k in range(abs(Tn - Te) + 1)]
    k = int(np.argmin(costs))
    if Tn > Te:
        ni, ei = np.arange(k, k + L), np.arange(L)
    else:
        ni, ei = np.arange(L), np.arange(k, k + L)
    return ni, ei, (f"novice frames {ni[0]}-{ni[-1]} matched to expert frames {ei[0]}-{ei[-1]} at the same speed "
                    f"({L} frames, {L / FPS:.1f} s)")


def time_slots(n, mode, phases, every):
    """(label, indices into the aligned timeline) for each row of time in the tables."""
    if mode == "keyframes":
        slots = []
        for start in range(0, n, every):
            idx = np.arange(start, min(start + every, n))
            slots.append((f"{start / FPS:.1f}-{(idx[-1] + 1) / FPS:.1f}s", idx))
        return slots
    names = {1: ["whole"], 2: ["first half", "second half"], 3: ["start", "middle", "end"]}.get(
        phases, [f"part {i + 1} of {phases}" for i in range(phases)])
    return [(name, idx) for name, idx in zip(names, np.array_split(np.arange(n), phases)) if len(idx)]


def slot_mean(series, frames, idx):
    """Mean of a per-frame series over the aligned frames in one time slot."""
    return float(np.mean(series[frames[idx]]))


# --------------------------------------------------------------------------- tables and prompt


def build_tables(nov, exp, ni, ei, slots):
    """The three CSV tables (joint angles, object, motion) as text, plus the same numbers as data."""
    rows_j, rows_o, rows_m = [], [], []
    for label, idx in slots:
        for name, *_ in JOINTS:
            e, n = slot_mean(exp["angles"][name], ei, idx), slot_mean(nov["angles"][name], ni, idx)
            rows_j.append([label, name, round(e), round(n), round(n - e)])
        for part, _ in BODY_PARTS:
            eo, no = exp["object"][part], nov["object"][part]
            rows_o.append([label, part] + [round(slot_mean(eo[k], ei, idx)) for k in ("distance", "forward", "left", "up")]
                          + [round(slot_mean(no[k], ni, idx)) for k in ("distance", "forward", "left", "up")])
        for part in [p for p, _ in BODY_PARTS] + ["object"]:
            e, n = slot_mean(exp["speed"][part], ei, idx), slot_mean(nov["speed"][part], ni, idx)
            rows_m.append([label, part, round(e, 2), round(n, 2), round(n - e, 2)])
    head_j = ["time", "joint", "expert_deg", "novice_deg", "novice_minus_expert_deg"]
    head_o = ["time", "body_part", "expert_distance_cm", "expert_object_forward_cm", "expert_object_left_cm",
              "expert_object_up_cm", "novice_distance_cm", "novice_object_forward_cm", "novice_object_left_cm",
              "novice_object_up_cm"]
    head_m = ["time", "body_part", "expert_speed_m_s", "novice_speed_m_s", "novice_minus_expert_m_s"]
    def as_csv(head, rows):
        """A header and rows as CSV text."""
        return "\n".join([",".join(head)] + [",".join(str(v) for v in r) for r in rows])

    text = {"joint_angles": as_csv(head_j, rows_j), "object": as_csv(head_o, rows_o), "motion": as_csv(head_m, rows_m)}
    data = {"joint_angles": [dict(zip(head_j, r)) for r in rows_j], "object": [dict(zip(head_o, r)) for r in rows_o],
            "motion": [dict(zip(head_m, r)) for r in rows_m]}
    return text, data


def build_messages(task, tables, align_note, time_note, used_mesh, novice_desc, expert_desc):
    """Chat messages: the task, who is who, the three tables, and the two things to do with them."""
    joint_meanings = "\n".join(f"- {name}: {meaning}" for name, _, _, meaning in JOINTS)
    distance_to = "surface" if used_mesh else "centre (no mesh was given, so treat distances as approximate)"
    system = f"You are an expert coach for this task: {task}."
    user = f"""Task: {task}

Two people did this task and their 3D motion was reconstructed.
- EXPERT: {expert_desc}
- NOVICE: {novice_desc}

The two recordings were lined up in time ({align_note}). {time_note}

JOINT ANGLES. One angle per joint, in degrees: the angle between the body segment entering the joint and the segment leaving it, so 0 means the body continues straight through the joint.
{joint_meanings}

{tables['joint_angles']}

OBJECT. For each body part: its distance to the object's {distance_to} in cm, and where the object's centre is from that body part in the person's own facing direction (forward = in front of them, left = to their left, up = above; negative values mean behind, to their right, below).

{tables['object']}

MOTION. How fast each body part, and the object, is moving, in metres per second.

{tables['motion']}

First, work out what the novice is doing differently from the expert that they need to change: which joint angles, positions relative to the object, or motions differ in ways that matter for this task, and which difference is the most important.

Then give the novice feedback that explains these differences in the context of the task, grounded in the numbers above. Do not invent measurements beyond the data given."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate(messages, model_name, max_new_tokens):
    """Run a chat model through transformers (the same pattern as sm/llm_absa.py) and return its reply."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_name)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    print(f"prompt length: {inputs.input_ids.shape[1]} tokens", flush=True)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    return tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------------------- CLI


def describe_rollout(path, m):
    """Short description of a rollout for the prompt: file, subject, length."""
    return f"{osp.basename(path)} (subject {m['subject']}, {len(m['body_pos'])} frames, {len(m['body_pos']) / FPS:.1f} s)"


def main():
    """Measure both rollouts, line them up, tabulate, and ask the model for feedback."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--novice", required=True, help="DUMP_TRAJ .npz (or 591-channel .pt) of the person getting feedback")
    ap.add_argument("--expert", required=True, help="DUMP_TRAJ .npz (or .pt) of the expert")
    ap.add_argument("--task", required=True, help="what they are doing, e.g. 'Mikan layup' or 'CPR chest compressions'")
    ap.add_argument("--obj", default=None, help="object mesh (.obj) shared by both, for surface distances")
    ap.add_argument("--novice_obj", default=None, help="novice's object mesh, if it differs from --obj")
    ap.add_argument("--expert_obj", default=None, help="expert's object mesh, if it differs from --obj")
    ap.add_argument("--align", choices=["overlap", "resample"], default="overlap")
    ap.add_argument("--time", choices=["phases", "keyframes"], default="phases",
                    help="phases: average over --phases parts; keyframes: average over every --every frames")
    ap.add_argument("--phases", type=int, default=3)
    ap.add_argument("--every", type=int, default=10, help="frames per row with --time keyframes")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=1000)
    ap.add_argument("--dry_run", action="store_true", help="print the prompt; load no model")
    ap.add_argument("--out", default=None, help="write the tables, prompt and feedback as JSON")
    args = ap.parse_args()

    nov_m, exp_m = load_motion(args.novice), load_motion(args.expert)
    nov_obj, exp_obj = args.novice_obj or args.obj, args.expert_obj or args.obj
    nov_pts = load_mesh_points(nov_obj) if nov_obj else None
    exp_pts = load_mesh_points(exp_obj) if exp_obj else None
    nov = {"angles": joint_angles(nov_m), "object": object_relations(nov_m, nov_pts), "speed": speeds(nov_m)}
    exp = {"angles": joint_angles(exp_m), "object": object_relations(exp_m, exp_pts), "speed": speeds(exp_m)}

    ni, ei, align_note = align(nov["angles"], exp["angles"], args.align)
    slots = time_slots(len(ni), args.time, args.phases, args.every)
    time_note = (f"Every value is the average over one of {len(slots)} equal parts of the lined-up movement "
                 f"({', '.join(s[0] for s in slots)})." if args.time == "phases" else
                 f"Every value is the average over a {args.every}-frame block of the lined-up movement; "
                 f"the time column is seconds from the start of the lined-up stretch.")
    tables, data = build_tables(nov, exp, ni, ei, slots)
    messages = build_messages(args.task, tables, align_note, time_note, nov_pts is not None and exp_pts is not None,
                              describe_rollout(args.novice, nov_m), describe_rollout(args.expert, exp_m))

    print(f"novice  {describe_rollout(args.novice, nov_m)}")
    print(f"expert  {describe_rollout(args.expert, exp_m)}")
    print(f"aligned {align_note}; {len(slots)} time rows")
    feedback = None
    if args.dry_run:
        print("\n----- prompt -----")
        for m in messages:
            print(f"[{m['role']}]\n{m['content']}\n")
    else:
        print(f"asking {args.model} ...", flush=True)
        feedback = generate(messages, args.model, args.max_new_tokens)
        print("\n----- feedback -----")
        print(feedback)

    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"novice": args.novice, "expert": args.expert, "task": args.task,
                       "alignment": {"note": align_note, "novice_frames": [int(ni[0]), int(ni[-1])],
                                     "expert_frames": [int(ei[0]), int(ei[-1])]},
                       "time": {"mode": args.time, "rows": [s[0] for s in slots]}, "tables": data,
                       "messages": messages, "model": None if args.dry_run else args.model,
                       "feedback": feedback}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

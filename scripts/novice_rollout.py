#!/usr/bin/env python3
"""Expert motion on the novice's body: retarget, roll out the distilled policy,
render, overlay on the novice's own video. The point is the RETARGET: an expert's
clip (training set) performed by a body the policy never saw (any CARI4D novice).

    retarget/<novice>/<expert clip>.pt   the expert clip contact-retargeted onto the
                                         novice body (scripts/retarget_contact.py, the
                                         solver every training reference went through)
    rollout.npz      the best of N simultaneous attempts on the novice body
                     (DUMP_PICK=best: an env that reached the clip's end, else the
                     farthest survivor; never a hard failure)
    skinned.mp4      the novice's SMPL-H surface performing it (render_mesh_replay.py)
    <seq>_sim.pth    the rollout in CARI4D's prediction format in the novice video's
                     camera, ANCHORED at the novice's first frame (sim_to_cari4d_bundle.py
                     --anchor-rollout: the novice's own clip + bundle give the camera)
    overlay          CARI4D tools/viz_pred.py --wild_video --flat_color: human AND object
                     as one flat figure over the footage (printed; --run-overlay runs it)

Inputs: the EXPERT clip (sub<E>_<object>_<idx>.pt, a training clip) and body MJCF;
the NOVICE's installed clip (sub<N>_..., same recon the bundle came from), bundle,
body MJCF and gender; the object mesh CARI4D posed; the policy's arm + checkpoint;
the SMPL model dir. Everything is derived and VALIDATED before any GPU time: the
expert clip's object has physical props in the arm's cfg (or a same-family entry
to copy), the novice has a height (arm's file, the task's table, or estimated
from its MJCF), both MJCFs exist, the novice clip and bundle have one frame
count. A missing piece is an error naming what to add -- never a default.

The policy runs in the ARM'S OWN EVAL CFG (scripts/check_eval_cfg.py) with keys
redirected into the output dir and written as a derived yaml there: motion_file
-> a one-clip dir (every env plays the expert clip from frame 0, stateInit
Start), retargetedMotionDir -> the tree holding the retargeted reference under
the novice body, subjectHeightsFile -> the arm's file + the novice, and
objectPropsFile -> the arm's file + the clip's object if it was a family copy.
The novice MJCF loads through REPLAY_ASSET_FILE (humanoid.py expects
smplx_omomo_<sub>.xml; a reconstructed person is smplh_behave_<sub>.xml).
Students run through intermimic.run_distill with DISTILL_SKIP_TEACHERS=1 (at
test the teachers are never used, so only the student .pth is needed).

    python3 scripts/novice_rollout.py --dry-run \\
        --expert-clip InterAct/behave_cari4d_act/sub401_bballd03s01rev003b_000.pt \\
        --novice-clip InterAct/behave_cari4d_rectinj_hy3d3_optj3d/sub204_bball_000.pt \\
        --novice-bundle ~/novice_Sub204/Date170_Sub204_bball_t004a/Date170_Sub204_bball_t004a.pth \\
        --object-mesh ~/novice_Sub204/Date170_Sub204_bball_t004a/Date170_Sub204_bball_t004a_067_align.obj \\
        --gender male --arm student_g3_act_xf_ret_nvadlr__f0 \\
        --checkpoint checkpoints/smplx_student_g3_act_xf_ret_nvadlr__f0/nn/mimic_00100000.pth \\
        --models /simurgh2/projects/ret-hoi/InterAct/models/smplx --out renders/novice_sub204_from_sub401
    # then the same line without --dry-run on a GPU node (scripts/slurm_novice_rollout.sh)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
ASSET_ROOT = REPO / "isaacgym/src/intermimic/data/assets"


class PlanError(SystemExit):
    pass


# ---------------------------------------------------------------- resolution (pure, tested)
def parse_clip_name(clip):
    """'sub204_bball_000.pt' -> ('sub204', 'bball', '000'). The task reads the object
    as the second-to-last '_' token and the subject as the first."""
    stem = Path(clip).name
    if not stem.endswith(".pt"):
        raise PlanError(f"clip must be a .pt motion tensor: {clip}")
    parts = stem[:-3].split("_")
    if len(parts) < 3 or not re.fullmatch(r"sub\d+", parts[0]):
        raise PlanError(f"clip name must be sub<N>_<object>_<idx>.pt, got {stem}")
    return parts[0], parts[-2], parts[-1]


def arm_cfgs(arm):
    """(eval env cfg, train cfg, entry module, task class) for an arm, the way
    eval_one.sh resolves them: the arm's OWN eval cfg via check_eval_cfg.py, the
    train cfg by the student/teacher naming rule, entry/task from the eval cfg."""
    r = subprocess.run([sys.executable, str(REPO / "scripts/check_eval_cfg.py"), "--arm", arm],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise PlanError(f"no eval cfg for arm {arm}:\n{r.stdout}{r.stderr}")
    env_yaml = Path(r.stdout.strip().splitlines()[-1])
    stem = arm[len("smplx_teacher_"):] if arm.startswith("smplx_teacher_") else arm
    train = CFG / "train/rlg" / (f"omomo_{stem}.yaml" if stem.startswith("student_") else f"omomo_teacher_{stem}.yaml")
    if not train.is_file():
        raise PlanError(f"train cfg not found for {arm}: {train}")
    doc = yaml.safe_load(open(env_yaml)) or {}
    return env_yaml, train, doc.get("evalEntry", "intermimic.run"), doc.get("evalTask", "InterMimic")


def mjcf_for(subject, explicit=None):
    """The subject's MJCF under the asset root: explicit path, else the CARI4D
    smplh_behave_<sub>.xml, else the OMOMO smplx_omomo_<sub>.xml. Returns the path
    RELATIVE to the asset root (what REPLAY_ASSET_FILE wants)."""
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            raise PlanError(f"MJCF not found: {p}")
        try:
            return str(p.relative_to(ASSET_ROOT.resolve()))
        except ValueError:
            raise PlanError(f"MJCF must live under the asset root {ASSET_ROOT} (Isaac Gym loads assets by "
                            f"path relative to it); copy it to {ASSET_ROOT}/smplx/ first: {p}")
    for cand in (f"smplx/smplh_behave_{subject}.xml", f"smplx/smplx_omomo_{subject}.xml"):
        if (ASSET_ROOT / cand).is_file():
            return cand
    raise PlanError(f"no MJCF for {subject}: expected {ASSET_ROOT}/smplx/smplh_behave_{subject}.xml "
                    f"(CARI4D subject) or smplx_omomo_{subject}.xml; generate it (scripts/cari4d_render.py "
                    f"installs one per bundle) or pass --expert-mjcf / --novice-mjcf")


def object_props_entry(props_path, obj):
    """The object's {mass, restitution} from the arm's objectPropsFile, or, when the
    exact name is absent, the entry of the same FAMILY (bballXXX for a new bball
    clip): CARI4D names each clip's mesh, so a new take of the same ball is new
    to the file. Returns (entry, source_name). Refuses when no family matches
    or the family's entries disagree."""
    if not Path(props_path).is_file():
        raise PlanError(f"objectPropsFile not found: {props_path} (the activity props file is generated "
                        f"by scripts/merge_activity_data.py on the machine with the activity data)")
    doc = yaml.safe_load(open(props_path)) or {}
    objs = doc.get("objects") or {}
    if obj in objs:
        return dict(objs[obj]), obj
    def family(name):
        """'bballd03s01rev003b' -> 'bball', 'soccerd71s96t005c' -> 'soccer', 'manikin' -> 'manikin':
        the leading letters up to CARI4D's date marker d<digits> (or the first digit)."""
        m = re.match(r"^([a-z]+?)(?=d\d|\d|$)", name)
        return m.group(1) if m else name
    fam = family(obj)
    same = sorted(n for n in objs if family(n) == fam)
    if not same:
        raise PlanError(f"object {obj!r} has no entry in {props_path} and no entry of the family {fam!r} to copy "
                        f"from (entries: {sorted(objs)[:8]}...). Add it: objects: {{{obj}: {{mass: <kg>, restitution: <0-1>}}}}")
    vals = {tuple(sorted(objs[n].items())) for n in same}
    if len(vals) != 1:
        raise PlanError(f"family {fam!r} entries disagree in {props_path} ({same[:4]}...); add {obj} explicitly")
    return dict(objs[same[0]]), same[0]


def height_for(subject, heights_path, mjcf_abs):
    """The subject's height in metres: the arm's heights file, else the task's
    built-in table (read from its source; importing the task needs Isaac Gym),
    else estimated from the MJCF the way scripts/compute_heights.py does for
    every other body. Returns (height, source)."""
    num = int(subject[3:])
    if heights_path and Path(heights_path).is_file():
        h = json.load(open(heights_path))
        if str(num) in h:
            return float(h[str(num)]), str(heights_path)
    src = (REPO / "isaacgym/src/intermimic/env/tasks/intermimic.py").read_text()
    m = re.search(r"SUBJECT_HEIGHTS\s*=\s*\{(.*?)\}", src, re.S)
    table = {int(k): float(v) for k, v in re.findall(r"(\d+)\s*:\s*([\d.]+)", m.group(1))} if m else {}
    if num in table:
        return table[num], "intermimic.py SUBJECT_HEIGHTS"
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch", REPO / "scripts/compute_heights.py")
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    return float(ch.estimate_height(Path(mjcf_abs))), f"estimated from {Path(mjcf_abs).name} (compute_heights.py)"


def betas_from_bundle(bundle, subject, gender, out_npz, width=16):
    """A one-subject betas npz (SMPLXPoser's layout: subject key + '_genders') from
    the bundle's averaged 10-dim SMPL-H shape, zero-padded to the 16 the model
    multiplies by. Returns (10-dim vector, bundle frame count)."""
    import numpy as np
    import importlib.util
    spec = importlib.util.spec_from_file_location("c2i", REPO / "scripts/cari4d_to_interact.py")
    c2i = importlib.util.module_from_spec(spec); spec.loader.exec_module(c2i)
    d = c2i._load_bundle(Path(bundle))
    if "pr" not in d or "betas" not in d["pr"]:
        raise PlanError(f"bundle has no pr.betas: {bundle}")
    betas = d["pr"]["betas"].detach().float().numpy().reshape(-1, 10).mean(0)
    padded = np.zeros(width, dtype=np.float32)
    padded[:10] = betas
    np.savez(out_npz, _genders=np.array([f"{subject}:{gender}"]), **{subject: padded})
    n_frames = int(d["pr"]["smpl_t"].shape[0]) if "smpl_t" in d["pr"] else None
    return betas, n_frames


def derived_env_cfg(base_yaml, redirects, out_yaml, note):
    """The arm's eval cfg with the given env keys redirected, written to the output
    dir. Loaded and dumped as yaml (never sed: block-style lists). A key absent
    from the base (a non-retargeting arm has no retargetedMotionDir) is skipped,
    not invented. Returns {key: (old, new)}."""
    doc = yaml.safe_load(open(base_yaml))
    env = doc["env"]
    changed = {}
    for k, v in redirects.items():
        if k not in env:
            continue
        changed[k] = (env[k], str(v))
        env[k] = str(v)
    header = (f"# DERIVED by scripts/novice_rollout.py from {Path(base_yaml).name} -- {note}\n"
              + "".join(f"#   {k}: {a} -> {b}\n" for k, (a, b) in changed.items())
              + "# Every other key is the eval cfg's. Not an as-run training cfg; safe to delete.\n")
    with open(out_yaml, "w") as fh:
        fh.write(header)
        yaml.safe_dump(doc, fh, sort_keys=False)
    return changed


# ---------------------------------------------------------------- the plan
def _file(path, what):
    # absolute(), not resolve(): a clip may be a symlink named for the subject it
    # is INSTALLED as (sub204_...pt -> some recon file); the name is the identity.
    p = Path(os.path.expanduser(str(path))).absolute()
    if not p.is_file():
        raise PlanError(f"{what} not found: {p}")
    return p


def build_plan(a):
    expert_clip = _file(a.expert_clip, "expert clip")
    novice_clip = _file(a.novice_clip, "novice clip")
    bundle = _file(a.novice_bundle, "novice bundle")
    mesh = _file(a.object_mesh, "object mesh")
    ckpt = _file(a.checkpoint, "checkpoint")
    expert, obj, idx = parse_clip_name(expert_clip)
    novice, novice_obj, _ = parse_clip_name(novice_clip)
    if expert == novice:
        raise PlanError(f"expert and novice are the same subject ({expert}); the point is a body the policy never saw")
    models = Path(os.path.expanduser(a.models)).resolve()
    if not any((models / f).exists() for f in ("SMPLH_MALE.pkl", "SMPLH_MALE.npz", "SMPLX_MALE.npz", "SMPLX_MALE.pkl")):
        raise PlanError(f"--models {models}: no SMPLH_*/SMPLX_* model file there (licensed; download them yourself)")

    env_yaml, train_yaml, entry, task = arm_cfgs(a.arm)
    env = yaml.safe_load(open(env_yaml))["env"]
    if expert not in env.get("dataSub", []):
        raise PlanError(f"expert {expert} is not in the arm's dataSub {env.get('dataSub')}: the expert must be a "
                        f"training source of {a.arm}")
    if novice in env.get("subjectBodies", []):
        raise PlanError(f"novice {novice} is in the arm's subjectBodies: not an unseen body")
    expert_mjcf = mjcf_for(expert, a.expert_mjcf)
    novice_mjcf = mjcf_for(novice, a.novice_mjcf)
    props_path = env.get("objectPropsFile")
    props = object_props_entry(REPO / props_path, obj) if props_path else None
    if not props_path and env.get("objectMass") is None and env.get("objectDensity") is None:
        raise PlanError(f"{env_yaml.name} sets none of objectPropsFile / objectMass / objectDensity")
    heights_file = env.get("subjectHeightsFile")
    height, height_src = height_for(novice, REPO / heights_file if heights_file else None, ASSET_ROOT / novice_mjcf)

    return {
        "expert": expert, "novice": novice, "object": obj, "novice_object": novice_obj, "idx": idx,
        "expert_clip": expert_clip, "novice_clip": novice_clip, "bundle": bundle, "seq": bundle.stem,
        "mesh": mesh, "ckpt": ckpt, "models": models, "arm": a.arm, "env_yaml": env_yaml,
        "train_yaml": train_yaml, "entry": entry, "task": task,
        "expert_mjcf": expert_mjcf, "novice_mjcf": novice_mjcf,
        "props": props, "props_path": props_path, "height": height, "height_src": height_src,
        "heights_file": heights_file, "out": Path(a.out).resolve(), "num_envs": a.num_envs,
        "frames": a.frames, "gender": a.gender, "video": a.video, "cari4d_root": a.cari4d_root,
        "flat_color": a.flat_color, "retarget_iters": a.retarget_iters,
        "is_student": entry.endswith("run_distill"),
    }


def write_derived(p):
    """Everything in the output dir that is not a GPU/render product: the one-clip
    motion dir, the heights file, the props file (if extended), the derived env cfg.
    The retargeted reference is written by the retarget STAGE, into retarget/<novice>/."""
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
    clip_dir = out / "clip"
    clip_dir.mkdir(exist_ok=True)
    link = clip_dir / p["expert_clip"].name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(p["expert_clip"])
    (out / "retarget" / p["novice"]).mkdir(parents=True, exist_ok=True)
    heights = {}
    if p["heights_file"] and (REPO / p["heights_file"]).is_file():
        heights = json.load(open(REPO / p["heights_file"]))
    heights[str(int(p["novice"][3:]))] = p["height"]
    hj = out / "heights.json"
    json.dump(heights, open(hj, "w"), indent=1)
    redirects = {"motion_file": clip_dir, "retargetedMotionDir": out / "retarget", "subjectHeightsFile": hj}
    if p["props"] is not None:
        entry, src = p["props"]
        doc = yaml.safe_load(open(REPO / p["props_path"]))
        doc.setdefault("objects", {})[p["object"]] = entry
        props_out = out / "object_props.yaml"
        with open(props_out, "w") as fh:
            fh.write(f"# DERIVED by scripts/novice_rollout.py: {Path(p['props_path']).name} + {p['object']} "
                     f"(copied from {src}: {entry})\n")
            yaml.safe_dump(doc, fh, sort_keys=False)
        redirects["objectPropsFile"] = props_out
    derived = out / "env_cfg.yaml"
    note = f"{p['novice']} performing {p['expert_clip'].name}, arm {p['arm']}"
    changed = derived_env_cfg(p["env_yaml"], redirects, derived, note)
    return derived, changed


def commands(p, derived):
    """The stage commands, as (name, env, argv) -- printed by --dry-run, run otherwise."""
    out = p["out"]
    rt_file = out / "retarget" / p["novice"] / p["expert_clip"].name
    dump = out / "rollout.npz"
    betas = out / f"betas_{p['novice']}.npz"
    skinned = out / "skinned.mp4"
    sim_pth = out / "hy3d" / f"{p['seq']}_sim.pth"          # 'hy3d' in the path: viz_pred picks the recon mesh
    py = sys.executable
    pp = {"PYTHONPATH": f"{REPO}/isaacgym/src:{REPO}"}
    retarget = [py, str(REPO / "scripts/retarget_contact.py"), "--clip", str(p["expert_clip"]),
                "--source", p["expert"], "--source-mjcf", str(ASSET_ROOT / p["expert_mjcf"]),
                "--target", p["novice"], "--target-mjcf", str(ASSET_ROOT / p["novice_mjcf"]),
                "--iters", str(p["retarget_iters"]), "--out-dir", str(rt_file.parent)]
    env_roll = {**pp, "DUMP_TRAJ": str(dump), "DUMP_PICK": "best", "DUMP_FRAMES": str(p["frames"]),
                "REPLAY_ASSET_FILE": p["novice_mjcf"]}
    if p["is_student"]:
        env_roll["DISTILL_SKIP_TEACHERS"] = "1"
    roll = [py, "-u", "-m", p["entry"], "--task", p["task"], "--cfg_env", str(derived),
            "--cfg_train", str(p["train_yaml"]), "--test", "--headless", "--checkpoint", str(p["ckpt"]),
            "--subject_bodies", p["novice"], "--data_sub", p["expert"], "--data_objects", p["object"],
            "--num_envs", str(p["num_envs"])]
    skin = [py, str(REPO / "scripts/render_mesh_replay.py"), "--dump", str(dump), "--betas", str(betas),
            "--models", str(p["models"]), "--model-type", "smplh", "--object", str(p["mesh"]), "--out", str(skinned)]
    bundle = [py, str(REPO / "scripts/sim_to_cari4d_bundle.py"), "--dump", str(dump), "--bundle", str(p["bundle"]),
              "--pt", str(p["novice_clip"]), "--anchor-rollout", "--betas", str(betas), "--subject", p["novice"],
              "--models", str(p["models"]), "--model-type", "smplh", "--object-mesh", str(p["mesh"]), "--out", str(sim_pth)]
    overlay = None
    if p["cari4d_root"]:
        overlay = ["python", "tools/viz_pred.py", "-pf", str(sim_pth), "--wild_video", "--kid", "0", "--no_side",
                   "--flat_color", p["flat_color"], "--hy3d_meshes_root", str(p["mesh"].parent.parent),
                   "--out_root", str(out / "overlay")] + (["--video", str(Path(p["video"]).resolve())] if p["video"] else [])
    return ([("retarget", pp, retarget), ("rollout", env_roll, roll), ("skinned", pp, skin), ("bundle", pp, bundle),
             ("overlay", {"cwd": p["cari4d_root"]}, overlay)], rt_file, dump, betas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-clip", required=True, help="the EXPERT's motion tensor sub<E>_<object>_<idx>.pt (a training clip)")
    ap.add_argument("--novice-clip", required=True, help="the NOVICE's own installed clip sub<N>_... (from the same recon as --novice-bundle)")
    ap.add_argument("--novice-bundle", required=True, help="the CARI4D .pth the novice clip was built from (camera + betas)")
    ap.add_argument("--object-mesh", required=True, help="the *_align.obj CARI4D posed for the novice sequence")
    ap.add_argument("--arm", required=True, help="the policy's arm, e.g. student_g3_act_xf_ret_nvadlr__f0 (its eval cfg is the env)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gender", required=True, choices=["male", "female", "neutral"], help="the novice's SMPL-H body model")
    ap.add_argument("--models", required=True, help="dir holding SMPLH_*.pkl / SMPLX_*.npz (licensed)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--expert-mjcf", default=None, help="the expert's MJCF under the asset root (default: smplh_behave_<E>.xml, else smplx_omomo_<E>.xml)")
    ap.add_argument("--novice-mjcf", default=None, help="the novice's MJCF under the asset root (default: smplh_behave_<N>.xml)")
    ap.add_argument("--num-envs", type=int, default=64, help="simultaneous attempts; the best one is kept")
    ap.add_argument("--frames", type=int, default=1000, help="DUMP_FRAMES cap per attempt")
    ap.add_argument("--retarget-iters", type=int, default=300)
    ap.add_argument("--video", default=None, help="the novice's source video for the overlay (viz_pred.py --video)")
    ap.add_argument("--cari4d-root", default=None, help="CARI4D repo; with it the overlay command is printed/run")
    ap.add_argument("--flat-color", default="255,140,0", help="one r,g,b for human AND object in the overlay")
    ap.add_argument("--run-overlay", action="store_true", help="also run viz_pred.py (needs the CARI4D env)")
    ap.add_argument("--dry-run", action="store_true", help="validate, write the derived files, print the commands, run nothing")
    ap.add_argument("--skip-retarget", action="store_true", help="reuse an existing retargeted reference in --out")
    ap.add_argument("--skip-rollout", action="store_true", help="reuse an existing rollout.npz in --out")
    a = ap.parse_args(argv)

    p = build_plan(a)
    derived, changed = write_derived(p)
    stages, rt_file, dump, betas = commands(p, derived)

    print(f"== novice_rollout: {p['novice']} performs {p['expert_clip'].name} (expert {p['expert']}, object {p['object']})")
    print(f"   novice recon : {p['novice_clip'].name} + bundle {p['seq']} (camera, betas, {p['gender']})")
    print(f"   arm          : {p['arm']}  ->  {p['entry']} / {p['task']}"
          + ("  [DISTILL_SKIP_TEACHERS=1: student .pth only]" if p["is_student"] else ""))
    print(f"   checkpoint   : {p['ckpt']}")
    print(f"   bodies       : expert {p['expert_mjcf']}  ->  novice {p['novice_mjcf']} (REPLAY_ASSET_FILE)")
    print(f"   novice height: {p['height']:.3f} m  ({p['height_src']})")
    if p["props"] is not None:
        print(f"   object props : {p['object']} -> {p['props'][0]}  (from {p['props'][1]} in {Path(p['props_path']).name})")
    print(f"   env cfg      : {derived}  (from {p['env_yaml'].name}; redirected: {', '.join(changed)})")
    print(f"   attempts     : {p['num_envs']} envs, cap {p['frames']} frames, keep the best (DUMP_PICK=best)")
    print(f"   out          : {p['out']}")
    for name, env, argv_ in stages:
        if argv_ is None:
            continue
        pre = " ".join(f"{k}={v}" for k, v in env.items() if k not in ("cwd", "PYTHONPATH"))
        print(f"   [{name}] {pre} {' '.join(argv_)}")
    if a.dry_run:
        print("   (DRY RUN: nothing executed)")
        return 0

    def run(stage, cwd=None):
        name, env, argv_ = stage
        r = subprocess.run(argv_, env={**os.environ, **{k: v for k, v in env.items() if k != "cwd"}},
                           cwd=str(cwd or REPO))
        if r.returncode != 0:
            raise PlanError(f"[{name}] failed (exit {r.returncode})")

    if not a.skip_retarget:
        run(stages[0])
    if not rt_file.is_file():
        raise PlanError(f"[retarget] wrote no reference at {rt_file} (the solver refuses a solve that got worse: "
                        f"raise --retarget-iters, see its FAILURES output)")
    if not a.skip_rollout:
        run(stages[1])
    if not dump.is_file():
        raise PlanError(f"[rollout] no {dump}")
    import numpy as np
    d = np.load(dump, allow_pickle=True)
    print(f"== rollout: env {int(d['chosen_env'])} kept, {d['body_pos'].shape[0]} frames, "
          f"succeeded={bool(d['chosen_succeeded'])} ({int(d['n_succeeded'])}/{int(d['num_envs'])} envs reached the end)")
    vec, n_bundle = betas_from_bundle(p["bundle"], p["novice"], p["gender"], betas)
    print(f"== betas: {betas} (10-dim, mean over the bundle's frames; bundle has {n_bundle} frames)")
    run(stages[2])
    run(stages[3])
    name, env, argv_ = stages[4]
    if argv_ is not None:
        print("== overlay (run inside the CARI4D env, from the CARI4D repo; needs the flat-colour patch to viz_pred.py):")
        print("   cd " + str(env["cwd"]) + " && " + " ".join(argv_))
        if a.run_overlay:
            run(stages[4], cwd=env["cwd"])
    print(f"== done: {p['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

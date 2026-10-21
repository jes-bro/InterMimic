#!/usr/bin/env python3
"""FINE-TUNE arms: a trained gen4 teacher continued under a changed reward / episode rule,
as a NEW arm (own name, own checkpoint dir) that starts from the trained checkpoint.

Jess, 2026-10-20: the six trained teachers (50-frame msexp) track body, object and
contacts well; only the feet slide (soft root) or tap (strict product). Instead of
seven days from scratch, fine-tune them: a converged policy answers a reward change in
hours. This is a WARM START from another run's checkpoint -- Jess's explicit decision
for the Nov 10 deadline; the result is reported as "trained, then fine-tuned with X".

Bases (checkpoints measured 2026-10-20): the trained 50-frame msexp teachers --
    chairs 44000  boxes 35000  stands 36000  tables 46000  bball7 49000  soccer15 49000
Variants, all FULL EPISODES (rolloutLength 1000 + minEpisodeFrames 30):
    flight            A  objectFlightReward {0.6, flat 0} replaces motionScaleReward
    flightfootw       B  A + rewardTerms.footWeight {5}         (the main bet: one knob)
    flightfootwswing  C  B + rewardTerms.footSwing {50}         (the explicit swing lift)
    footw             D  motionScaleReward kept + footWeight {5} (the foot fix alone)
Arm name: <base without _msexp__f0>_ft<epoch/1000>k_<variant>_fullep__f0, e.g.
    gen4_omomo_geoall_chairs_xf_nvadlr_nopose_ft44k_flightfootw_fullep__f0

Each arm is DERIVED by text from the base's four files (env, train, launcher, eval cfg):
  env     the block edits above (utils/object_flight.py, utils/foot_terms.py), key-sorted
  train   full_experiment_name -> smplx_teacher_<arm>; resume_from -> the BASE checkpoint
  launch  names; the reward guards swapped/added; the resume block replaced: the arm's
          OWN checkpoints/<arm>/nn/mimic.pth when it exists (a resubmission continues the
          fine-tune), else the base checkpoint, which must exist or the job refuses
  eval    the base's eval cfg with the same block edits (+ minEpisodeFrames mirrored)
The generator REFUSES an env cfg that differs from its base by anything but the
variant's keys (yaml-level diff). tests/test_finetune_cfgs.py pins it.

  python3 scripts/make_finetune_cfgs.py                       # all 6 bases x 4 variants = 24 arms
  python3 scripts/make_finetune_cfgs.py --bases chairs --variants flightfootw --diff
"""
import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
_s = importlib.util.spec_from_file_location("mkf", REPO / "scripts/make_gen4_flight_teacher_cfgs.py")
mkf = importlib.util.module_from_spec(_s); _s.loader.exec_module(mkf)

TRAINED = {   # short name -> (base arm, checkpoint epoch)
    "chairs":   ("gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0", 44000),
    "boxes":    ("gen4_omomo_geoall_boxes_xf_nvadlr_nopose_msexp__f0", 35000),
    "stands":   ("gen4_omomo_geoall_stands_xf_nvadlr_nopose_msexp__f0", 36000),
    "tables":   ("gen4_omomo_geoall_tables_xf_nvadlr_nopose_msexp__f0", 46000),
    "bball7":   ("gen4_bball7_geoall_xf_nvadlr_nopose_msexp__f0", 49000),
    "soccer15": ("gen4_soccer15_geoall_xf_nvadlr_nopose_msexp__f0", 49000),
}
VARIANTS = {  # tag -> (flight rule on, footWeight on, footSwing on)
    "flight":           (True, False, False),
    "flightfootw":      (True, True, False),
    "flightfootwswing": (True, True, True),
    "footw":            (False, True, False),
}
FOOT_WEIGHT = 5.0            # default; --weight W renames the footw variants footw<W> (footw2, flightfootw2, ...)
SWING_LAMBDA = 50.0


def set_weight(w):
    """Pick the foot weight: variant tags carry it unless it is 5 (the first set)."""
    global FOOT_WEIGHT, VARIANTS
    FOOT_WEIGHT = float(w)
    if FOOT_WEIGHT != 5.0:
        sfx = str(int(w)) if float(w).is_integer() else str(w).replace(".", "p")
        VARIANTS = {(k.replace("footw", f"footw{sfx}") if "footw" in k else k): v for k, v in VARIANTS.items() if "footw" in k}
FULLEP_ROLLOUT, FULLEP_MIN_FRAMES = 1000, 30
FLAT = 0.0

RESUME_BLOCK = re.compile(r"^# --- auto-resume:.*?\nelse\n    echo \"\[teacher\] fresh start \(no checkpoint at \$\{CKPT\}\)\"\nfi\n", re.M | re.S)


def arm_name(base, epoch, tag):
    assert base.endswith("_msexp__f0"), base
    return base[:-len("_msexp__f0")] + f"_ft{epoch // 1000}k_{tag}_fullep__f0"


def ckpt_path(base, epoch):
    return f"checkpoints/smplx_teacher_{base}/nn/mimic_{epoch:08d}.pth"


def note(base, epoch, tag, what):
    fl, fw, fs = VARIANTS[tag]
    edits = ["full episodes (rolloutLength 1000 + minEpisodeFrames 30)"]
    edits.append(f"objectFlightReward {{rootFlight {mkf.ROOT_FLIGHT}, flatFlight {FLAT}}} in place of motionScaleReward" if fl
                 else "motionScaleReward kept")
    if fw: edits.append(f"rewardTerms.footWeight {{weight {FOOT_WEIGHT}}}")
    if fs: edits.append(f"rewardTerms.footSwing {{lambda {SWING_LAMBDA}}}")
    return (f"# FINE-TUNE arm ({tag}): DERIVED from the {base} {what} by scripts/make_finetune_cfgs.py.\n"
            f"# Starts from that teacher's checkpoint at epoch {epoch} ({ckpt_path(base, epoch)}) and\n"
            f"# continues under: " + "; ".join(edits) + ".\n"
            f"# Nothing else differs (pinned by tests/test_finetune_cfgs.py). A WARM START from another run's\n"
            f"# checkpoint, Jess 2026-10-20, for the Nov 10 deadline: reported as 'trained, then fine-tuned'.\n")


def edit_env_blocks(text, base, tag, is_eval=False):
    """The variant's block edits on an env/eval cfg body (shared by both)."""
    fl, fw, fs = VARIANTS[tag]
    if not is_eval:
        text, n = re.subn(r"^(\s*)rolloutLength:\s*50\s*$", lambda m: f"{m.group(1)}rolloutLength: {FULLEP_ROLLOUT}", text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"ERROR: {base}: rolloutLength: 50 not found exactly once")
    if re.search(r"^\s*minEpisodeFrames:", text, re.M):
        raise SystemExit(f"ERROR: {base}: already has minEpisodeFrames")
    m = re.search(r"^(\s*)moreRigid:", text, re.M)
    if not m:
        raise SystemExit(f"ERROR: {base}: moreRigid not found (anchor for minEpisodeFrames)")
    text = text[:m.start()] + f"{m.group(1)}minEpisodeFrames: {FULLEP_MIN_FRAMES}\n" + text[m.start():]
    if fl:
        m = mkf.MS_BLOCK.search(text)
        if not m or mkf.MS_BLOCK.search(text, m.end()):
            raise SystemExit(f"ERROR: {base}: motionScaleReward block not found exactly once")
        text = text[:m.start()] + text[m.end():]
        a = re.search(r"^(?P<ind>[ \t]*)(objectMass|objectPropsFile|obsHorizons):", text, re.M)
        text = text[:a.start()] + mkf.flight_block(a.group("ind"), FLAT) + text[a.start():]
    if fw or fs:
        r = re.search(r"^(?P<ind>[ \t]*)rewardTerms:\n(?P<ind2>[ \t]+)(freeFlightGate|pose):", text, re.M)
        if not r:
            raise SystemExit(f"ERROR: {base}: rewardTerms block not found")
        ind = r.group("ind2")
        blk = ""
        if fs:   # 'footSwing' < 'footWeight' < 'freeFlightGate'
            blk += f"{ind}footSwing:\n{ind}  enable: true\n{ind}  lambda: {SWING_LAMBDA}\n"
        if fw:
            blk += f"{ind}footWeight:\n{ind}  enable: true\n{ind}  weight: {FOOT_WEIGHT}\n"
        text = text[:r.start("ind2")] + blk + text[r.start("ind2"):]
    return text


def derive_env(text, base, epoch, tag):
    arm = arm_name(base, epoch, tag)
    text = edit_env_blocks(text, base, tag)
    text = text.replace(base, arm)
    if VARIANTS[tag][0]:
        text = mkf.reword(text, FLAT)
    return note(base, epoch, tag, "env cfg") + text


def derive_train(text, base, epoch, tag):
    arm = arm_name(base, epoch, tag)
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{re.escape(base)}\s*$",
                     lambda m: m.group(1) + f"smplx_teacher_{arm}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: {base}: full_experiment_name not found exactly once in the train cfg")
    new = new.replace(base, arm)          # rename BEFORE the checkpoint path: it carries the base's name
    new, n = re.subn(r"^(\s*resume_from:)\s*'?None'?\s*$", lambda m: f"{m.group(1)} '{ckpt_path(base, epoch)}'", new, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: {base}: resume_from: None not found exactly once in the train cfg")
    return note(base, epoch, tag, "train cfg") + new


def guards(tag):
    fl, fw, fs = VARIANTS[tag]
    g = (f"# Full-episode guard: this arm fine-tunes with starts anywhere and episodes to the clip's end.\n"
         f"if ! grep -qE '^\\s*rolloutLength:\\s*{FULLEP_ROLLOUT}\\s*$' \"$CFG_ENV\" || ! grep -qE '^\\s*minEpisodeFrames:\\s*{FULLEP_MIN_FRAMES}\\s*$' \"$CFG_ENV\"; then\n"
         f"    echo \"[teacher] ERROR: full-episode arm without rolloutLength {FULLEP_ROLLOUT} + minEpisodeFrames {FULLEP_MIN_FRAMES} in $CFG_ENV\" >&2; exit 1\n"
         f"fi\n")
    if fw:
        g += (f"# Foot-weight guard: the feet count x{FOOT_WEIGHT} in the position / rotation terms (utils/foot_terms.py).\n"
              f"if ! grep -qE '^\\s*footWeight:' \"$CFG_ENV\" || ! grep -qE '^\\s*weight:\\s*{FOOT_WEIGHT}\\s*$' \"$CFG_ENV\"; then\n"
              f"    echo \"[teacher] ERROR: footWeight arm without rewardTerms.footWeight weight {FOOT_WEIGHT} in $CFG_ENV\" >&2; exit 1\n"
              f"fi\n")
    if fs:
        g += (f"# Foot-swing guard: a toe below the reference toe on reference swing frames costs (utils/foot_terms.py).\n"
              f"if ! grep -qE '^\\s*footSwing:' \"$CFG_ENV\" || ! grep -qE '^\\s*lambda:\\s*{SWING_LAMBDA}\\s*$' \"$CFG_ENV\"; then\n"
              f"    echo \"[teacher] ERROR: footSwing arm without rewardTerms.footSwing lambda {SWING_LAMBDA} in $CFG_ENV\" >&2; exit 1\n"
              f"fi\n")
    return g


def resume_block(base, epoch):
    ck = ckpt_path(base, epoch)
    return (f"# --- fine-tune resume: the arm's OWN latest checkpoint when it has one (a resubmission\n"
            f"# continues the fine-tune), else the BASE teacher's checkpoint written into the train cfg\n"
            f"# (resume_from). The base must exist: a fine-tune that cannot load it must not start fresh.\n"
            f"EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' \"$CFG_TRAIN\" | awk '{{print $2}}')\n"
            f"CKPT=\"checkpoints/${{EXP}}/nn/mimic.pth\"\n"
            f"BASE_CKPT=\"{ck}\"\n"
            f"[ -f \"$BASE_CKPT\" ] || {{ echo \"[teacher] ERROR: base checkpoint not found: $BASE_CKPT\" >&2; exit 1; }}\n"
            f"if [ -f \"$CKPT\" ]; then\n"
            f"    RESUME_TRAIN=\"/tmp/${{EXP}}_resume_${{SLURM_JOB_ID:-$$}}.yaml\"\n"
            f"    sed -E \"s|^(\\s*resume_from:).*$|\\1 '${{CKPT}}'|\" \"$CFG_TRAIN\" > \"$RESUME_TRAIN\"\n"
            f"    if ! grep -qF \"resume_from: '${{CKPT}}'\" \"$RESUME_TRAIN\"; then\n"
            f"        echo \"[teacher] ERROR: could not rewrite resume_from in $CFG_TRAIN -- refusing to\" \\\n"
            f"             \"restart the fine-tune from the base over ${{CKPT}}\" >&2; exit 1\n"
            f"    fi\n"
            f"    CFG_TRAIN=\"$RESUME_TRAIN\"\n"
            f"    echo \"[teacher] RESUMING the fine-tune from its own ${{CKPT}}\"\n"
            f"else\n"
            f"    echo \"[teacher] FINE-TUNE from the base checkpoint $BASE_CKPT (epoch {epoch}); resume_from in $CFG_TRAIN\"\n"
            f"fi\n")


def derive_launcher(text, base, epoch, tag):
    arm = arm_name(base, epoch, tag)
    fl, fw, fs = VARIANTS[tag]
    m = mkf.MS_GUARD.search(text)
    if not m or mkf.MS_GUARD.search(text, m.end()):
        raise SystemExit(f"ERROR: {base}: motion-scale guard not found exactly once in the launcher")
    text = text[:m.end()] + guards(tag) + text[m.end():]
    if fl:   # the flight variants: the motion-scale guard becomes the object-flight guard
        m2 = mkf.MS_GUARD.search(text)
        text = text[:m2.start()] + mkf.flight_guard(FLAT) + text[m2.end():]
    text = text.replace(base, arm)        # rename BEFORE the resume block: BASE_CKPT carries the base's name
    rb = RESUME_BLOCK.search(text)
    if not rb or RESUME_BLOCK.search(text, rb.end()):
        raise SystemExit(f"ERROR: {base}: auto-resume block not found exactly once in the launcher")
    text = text[:rb.start()] + resume_block(base, epoch) + text[rb.end():]
    if fl:
        text = mkf.reword(text, FLAT)
    extra = " + FOOT WEIGHT x5" * fw + " + FOOT SWING" * fs
    text = text.replace("rollout 50,", f"FULL EPISODES (fine-tune from epoch {epoch}){extra},")
    lines = text.splitlines(keepends=True)
    last = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    return "".join(lines[:last + 1]) + note(base, epoch, tag, "launcher") + "".join(lines[last + 1:])


def derive_eval(text, base, epoch, tag):
    arm = arm_name(base, epoch, tag)
    body = edit_env_blocks(text, base, tag, is_eval=True)
    body = body.replace(base, arm)
    if VARIANTS[tag][0]:
        body = mkf.reword(body, FLAT)
    return (f"# EVAL config -- {arm}: DERIVED from omomo_eval_{base}.yaml by scripts/make_finetune_cfgs.py\n"
            f"# with the fine-tune arm's block edits (minEpisodeFrames mirrored: at eval Start init begins every\n"
            f"# episode at frame 0 and rolloutLength 1000 runs it to the end). check_eval_cfg.py proves it mirrors the arm.\n") + body


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def expected_diff(tag):
    fl, fw, fs = VARIANTS[tag]
    d = {"env.rolloutLength", "env.minEpisodeFrames"}
    if fl:
        d |= {"env.motionScaleReward.enable", "env.motionScaleReward.referenceStd",
              "env.objectFlightReward.enable", "env.objectFlightReward.rootFlight", "env.objectFlightReward.flatFlight"}
    if fw:
        d |= {"env.rewardTerms.footWeight.enable", "env.rewardTerms.footWeight.weight"}
    if fs:
        d |= {"env.rewardTerms.footSwing.enable", "env.rewardTerms.footSwing.lambda"}
    return d


def verify(base, epoch, tag, env_base, env_out):
    a, b = flatten(yaml.safe_load(open(env_base))), flatten(yaml.safe_load(open(env_out)))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    if diff != expected_diff(tag):
        raise SystemExit(f"ERROR: {arm_name(base, epoch, tag)}: env cfg differs from its base by {sorted(diff)}, "
                         f"expected {sorted(expected_diff(tag))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", nargs="*", default=sorted(TRAINED), choices=sorted(TRAINED))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--diff", action="store_true", help="cfg_diff each env cfg vs its base")
    ap.add_argument("--weight", type=float, default=5.0, help="footWeight.weight for the footw variants; W != 5 writes only the footw<W> variants")
    a = ap.parse_args()
    set_weight(a.weight)
    if a.weight != 5.0:
        a.variants = [v for v in VARIANTS]
    n = 0
    for short in a.bases:
        base, epoch = TRAINED[short]
        srcs = {"env": CFG / f"omomo_teacher_{base}.yaml", "train": CFG / "train/rlg" / f"omomo_teacher_{base}.yaml",
                "sh": REPO / f"slurm_teacher_{base}.sh", "eval": CFG / f"omomo_eval_{base}.yaml"}
        for p in srcs.values():
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p}")
        for tag in a.variants:
            arm = arm_name(base, epoch, tag)
            outs = {"env": CFG / f"omomo_teacher_{arm}.yaml", "train": CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml",
                    "sh": REPO / f"slurm_teacher_{arm}.sh", "eval": CFG / f"omomo_eval_{arm}.yaml"}
            outs["env"].write_text(derive_env(mkf_strip(srcs["env"].read_text()), base, epoch, tag))
            outs["train"].write_text(derive_train(mkf_strip(srcs["train"].read_text()), base, epoch, tag))
            outs["sh"].write_text(derive_launcher(srcs["sh"].read_text(), base, epoch, tag))
            outs["sh"].chmod(0o755)
            outs["eval"].write_text(derive_eval(mkf_strip(srcs["eval"].read_text()), base, epoch, tag))
            verify(base, epoch, tag, srcs["env"], outs["env"])
            n += 4
            print(arm)
            if a.diff:
                subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(srcs["env"]), str(outs["env"])], check=False)
    print(f"wrote {n} files for {n // 4} fine-tune arms")


def mkf_strip(text):
    """Drop a yaml file's leading comment header (the base's own note); the body keeps its keys."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(lines[i:])


if __name__ == "__main__":
    main()

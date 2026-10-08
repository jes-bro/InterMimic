#!/usr/bin/env python3
"""gen4 EgoExo4D teachers: basketball, soccer, and a basketball+soccer generalist,
on the recipe the gen4 OMOMO object-group arms train with (Jess, 2026-10-07).

Three arms, each a COPY of its g3 transformer twin with exactly these edits and
nothing else (tests/test_gen4_egoexo_teacher_cfgs.py pins it):

  gen4_bball7_geoall_xf_nvadlr_nopose_msexp__f0    <- g3_bball7_geoall_xf_nvadlr_nopose__f0
  gen4_soccer15_geoall_xf_nvadlr_nopose_msexp__f0  <- g3_soccer15_geoall_xf_nvadlr_nopose__f0
      env   subjectBodies        -> 13 fold-0 training reals + sub600..sub629 (43; test bodies
                                    sub4 sub10 sub13 sub16 never train)
            subjectHeightsFile   -> scripts/synthetic_heights_gen4.json
            humanoidAssetSuffix  -> _inertial   (smplx_omomo_<sub>_inertial.xml, the sub4 fix)
            motionScaleReward    -> {enable, referenceStd 0.5157}  (utils/motion_scale.py: the
                                    4th root becomes clamp((0.5157/std)^2, 1/4, 1) per clip --
                                    PRODUCT for clips moving no more than OMOMO's median, the
                                    root at twice that; basketball's median 1.02 m sits at the root)
      train full_experiment_name -> the gen4 arm
      sh    names/paths, HELDOUT hint gains sub4, the gen4 guards (inertial MJCF + retargeted
            refs for every body) and the motion-scale guard

  gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0     <- the bball7 gen4 arm above, PLUS
      the GENERALIST: one teacher on basketball AND soccer (22 people, 109 clips; CPR left
      out -- Jess set it aside, 2026-10-05, the reconstructions are too poor). It is the g3
      activity STUDENT's data layout (omomo_student_g3_act_xf_ret_nvadlr__f0.yaml) on a
      teacher: the merged activity motion dir + retarget tree, per-object mass/restitution
      from the props file (a basketball and a soccer ball differ), one plane for both.
      env   dataSub              -> the 22 bball7 + soccer15 sources (the act dir also holds
                                    CPR clips; dataSub keeps them out)
            motion_file          -> InterAct/behave_cari4d_act
            retargetedMotionDir  -> InterAct/behave_cari4d_act_f0_bodymajor
            objectMass           -> REMOVED;  objectShapeProps -> REMOVED
            objectPropsFile      -> isaacgym/src/intermimic/data/cfg/object_props_g3_act.yaml
                                    (written on the cluster by scripts/merge_activity_data.py,
                                    NOT tracked; the launcher refuses to start without it)
            plane.restitution    -> 0.7 (the student's single plane; the props file's
                                    restitutions were solved against it)
      sh    objectPropsFile guard instead of the objectMass one; the 22-source data loop;
            --mem=96G (109 clips x 43 bodies padded to the 677-frame longest soccer clip
            ~34 GiB of motion -> ~83 GiB at the 2.02x model)

Prerequisites on the cluster BEFORE launching (every launcher guard fails loudly):
  1. retarget the activity clips onto sub4 + sub600-629, into the existing per-source
     trees (additive; the HELDOUT pass of the activity retarget arrays takes the list):
       HELDOUT="sub4 $(seq -f sub%g 600 629 | tr '\\n' ' ')" sbatch --array=0-6  --exclude=simurgh6,simurgh2 scripts/slurm_cari4d_bball7_retarget.sh
       HELDOUT="sub4 $(seq -f sub%g 600 629 | tr '\\n' ' ')" sbatch --array=0-14 --exclude=simurgh6,simurgh2 scripts/slurm_cari4d_soccer_retarget.sh
  2. re-merge each arm's tree (additive, links only what is missing) -- the merge commands
     in those two scripts' headers, unchanged;
  3. add the new bodies to the activity tree for the generalist, from the two arms' trees:
       python3 scripts/merge_retarget_trees.py --sources InterAct/behave_cari4d_bball7_f0_bodymajor InterAct/behave_cari4d_soccer_f0_bodymajor --out InterAct/behave_cari4d_act_f0_bodymajor --bodies-from isaacgym/src/intermimic/data/cfg/omomo_teacher_gen4_act22_geoall_xf_nvadlr_nopose_msexp__f0.yaml

  python3 scripts/make_gen4_egoexo_teacher_cfgs.py            # writes 9 files
  python3 scripts/make_gen4_egoexo_teacher_cfgs.py --diff     # + cfg_diff of each env cfg vs its base
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
TRAIN_REALS = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
               "sub11", "sub12", "sub14", "sub15", "sub17"]
GEN4 = [f"sub{i}" for i in range(600, 630)]
HEIGHTS = "scripts/synthetic_heights_gen4.json"
SUFFIX = "_inertial"
MS_REFERENCE_STD = 0.5157              # OMOMO_new median key-body std, all 4421 clips (utils/motion_scale.py)

BBALL7 = ["sub401", "sub402", "sub404", "sub409", "sub411", "sub412", "sub458"]
SOCCER15 = ["sub405", "sub480", "sub482", "sub484", "sub485", "sub487", "sub488", "sub489",
            "sub490", "sub491", "sub493", "sub494", "sub495", "sub496", "sub497"]
ACT22 = sorted(BBALL7 + SOCCER15, key=lambda s: int(s[3:]))
ACT_MOTION = "InterAct/behave_cari4d_act"
ACT_TREE = "InterAct/behave_cari4d_act_f0_bodymajor"
ACT_PROPS = "isaacgym/src/intermimic/data/cfg/object_props_g3_act.yaml"
ACT_PLANE_RESTITUTION = 0.7

# arm name -> (g3 base arm, sources, --mem, generalist?)
ARMS = {
    "bball7":   ("g3_bball7_geoall_xf_nvadlr_nopose__f0",   BBALL7,   None,  False),
    "soccer15": ("g3_soccer15_geoall_xf_nvadlr_nopose__f0", SOCCER15, None,  False),
    "act22":    ("g3_bball7_geoall_xf_nvadlr_nopose__f0",   ACT22,    "96G", True),
}


def arm(name):
    return f"gen4_{name}_geoall_xf_nvadlr_nopose_msexp__f0"


def strip_header(text):
    """Drop the leading '#' comment block of a yaml/sh file (keeps a shebang)."""
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    if lines and lines[0].startswith("#!"):
        out.append(lines[0]); i = 1
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(out) + "".join(lines[i:])


def yaml_list(items):
    return "[" + ", ".join(f"'{x}'" for x in items) + "]"


def env_header(name):
    base, srcs, _, general = ARMS[name]
    what = ("ONE teacher on basketball AND soccer (22 people, CPR left out): the g3 activity\n"
            "# student's data layout on a teacher -- merged act motion dir + retarget tree, per-object\n"
            "# props file, one plane (0.7)" if general else
            f"the g3 {name} transformer teacher")
    extra = ("#   dataSub / motion_file / retargetedMotionDir -> the 22 sources, the act dir, the act tree\n"
             "#   objectMass + objectShapeProps REMOVED -> objectPropsFile (per-object mass/restitution)\n"
             "#   plane.restitution 0.85 -> 0.7 (the student's plane the props were solved against)\n"
             if general else "")
    return (f"# gen4 EGOEXO TEACHER {arm(name)}: {what}\n"
            f"# on the gen4 bodies and the motion-scale reward. This file is {base}'s env cfg with\n"
            f"# exactly these edits (python3 scripts/cfg_diff.py the two files; pinned by\n"
            f"# tests/test_gen4_egoexo_teacher_cfgs.py):\n"
            f"#   subjectHeightsFile  -> {HEIGHTS}\n"
            f"#   subjectBodies       -> 13 fold-0 training reals + sub600..sub629 (43 bodies;\n"
            f"#                          held out: sub4 sub10 sub13 sub16)\n"
            f"#   humanoidAssetSuffix -> {SUFFIX}  (explicit link inertials: the stock files trip a PhysX\n"
            f"#                          joint-angle rollover on ~1 body in 10)\n"
            f"#   motionScaleReward   -> enable, referenceStd {MS_REFERENCE_STD} (utils/motion_scale.py: the 4th root\n"
            f"#                          becomes a per-clip exponent clamp((ref/std)^2, 1/4, 1); basketball's\n"
            f"#                          median motion 1.02 m sits at the root, OMOMO-sized clips at PRODUCT)\n"
            + extra +
            f"# Recipe otherwise unchanged: geometric_all reward, no pose term, 6 obs horizons,\n"
            f"# retargeted references (the arm's tree must hold sub600-629 -- see the generator's\n"
            f"# docstring for the retarget + merge commands), PSI 3, body-normalised reward.\n")


def train_header(name):
    base = ARMS[name][0]
    return (f"# gen4 EGOEXO TEACHER train cfg for {arm(name)}: byte-identical to the g3 twin's\n"
            f"# train cfg (train/rlg/omomo_teacher_{base}.yaml) except full_experiment_name\n"
            f"# (own checkpoint dir). Transformer 6 tokens, normalize_value, adaptive LR exact-KL 0.06.\n")


def sh_header(name):
    base, srcs, mem, general = ARMS[name]
    g = ("" if not general else
         f"# GENERALIST: basketball + soccer in one teacher (22 sources); objectPropsFile guard\n"
         f"# instead of objectMass, the 22-source data loop, --mem={mem}.\n")
    return (f"# gen4 EGOEXO TEACHER launcher for {arm(name)}: the g3 launcher\n"
            f"# (slurm_teacher_{base}.sh) with the names/paths swapped to this arm, the eval hint\n"
            f"# holding out sub4 as well, and three guards (every body's *_inertial MJCF and\n"
            f"# retargeted reference tree must exist; the motionScaleReward block must be present).\n"
            + g +
            f"# Eval when done (hand-write omomo_eval_{arm(name)}.yaml first, mirroring\n"
            f"# humanoidAssetSuffix + motionScaleReward):  HELDOUT=\"sub4 sub10 sub13 sub16\" sh scripts/eval_one.sh {arm(name)}\n")


def rewrite_env(text, name):
    base, srcs, _, general = ARMS[name]
    if re.search(r"^\s*betas_file:", text, flags=re.M):
        raise SystemExit("ERROR: base has betas_file -- this recipe is nobetas; check the base")
    for key in ("humanoidAssetSuffix", "motionScaleReward"):
        if re.search(rf"^\s*{key}:", text, flags=re.M):
            raise SystemExit(f"ERROR: base already has {key}")
    text, n = re.subn(r"^(\s*)subjectHeightsFile:.*$", rf"\1subjectHeightsFile: {HEIGHTS}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: subjectHeightsFile not found exactly once")
    m = re.search(r"^(\s*)subjectBodies:\s*\n((?:\s*- sub\d+\s*\n)+)", text, flags=re.M)
    if not m:
        raise SystemExit("ERROR: subjectBodies block not found")
    ind = m.group(1)
    text = text[:m.start()] + f"{ind}subjectBodies:\n" + "".join(f"{ind}- {b}\n" for b in TRAIN_REALS + GEN4) + text[m.end():]
    # new keys in alphabetical position (the fleet's cfgs are key-sorted)
    m = re.search(r"^(\s*)(hybridInitProb|initRootHeight|initVel|keyBodies):", text, flags=re.M)
    text = text[:m.start()] + f"{m.group(1)}humanoidAssetSuffix: {SUFFIX}\n" + text[m.start():]
    m = re.search(r"^(\s*)motion_file:", text, flags=re.M)           # 'motionScaleReward' < 'motion_file'
    if not m:
        raise SystemExit("ERROR: motion_file not found (anchor for motionScaleReward)")
    ind = m.group(1)
    text = (text[:m.start()] + f"{ind}motionScaleReward:\n{ind}  enable: true\n"
            f"{ind}  referenceStd: {MS_REFERENCE_STD}\n" + text[m.start():])
    if general:
        text = rewrite_generalist(text)
    return text


def rewrite_generalist(text):
    """The bball7 gen4 env -> the basketball+soccer generalist (see the module docstring)."""
    def one(pat, rep, what):
        nonlocal text
        text, n = re.subn(pat, rep, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"ERROR: generalist edit '{what}' did not match exactly once")
    one(r"^(\s*dataSub:).*$", rf"\1 {yaml_list(ACT22)}", "dataSub")
    one(r"^(\s*motion_file:).*$", rf"\1 {ACT_MOTION}", "motion_file")
    one(r"^(\s*retargetedMotionDir:).*$", rf"\1 {ACT_TREE}", "retargetedMotionDir")
    one(r"^\s*objectMass:.*\n", "", "objectMass removed")
    one(r"^(\s*)objectShapeProps:\s*\n\s+restitution:.*\n", "", "objectShapeProps removed")
    one(r"^(\s*)(objectConvexHull:.*)$", rf"\1\2\n\1objectPropsFile: {ACT_PROPS}", "objectPropsFile")
    # plane.restitution: the only 'restitution' left under 'plane:' after the block removal
    m = re.search(r"^(\s*)plane:\s*\n((?:\s+\w+:.*\n)+)", text, flags=re.M)
    if not m:
        raise SystemExit("ERROR: plane block not found")
    blk, n = re.subn(r"^(\s*restitution:).*$", rf"\1 {ACT_PLANE_RESTITUTION}", m.group(2), count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: plane.restitution not found exactly once")
    text = text[:m.start(2)] + blk + text[m.end(2):]
    return text


def rewrite_train(text, name):
    base = ARMS[name][0]
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{base}\s*$",
                     rf"\1smplx_teacher_{arm(name)}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: full_experiment_name of {base} not found exactly once")
    return new


GUARD_GEN4 = '''# gen4 guards: every body must have its *_inertial MJCF and its retargeted
# reference tree, or the run would silently load stock files / miss bodies.
RT_DIR=$(grep -E '^\\s*retargetedMotionDir:' "$CFG_ENV" | awk '{print $2}')
if ! grep -qE '^\\s*humanoidAssetSuffix:\\s*_inertial\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: gen4 arm without humanoidAssetSuffix: _inertial in $CFG_ENV" >&2; exit 1
fi
for b in $(grep -E '^\\s*- sub[0-9]+\\s*$' "$CFG_ENV" | awk '{print $2}'); do
    f="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${b}_inertial.xml"
    [ -f "$f" ] || { echo "[teacher] ERROR: missing $f (run scripts/mjcf_add_inertials.py --all)" >&2; exit 1; }
    [ -d "$RT_DIR/$b" ] || { echo "[teacher] ERROR: no retargeted refs $RT_DIR/$b (retarget the activity clips onto the gen4 bodies + re-merge; see scripts/make_gen4_egoexo_teacher_cfgs.py)" >&2; exit 1; }
done
# Motion-scale guard: this arm trains on the per-clip reward exponent; a cfg that
# lost the block would train the plain recipe under this arm's name.
if ! grep -qE '^\\s*motionScaleReward:' "$CFG_ENV" || ! grep -qE '^\\s*referenceStd:\\s*REFSTD\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: motion-scale arm without motionScaleReward.referenceStd REFSTD in $CFG_ENV" >&2; exit 1
fi
'''

GUARD_PROPS = '''# Generalist: a basketball and a soccer ball do not share a mass or a restitution,
# so the per-object props file must be there (written on the cluster by
# scripts/merge_activity_data.py; it is not tracked).
PROPS=$(grep -oE '^\\s*objectPropsFile:\\s*\\S+' "$CFG_ENV" | awk '{print $2}')
[ -n "$PROPS" ] || { echo "[teacher] ERROR: generalist without objectPropsFile in $CFG_ENV" >&2; exit 1; }
[ -f "$PROPS" ] || { echo "[teacher] ERROR: $PROPS missing -- run scripts/merge_activity_data.py (see its docstring)" >&2; exit 1; }
'''


def rewrite_sh(text, name, is_sbatch):
    base, srcs, mem, general = ARMS[name]
    text = text.replace(base, arm(name))         # job name, output, cfg paths, echo, ckpt dir
    text = text.replace('HELDOUT="sub10 sub13 sub16"', 'HELDOUT="sub4 sub10 sub13 sub16"')
    if is_sbatch:
        if mem:
            text, n = re.subn(r"^#SBATCH --mem=\S+$", f"#SBATCH --mem={mem}", text, count=1, flags=re.M)
            if n != 1:
                raise SystemExit("ERROR: #SBATCH --mem line not found exactly once")
        return text
    # gen4 + motion-scale guards go right after the retargetedMotionDir guard block
    anchor = "retarget arm without retargetedMotionDir"
    i = text.find(anchor)
    if i < 0:
        raise SystemExit("ERROR: retargetedMotionDir guard not found in launcher")
    j = text.find("\nfi\n", i) + len("\nfi\n")
    text = text[:j] + GUARD_GEN4.replace("REFSTD", str(MS_REFERENCE_STD)) + text[j:]
    if general:
        # the objectMass guard (bball7: 'bball7 without objectMass') -> objectPropsFile guard
        m = re.search(r"^# Per-clip balls need per-object mass.*\nif ! grep -qE '\^\\s\*objectMass:' \"\$CFG_ENV\"; then\n.*\nfi\n", text, flags=re.M)
        if not m:
            raise SystemExit("ERROR: objectMass guard not found in the bball7 launcher")
        text = text[:m.start()] + GUARD_PROPS + text[m.end():]
        text, n = re.subn(r"^for s in (sub\d+ )+sub\d+; do$", f"for s in {' '.join(ACT22)}; do", text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("ERROR: the per-source data loop not found exactly once")
        text = text.replace("run scripts/slurm_cari4d_bball7_convert.sh",
                            "build it with scripts/merge_activity_data.py (bball7 + soccer15)")
    # the recipe echo lines name the g3 arm's lineage; say what this arm is instead
    srcs_txt = " ".join(srcs)
    text, n = re.subn(r'^echo "\[teacher\] G3 RECIPE .*$',
                      f'echo "[teacher] GEN4 EGOEXO RECIPE {arm(name)} (sources {srcs_txt}; 43 gen4 bodies, '
                      f'_inertial MJCFs, 6-TOKEN TRANSFORMER, normval+adaptive LR, NO POSE TERM, MOTION-SCALE EXPONENT '
                      f'ref {MS_REFERENCE_STD} capped at product): no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"',
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: G3 RECIPE echo not found exactly once")
    text, n = re.subn(r'^echo "\[teacher\] METHOD CANDIDATE .*$',
                      f'echo "[teacher] gen4 twin of {base}: gen4 bodies + inertial MJCFs + motionScaleReward'
                      + ("; GENERALIST over basketball + soccer (objectPropsFile, plane 0.7)" if general else "") + '"',
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: METHOD CANDIDATE echo not found exactly once")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="print cfg_diff of each env cfg vs its g3 base")
    a = ap.parse_args()
    for name, (base, srcs, mem, general) in ARMS.items():
        env_src = CFG / f"omomo_teacher_{base}.yaml"
        tr_src = CFG / "train/rlg" / f"omomo_teacher_{base}.yaml"
        sh_src = REPO / f"slurm_teacher_{base}.sh"
        for p in (env_src, tr_src, sh_src):
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p}")
        env_out = CFG / f"omomo_teacher_{arm(name)}.yaml"
        tr_out = CFG / "train/rlg" / f"omomo_teacher_{arm(name)}.yaml"
        sh_out = REPO / f"slurm_teacher_{arm(name)}.sh"
        env_out.write_text(env_header(name) + rewrite_env(strip_header(env_src.read_text()), name))
        tr_out.write_text(train_header(name) + rewrite_train(strip_header(tr_src.read_text()), name))
        sh_text = sh_src.read_text()
        shebang = sh_text.splitlines(keepends=True)[0]
        body = strip_header(sh_text)[len(shebang):]
        sbatch = "".join(l for l in sh_text.splitlines(keepends=True) if l.startswith("#SBATCH"))
        sh_out.write_text(shebang + rewrite_sh(sbatch, name, True) + sh_header(name) + rewrite_sh(body, name, False))
        sh_out.chmod(0o755)
        print(f"{name:9s}: {env_out.name}  {tr_out.name}  {sh_out.name}")
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {3 * len(ARMS)} files for {len(ARMS)} gen4 EgoExo teacher arms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gen4 OBJECT-GROUP teachers: four OMOMO teachers, one per object family, on the
gen4 recipe. The OMOMO analogue of the EgoExo4D activity arms (bball7, soccer15,
cpr13): one teacher per "thing being manipulated", every motion done with it
inside. Groups (Jess, 2026-09-30; monitor rides with boxes for this first pass):

    boxes   largebox smallbox plasticbox suitcase trashcan monitor   1315 train clips
    stands  tripod clothesstand floorlamp                              827
    chairs  whitechair woodchair                                       566
    tables  largetable smalltable                                      648

Each arm is a COPY of the gen4 src1 nopose transformer teacher
(gen4_omomo_geoall_src1_xf_nvadlr_nopose__f0) with exactly these edits, nothing
else (tests/test_gen4_activity_teacher_cfgs.py pins it):

  env cfg   dataSub              -> the 13 fold-0 training sources (was ['sub1'])
            dataObjects          -> the group's objects (NEW key; intermimic.py:217
                                    keeps only clips of those objects)
            raggedMotionData     -> true (NEW key: per-clip storage; padded boxes
                                    would need ~423 GiB at the 2.02x model)
            retargetedMotionDir  -> InterAct/OMOMO_retarget_contact_srcall13
                                    (the merged 13-source tree; a group arm reads
                                    clips of several sources from ONE dir)
  train cfg full_experiment_name -> smplx_teacher_gen4_omomo_geoall_<group>_xf_nvadlr_nopose__f0
  launcher  names/paths          -> the group arm; --mem from
                                    scripts/motion_memory_budget.py --objects (ragged,
                                    2.02x model, rounded up); ragged + merged-tree guards

Sampling is UNCHANGED (kept on purpose, Jess 2026-09-30): env e owns object
e % n_obj at creation, so every object in a group gets the same share of envs
regardless of clip count (monitor = 1/6 of the boxes arm's envs, not 10%).

Prerequisites on the cluster BEFORE launching (the launcher guards fail loudly):
  - the gen4 per-source retarget array has finished:  sbatch --exclude=simurgh6,simurgh2 slurm_retarget_gen4.sh
  - the merged tree has the gen4 bodies: re-run the merge_retarget_trees.py command
    in slurm_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0.sh's header with
    --bodies-from isaacgym/src/intermimic/data/cfg/omomo_teacher_gen4_omomo_geoall_boxes_xf_nvadlr_nopose__f0.yaml
    (it skips links that already exist, so it only ADDS sub4 + sub600-629)

  python3 scripts/make_gen4_activity_teacher_cfgs.py            # writes 12 files
  python3 scripts/make_gen4_activity_teacher_cfgs.py --diff     # + cfg_diff of each env cfg vs its base

MOTION-SCALE VARIANT (Jess, 2026-10-03: "train on the new reward"). --motion-scale
writes a SECOND set of four arms, tagged _msexp, with one more env edit:

  motionScaleReward: {enable: true, referenceStd: 0.5157}

utils/motion_scale.py: the reward's 4th root becomes a per-clip exponent
clamp((0.5157 / std)^2, 1/4, 1) -- a clip moving no more than OMOMO's median
(0.5157 m key-body std) is graded as PRODUCT, a clip moving twice that or more
keeps today's root. The exponent is CAPPED at 1 (2026-10-05): stricter-than-
product was measured to shrink a clip's pull on the policy at the tracking
quality these teachers reach, so there is no floor and no further knob. These
arms differ from the gen4 per-source fleet by the grouping AND the reward (the
group-vs-source read is confounded; accepted).

  python3 scripts/make_gen4_activity_teacher_cfgs.py --motion-scale --diff
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
BASE = "gen4_omomo_geoall_src1_xf_nvadlr_nopose__f0"
TRAIN_SOURCES = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
                 "sub11", "sub12", "sub14", "sub15", "sub17"]
MERGED_TREE = "InterAct/OMOMO_retarget_contact_srcall13"
MS_TAG = "_msexp"                      # motion-scale-exponent arms: ..._nopose_msexp__f0
MS_REFERENCE_STD = 0.5157              # OMOMO_new median key-body std, all 4421 clips (utils/motion_scale.py)

# group -> (objects, #SBATCH --mem). Memory = scripts/motion_memory_budget.py
# --sources <13> --bodies 43 --psi 3 --objects <group>, ragged row, 2.02x model,
# rounded UP to the next 16G (boxes 189 / stands 128 / chairs 94 / tables 104 GiB,
# measured 2026-09-30 on ~/new_one/OMOMO_new). The 2.02x factor is the PADDED
# loader's; ragged should be nearer 1.0x but that is still unmeasured.
GROUPS = {
    "boxes":  (["largebox", "smallbox", "plasticbox", "suitcase", "trashcan", "monitor"], "208G"),
    "stands": (["tripod", "clothesstand", "floorlamp"], "144G"),
    "chairs": (["whitechair", "woodchair"], "112G"),
    "tables": (["largetable", "smalltable"], "112G"),
}
ALL_OBJECTS = {"clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
               "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"}


def arm(g, tag=""):
    return f"gen4_omomo_geoall_{g}_xf_nvadlr_nopose{tag}__f0"


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


def env_header(g, tag="", ms=False):
    objs, _ = GROUPS[g]
    n_edits = "five" if ms else "four"
    ms_lines = "" if not ms else (
        f"#   motionScaleReward   -> enable, referenceStd {MS_REFERENCE_STD} (utils/motion_scale.py: the 4th root\n"
        f"#                          becomes a per-clip exponent clamp((ref/std)^2, 1/4, 1): a clip moving no more\n"
        f"#                          than OMOMO's median is graded as PRODUCT, twice that or more keeps the root)\n")
    return (f"# gen4 OBJECT-GROUP TEACHER {arm(g, tag)}: every OMOMO clip whose object is one of\n"
            f"#   {' '.join(objs)}\n"
            f"# from the 13 fold-0 training sources, on the gen4 bodies. This file is the gen4 src1\n"
            f"# nopose transformer teacher's env cfg ({BASE}) with exactly {n_edits} edits\n"
            f"# (python3 scripts/cfg_diff.py the two files; pinned by tests/test_gen4_activity_teacher_cfgs.py):\n"
            f"#   dataSub             -> the 13 training sources (held out: sub4 sub10 sub13 sub16)\n"
            f"#   dataObjects         -> the group (intermimic.py keeps only these objects' clips)\n"
            f"#   raggedMotionData    -> true (per-clip storage; see scripts/make_gen4_activity_teacher_cfgs.py)\n"
            f"#   retargetedMotionDir -> {MERGED_TREE} (merged 13-source tree, must hold sub600-629)\n"
            + ms_lines +
            f"# Sampling unchanged: env e owns object e % n_obj, uniform clips within the object.\n"
            f"# Recipe otherwise the gen4 fleet's: geometric_all reward, no pose term, 6 obs horizons,\n"
            f"# retargeted references, PSI 3, body-normalised reward, _inertial MJCFs, 43 bodies.\n")


def train_header(g, tag=""):
    return (f"# gen4 OBJECT-GROUP TEACHER train cfg for {arm(g, tag)}: byte-identical to the gen4 src1\n"
            f"# twin's train cfg (train/rlg/omomo_teacher_{BASE}.yaml) except full_experiment_name\n"
            f"# (own checkpoint dir). Transformer 6 tokens, normalize_value, adaptive LR exact-KL 0.06.\n")


def sh_header(g, tag="", ms=False):
    objs, mem = GROUPS[g]
    extra = "" if not ms else (
        f"# PLUS the motion-scale reward exponent (motionScaleReward, referenceStd {MS_REFERENCE_STD}; a third\n"
        f"# guard checks the block is present) -- this arm trains on the NEW reward, so it differs from\n"
        f"# the gen4 per-source fleet by the grouping AND the reward.\n")
    mirror = " + motionScaleReward" if ms else ""
    return (f"# gen4 OBJECT-GROUP TEACHER launcher for {arm(g, tag)}: the gen4 src1 launcher\n"
            f"# (slurm_teacher_{BASE}.sh) with the names/paths swapped to this arm, --mem={mem}\n"
            f"# (ragged budget, scripts/motion_memory_budget.py --objects {' '.join(objs)}), and two\n"
            f"# more guards (raggedMotionData on; the merged retarget tree holds this group's clips\n"
            f"# for a real AND a gen4 body). Nothing else differs.\n"
            + extra +
            f"# Eval when done (hand-write omomo_eval_{arm(g, tag)}.yaml first, mirroring\n"
            f"# humanoidAssetSuffix + dataObjects{mirror}):  HELDOUT=\"sub4 sub10 sub13 sub16\" sh scripts/eval_one.sh {arm(g, tag)}\n")


def rewrite_env(text, g, ms=False):
    objs, _ = GROUPS[g]
    text, n = re.subn(r"^(\s*)dataSub:\s*\['sub1'\]\s*$",
                      rf"\1dataSub: {yaml_list(TRAIN_SOURCES)}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: dataSub: ['sub1'] not found exactly once in base env cfg")
    for key in ("dataObjects", "raggedMotionData", "motionScaleReward"):
        if re.search(rf"^\s*{key}:", text, flags=re.M):
            raise SystemExit(f"ERROR: base already has {key}")
    # new keys in alphabetical position (the fleet's cfgs are key-sorted)
    m = re.search(r"^(\s*)dataSub:", text, flags=re.M)
    text = text[:m.start()] + f"{m.group(1)}dataObjects: {yaml_list(objs)}\n" + text[m.start():]
    m = re.search(r"^(\s*)resetThresholds:", text, flags=re.M)
    if not m:
        raise SystemExit("ERROR: resetThresholds not found (anchor for raggedMotionData)")
    text = text[:m.start()] + f"{m.group(1)}raggedMotionData: true\n" + text[m.start():]
    if ms:
        # 'motionScaleReward' sorts before 'motion_file' ('S' < '_')
        m = re.search(r"^(\s*)motion_file:", text, flags=re.M)
        if not m:
            raise SystemExit("ERROR: motion_file not found (anchor for motionScaleReward)")
        ind = m.group(1)
        blk = f"{ind}motionScaleReward:\n{ind}  enable: true\n{ind}  referenceStd: {MS_REFERENCE_STD}\n"
        text = text[:m.start()] + blk + text[m.start():]
    text, n = re.subn(r"^(\s*retargetedMotionDir:\s*)InterAct/OMOMO_retarget_contact_src1\s*$",
                      rf"\1{MERGED_TREE}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: retargetedMotionDir src1 not found exactly once in base env cfg")
    return text


def rewrite_train(text, g, tag=""):
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{BASE}\s*$",
                     rf"\1smplx_teacher_{arm(g, tag)}", text, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: full_experiment_name of the base not found exactly once")
    return new


GUARD_GROUP = """# Object-group guards: this arm does not fit padded (see header), so a cfg that
# lost the ragged flag must not get to OOM the node 20 minutes in; and the
# merged tree must hold this group's clips for a real body AND a gen4 body, or
# the task dies at startup on the first missing (body, clip) file anyway.
if ! grep -qE '^\\s*raggedMotionData:\\s*[Tt]rue' "$CFG_ENV"; then
    echo "[teacher] ERROR: object-group arm without raggedMotionData in $CFG_ENV" >&2; exit 1
fi
for o in OBJECTS; do
    for b in sub2 sub600; do
        if ! ls "$RT_DIR"/$b/*_${o}_*.pt >/dev/null 2>&1; then
            echo "[teacher] ERROR: $RT_DIR has no ${o} clips under body $b -- re-run the" \\
                 "merge_retarget_trees.py command (see scripts/make_gen4_activity_teacher_cfgs.py)" >&2; exit 1
        fi
    done
done
"""

GUARD_MS = """# Motion-scale guard: this arm trains on the per-clip reward exponent; a cfg that
# lost the block would train the plain group arm under this arm's name.
if ! grep -qE '^\\s*motionScaleReward:' "$CFG_ENV" || ! grep -qE '^\\s*referenceStd:\\s*REFSTD\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: motion-scale arm without motionScaleReward.referenceStd REFSTD in $CFG_ENV" >&2; exit 1
fi
"""


def guard(g, ms=False):
    objs, _ = GROUPS[g]
    out = GUARD_GROUP.replace("OBJECTS", " ".join(objs))
    if ms:
        out += GUARD_MS.replace("REFSTD", str(MS_REFERENCE_STD))
    return out


def rewrite_sbatch(text, g, tag=""):
    """The #SBATCH block: names swapped, --mem set to the group's budget."""
    _, mem = GROUPS[g]
    text = text.replace(BASE, arm(g, tag))            # job name, output
    text, n = re.subn(r"^#SBATCH --mem=\S+$", f"#SBATCH --mem={mem}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: #SBATCH --mem line not found exactly once")
    return text


def rewrite_body(text, g, tag="", ms=False):
    """The launcher body: names swapped, group guards added, recipe echoes rewritten."""
    objs, _ = GROUPS[g]
    text = text.replace(BASE, arm(g, tag))            # cfg paths, echo, ckpt dir
    # the gen4 guard loop ends with 'done'; the group guards go right after it
    anchor = "run slurm_retarget_gen.sh for this source"
    i = text.find(anchor)
    if i < 0:
        raise SystemExit("ERROR: gen4 retarget guard not found in launcher")
    j = text.find("\ndone\n", i) + len("\ndone\n")
    text = text[:j] + guard(g, ms) + text[j:]
    reward_note = "" if not ms else f", MOTION-SCALE EXPONENT (ref {MS_REFERENCE_STD}, capped at product)"
    ms_note = "" if not ms else "; reward exponent per clip from its key-body std (utils/motion_scale.py)"
    # the two recipe echo lines name the src1 arm's lineage; say what this arm is instead
    text, n = re.subn(r'^echo "\[teacher\] G3 RECIPE .*$',
                      f'echo "[teacher] GEN4 OBJECT-GROUP RECIPE {arm(g, tag)} (OMOMO data, objects {" ".join(objs)}, '
                      f'13 sources, RAGGED motion, 6-TOKEN TRANSFORMER, normval+adaptive LR, NO POSE TERM{reward_note}): '
                      f'43 bodies, no betas, gate resets=true, rollout 50, buf=12.0 num_envs=$NUM_ENVS"',
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: G3 RECIPE echo not found exactly once")
    text, n = re.subn(r'^echo "\[teacher\] METHOD CANDIDATE .*$',
                      f'echo "[teacher] OBJECT GROUP {g}: the gen4 src1 nopose recipe with dataSub = 13 sources, '
                      f'dataObjects = {" ".join(objs)}, ragged storage, merged retarget tree; sampling unchanged '
                      f'(env e owns object e % n_obj){ms_note}"',
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ERROR: METHOD CANDIDATE echo not found exactly once")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="print cfg_diff of each env cfg vs the gen4 src1 base")
    ap.add_argument("--motion-scale", action="store_true",
                    help=f"write the _msexp arms instead: motionScaleReward on with referenceStd "
                         f"{MS_REFERENCE_STD} (exponent clamp((ref/std)^2, 1/4, 1); no floor)")
    a = ap.parse_args()
    ms = a.motion_scale
    tag = MS_TAG if ms else ""
    listed = [o for objs, _ in GROUPS.values() for o in objs]
    if sorted(listed) != sorted(ALL_OBJECTS):
        raise SystemExit(f"ERROR: groups must partition the 13 OMOMO objects exactly; got {sorted(listed)}")
    env_src = CFG / f"omomo_teacher_{BASE}.yaml"
    tr_src = CFG / "train/rlg" / f"omomo_teacher_{BASE}.yaml"
    sh_src = REPO / f"slurm_teacher_{BASE}.sh"
    for p in (env_src, tr_src, sh_src):
        if not p.is_file():
            raise SystemExit(f"ERROR: base file missing: {p}")
    for g in GROUPS:
        env_out = CFG / f"omomo_teacher_{arm(g, tag)}.yaml"
        tr_out = CFG / "train/rlg" / f"omomo_teacher_{arm(g, tag)}.yaml"
        sh_out = REPO / f"slurm_teacher_{arm(g, tag)}.sh"
        env_out.write_text(env_header(g, tag, ms) + rewrite_env(strip_header(env_src.read_text()), g, ms))
        tr_out.write_text(train_header(g, tag) + rewrite_train(strip_header(tr_src.read_text()), g, tag))
        sh_text = sh_src.read_text()
        shebang = sh_text.splitlines(keepends=True)[0]
        body = strip_header(sh_text)[len(shebang):]
        sbatch = "".join(l for l in sh_text.splitlines(keepends=True) if l.startswith("#SBATCH"))
        sh_out.write_text(shebang + rewrite_sbatch(sbatch, g, tag) + sh_header(g, tag, ms) + rewrite_body(body, g, tag, ms))
        sh_out.chmod(0o755)
        print(f"{g:7s}: {env_out.name}  {tr_out.name}  {sh_out.name}")
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {3 * len(GROUPS)} files for {len(GROUPS)} object-group teacher arms"
          + (" (motion-scale variant)" if ms else ""))


if __name__ == "__main__":
    main()

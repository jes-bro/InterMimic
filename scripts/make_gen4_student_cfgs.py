#!/usr/bin/env python3
"""The gen4 OBJECT-GROUP student: one 6-token transformer distilled from the four
gen4 object-group msexp teachers (boxes / stands / chairs / tables), routed by the
clip's OBJECT through InterMimicDistillG4 (Jess, 2026-10-15; teachers at their
LATEST snapshot, not pinned).

    student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0

Each of its three files is a COPY of the g3 OMOMO XF student
(student_g3_omomo_xf_ret_nvadlr__f0) with exactly these edits, nothing else
(tests/test_gen4_student_cfgs.py pins it):

  env cfg   subjectBodies        -> the gen4 roster: 13 fold-0 reals + sub600-629 (43), read
                                    from the chairs msexp TEACHER cfg so student and teachers
                                    cannot disagree on a body
            subjectHeightsFile   -> scripts/synthetic_heights_gen4.json
            humanoidAssetSuffix  -> _inertial (NEW key: the regenerated MJCFs, sub4 PhysX fix)
            rewardTerms.pose     -> enable false (the gen4 teachers are nopose)
            motionScaleReward    -> enable, referenceStd 0.5157 (NEW block). The DAgger agent is
                                    not pure imitation: its PPO term reads the env reward
                                    (learning/intermimic_agent_distill.py), so the student is
                                    graded by the reward its teachers trained under
            teacherPolicy        -> checkpoints/teachers/gen4_omomo_groups_msexp (OBJECT-keyed
                                    manifest from collect_g3_teachers.py --gen4-objects)
            teacherPolicyCFG     -> the chairs msexp teacher's train cfg (the fleet's network:
                                    6-token transformer; all four teachers share it)
  train cfg full_experiment_name -> smplx_student_<arm>
  launcher  names/paths; --task InterMimicDistillG4; fold guard holds out sub4 too; the
            teacher-set guard checks the manifest covers the 13 OBJECTS (not sources); the
            gen4 guards (humanoidAssetSuffix + every body's _inertial MJCF + retarget dir) and
            the motion-scale guard are added. --mem stays 480G (all 13 objects, 13 sources:
            the srcall13 / all-objects budget).

Unchanged on purpose: all 13 objects (no dataObjects), 13 sources, merged srcall13
tree, ragged storage, studentObsHorizons / numObsRetarget 9594, normalize_value +
adaptive LR (exact-KL 0.06), 2048 envs, 7-day walltime.

Prerequisites on the machine that trains it (the launcher guards fail loudly):
  - the four gen4 object-group msexp teachers have checkpoints
  - python3 scripts/collect_g3_teachers.py --gen4-objects boxes stands chairs tables \\
        --out checkpoints/teachers/gen4_omomo_groups_msexp
  - the merged tree holds the gen4 bodies (sub600-629); every body's _inertial MJCF

  python3 scripts/make_gen4_student_cfgs.py            # writes 3 files
  python3 scripts/make_gen4_student_cfgs.py --diff     # + cfg_diff vs the g3 student and vs the chairs teacher
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"

BASE = "student_g3_omomo_xf_ret_nvadlr__f0"
ARM = "student_gen4_omomo_groups_xf_ret_nvadlr_nopose_msexp__f0"
# the launchers name the arm WITHOUT the student_ prefix (job stu-<short>, cfg omomo_student_<short>)
SHORT_BASE, SHORT_ARM = BASE[len("student_"):], ARM[len("student_"):]
TEACHER_ARM = "gen4_omomo_geoall_chairs_xf_nvadlr_nopose_msexp__f0"      # roster + network come from here
TEACHER_SET = "checkpoints/teachers/gen4_omomo_groups_msexp"
GROUPS = ["boxes", "stands", "chairs", "tables"]
OBJECTS = ["clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
           "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"]
MS_REFERENCE_STD = 0.5157
HELDOUT = ["sub4", "sub10", "sub13", "sub16"]


def strip_header(text):
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    if lines and lines[0].startswith("#!"):
        out.append(lines[0]); i = 1
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(out) + "".join(lines[i:])


def sub1(pattern, repl, text, what, flags=re.M, literal=True):
    """Replace exactly one match or refuse: a silent no-op would ship the g3 value.
    repl is inserted LITERALLY (no backslash templating) unless literal=False,
    which the few replacements that reuse a captured group (\\1) pass."""
    new, n = re.subn(pattern, (lambda m: repl) if literal else repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"ERROR: {what}: expected exactly one match, got {n}")
    return new


def teacher_roster():
    import yaml
    env = yaml.safe_load(open(CFG / f"omomo_teacher_{TEACHER_ARM}.yaml"))["env"]
    bodies = list(env["subjectBodies"])
    if len(bodies) != 43 or set(bodies) & set(HELDOUT):
        raise SystemExit(f"ERROR: teacher roster is not the 43-body gen4 set: {bodies}")
    if env.get("humanoidAssetSuffix") != "_inertial" or env.get("motionScaleReward", {}).get("referenceStd") != MS_REFERENCE_STD:
        raise SystemExit("ERROR: the chairs msexp teacher cfg is not what this generator assumes")
    return bodies


# ---------------------------------------------------------------- env cfg
ENV_HEADER = f"""# STUDENT, gen4 OBJECT GROUPS, XF / RET / NVADLR / NOPOSE / MSEXP. Distills the four gen4
# object-group teachers (boxes / stands / chairs / tables, motion-scale reward, latest
# snapshots) into one 6-TOKEN TRANSFORMER student via InterMimicDistillG4, which routes
# each env to the teacher that owns its clip's OBJECT (teachers.yaml is object-keyed).
# This file is the g3 OMOMO XF student's env cfg (omomo_{BASE}.yaml) with exactly these
# edits (python3 scripts/cfg_diff.py; pinned by tests/test_gen4_student_cfgs.py):
#   subjectBodies        -> the gen4 roster (13 reals + sub600-629), read from the teachers' cfg
#   subjectHeightsFile   -> scripts/synthetic_heights_gen4.json
#   humanoidAssetSuffix  -> _inertial (regenerated MJCFs; sub4 PhysX fix)
#   rewardTerms.pose     -> enable false (gen4 teachers are nopose)
#   motionScaleReward    -> enable, referenceStd {MS_REFERENCE_STD} (the teachers' reward; the DAgger
#                           agent's PPO term reads the env reward, so the student is graded by it)
#   teacherPolicy        -> {TEACHER_SET} (OBJECT-keyed manifest)
#   teacherPolicyCFG     -> the chairs msexp teacher's train cfg (the fleet's 6-token transformer)
# Everything else is the g3 student's: all 13 objects, 13 sources, merged srcall13 tree,
# ragged storage, student obs 6 horizons x 1599 = 9594. Held out: sub4 sub10 sub13 sub16.
# Generated by scripts/make_gen4_student_cfgs.py -- never edit as-run cfgs.
"""


def rewrite_env(text):
    bodies = teacher_roster()
    block = "  subjectBodies:\n" + "".join(f"  - {b}\n" for b in bodies)
    text = sub1(r"^  subjectBodies:\n(?:  - sub\d+\n)+", block, text, "subjectBodies block")
    text = sub1(r"^  subjectHeightsFile: scripts/synthetic_heights\.json$",
                "  subjectHeightsFile: scripts/synthetic_heights_gen4.json", text, "subjectHeightsFile")
    for key in ("humanoidAssetSuffix", "motionScaleReward", "dataObjects"):
        if re.search(rf"^\s*{key}:", text, flags=re.M):
            raise SystemExit(f"ERROR: base student cfg already has {key}")
    text = sub1(r"^(  hybridInitProb:)", r"  humanoidAssetSuffix: _inertial\n\1", text, "humanoidAssetSuffix anchor", literal=False)
    text = sub1(r"^    pose:\n      enable: true\n      lambda: 0\.02$",
                "    pose:\n      enable: false\n      lambda: 0.02", text, "rewardTerms.pose.enable")
    text = sub1(r"^(  motion_file:)",
                f"  motionScaleReward:\n    enable: true\n    referenceStd: {MS_REFERENCE_STD}\n\\1",
                text, "motionScaleReward anchor", literal=False)
    text = sub1(r"^  teacherPolicy: checkpoints/teachers/g3_omomo$", f"  teacherPolicy: {TEACHER_SET}", text, "teacherPolicy")
    text = sub1(r"^  teacherPolicyCFG: intermimic/data/cfg/train/rlg/omomo_teacher_g3_omomo_geoall__f0\.yaml$",
                f"  teacherPolicyCFG: intermimic/data/cfg/train/rlg/omomo_teacher_{TEACHER_ARM}.yaml", text, "teacherPolicyCFG")
    text = sub1(r"^  # --- distillation \(InterMimicDistillG3\) ---$",
                "  # --- distillation (InterMimicDistillG4: routed by the clip's OBJECT) ---", text, "distillation comment")
    return ENV_HEADER + text


# ---------------------------------------------------------------- train cfg
TRAIN_HEADER = f"""# STUDENT train cfg for {ARM}: byte-identical to the g3 OMOMO XF student's
# (train/rlg/omomo_{BASE}.yaml) except full_experiment_name. 6-token transformer,
# normalize_value + adaptive LR (exact-KL 0.06), expert_loss_coef 1, save_intermediate on.
# Generated by scripts/make_gen4_student_cfgs.py.
"""


def rewrite_train(text):
    text = sub1(rf"^(\s*full_experiment_name:\s*)smplx_{BASE}\s*$", rf"\1smplx_{ARM}", text, "full_experiment_name", literal=False)
    return TRAIN_HEADER + text


# ---------------------------------------------------------------- launcher
SH_HEADER = f"""# STUDENT, gen4 OBJECT GROUPS, XF / RET / NVADLR / NOPOSE / MSEXP: the four gen4 object-group
# teachers (boxes / stands / chairs / tables, motion-scale reward, LATEST snapshots) distilled
# into one 6-TOKEN TRANSFORMER student via InterMimicDistillG4 (routes each env by its clip's
# OBJECT; teachers.yaml is object-keyed). This launcher is the g3 OMOMO XF student's
# (slurm_{BASE}.sh) with the names/paths swapped, --task InterMimicDistillG4, the fold
# guard holding out sub4 too, the teacher-set guard checking the 13 OBJECTS, and the gen4 +
# motion-scale guards added. --mem=480G as the srcall13 / all-objects arms (same data).
#
# BEFORE THE FIRST SUBMISSION, on the machine that holds the teacher checkpoints:
#   1. the merged retarget tree must hold the gen4 bodies (sub600-629) -- see
#      scripts/make_gen4_activity_teacher_cfgs.py for the merge command
#   2. python3 scripts/collect_g3_teachers.py --gen4-objects {' '.join(GROUPS)} \\
#          --out {TEACHER_SET}
#      (latest snapshot of each; refuses a partial set or an object owned twice; records epochs)
# Startup prints one line per teacher (file, objects, epoch) and the object->teacher table, and
# refuses on a source-keyed manifest, an uncovered object, or any obs-width / token mismatch.
# Eval when done: hand-write omomo_eval_{ARM}.yaml (evalTask InterMimicDistillG4) first.
"""

GUARD_TEACHERS_PY = '''import yaml,sys
t=yaml.safe_load(open('$TP/teachers.yaml'))['teachers']
if any('sources' in e for e in t):
    print('[student] ERROR: teachers.yaml is keyed by sources -- a g3 manifest; this student routes by OBJECT (collect with --gen4-objects)'); sys.exit(1)
have={o for e in t for o in e['objects']}
need={'clothesstand','floorlamp','largebox','largetable','monitor','plasticbox','smallbox','smalltable','suitcase','trashcan','tripod','whitechair','woodchair'}
miss=sorted(need-have)
print('[student] teachers:', len(t), 'files;', 'missing objects:', miss or 'none')
sys.exit(1 if miss else 0)"; then
    echo "[student] ERROR: $TP/teachers.yaml does not cover every OMOMO object" >&2; exit 1
fi'''

GUARD_GEN4 = f'''
# gen4 guards: every body must have its *_inertial MJCF and its retargeted reference
# dir in the merged tree, or the run would silently load stock files / miss bodies.
if ! grep -qE '^\\s*humanoidAssetSuffix:\\s*_inertial\\s*$' "$CFG_ENV"; then
    echo "[student] ERROR: gen4 student without humanoidAssetSuffix: _inertial in $CFG_ENV" >&2; exit 1
fi
for b in $(grep -E '^\\s*- sub[0-9]+\\s*$' "$CFG_ENV" | awk '{{print $2}}'); do
    f="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${{b}}_inertial.xml"
    [ -f "$f" ] || {{ echo "[student] ERROR: missing $f (run scripts/mjcf_add_inertials.py --all)" >&2; exit 1; }}
    [ -d "$RT/$b" ] || {{ echo "[student] ERROR: no retargeted refs $RT/$b in the merged tree (re-run merge_retarget_trees.py)" >&2; exit 1; }}
done
# Motion-scale guard: the student is graded by the teachers' reward; a cfg that lost
# the block would train it against the old reward under this arm's name.
if ! grep -qE '^\\s*motionScaleReward:' "$CFG_ENV" || ! grep -qE '^\\s*referenceStd:\\s*{MS_REFERENCE_STD}\\s*$' "$CFG_ENV"; then
    echo "[student] ERROR: motion-scale student without motionScaleReward.referenceStd {MS_REFERENCE_STD} in $CFG_ENV" >&2; exit 1
fi
'''


def rewrite_launcher(text):
    shebang = text.splitlines(keepends=True)[0]
    sbatch = "".join(l for l in text.splitlines(keepends=True) if l.startswith("#SBATCH"))
    body = strip_header(text)[len(shebang):]
    sbatch = sbatch.replace(SHORT_BASE, SHORT_ARM)
    if f"--job-name=\"stu-{SHORT_ARM}\"" not in sbatch or "#SBATCH --mem=480G\n" not in sbatch:
        raise SystemExit("ERROR: #SBATCH block did not take the arm name / 480G")
    body = body.replace(SHORT_BASE, SHORT_ARM)                  # CFG paths, echoes, ckpt dir
    body = sub1(r"^for b in sub10 sub13 sub16; do$", "for b in sub4 sub10 sub13 sub16; do", body, "fold guard")
    # teacher-set guard: objects, not sources
    body = sub1(r"^# Teacher-set guard: teachers\.yaml must exist and cover all 13 sources, or the$",
                "# Teacher-set guard: teachers.yaml must exist, be OBJECT-keyed, and cover all 13 objects, or the",
                body, "teacher guard comment")
    body = sub1(r"^# task refuses at startup anyway -- say so here, before a 480G job is scheduled\.$",
                "# task refuses at startup anyway -- say so here, before a 480G job is scheduled.\n"
                "# (one manifest routes one way: a per-source g3 manifest here would be a wrong-teacher run)",
                body, "teacher guard comment 2")
    body = sub1(r"run scripts/collect_g3_teachers\.py \(see header\)",
                "run scripts/collect_g3_teachers.py --gen4-objects (see header)", body, "collect hint")
    body = sub1(r"import yaml,sys\nt=yaml\.safe_load\(open\('\$TP/teachers\.yaml'\)\)\['teachers'\]\n"
                r"have=\{s for e in t for s in e\['sources'\]\}\n"
                r"need=\{1,2,3,5,6,7,8,9,11,12,14,15,17\}\n"
                r"miss=sorted\(need-have\)\n"
                r"print\('\[student\] teachers:', len\(t\), 'files;', 'missing sources:', miss or 'none'\)\n"
                r"sys\.exit\(1 if miss else 0\)\"; then\n"
                r"    echo \"\[student\] ERROR: \$TP/teachers\.yaml does not cover every OMOMO source\" >&2; exit 1\nfi",
                GUARD_TEACHERS_PY, body, "teacher-set guard body")
    # gen4 + motion-scale guards, right after the merged-tree guard loop (which defines $RT)
    anchor = "merge_retarget_trees.py command in the srcall13 launcher header"
    i = body.find(anchor)
    if i < 0:
        raise SystemExit("ERROR: merged-tree guard not found in the base launcher")
    j = body.find("\ndone\n", i) + len("\ndone\n")
    body = body[:j] + GUARD_GEN4 + body[j:]
    body = sub1(r"--task InterMimicDistillG3 --cfg_env", "--task InterMimicDistillG4 --cfg_env", body, "task in echo")
    body = sub1(r"^    --task InterMimicDistillG3 \\$", "    --task InterMimicDistillG4 \\", body, "task in invocation")
    body = sub1(r'^echo "\[student\] OMOMO XF/RET/NVADLR student: 13 g3 teachers -> 6-token transformer \(9594\), '
                r'normval \+ adaptive LR \(KL 0\.06\), RAGGED, num_envs=\$NUM_ENVS"$',
                f'echo "[student] GEN4 OBJECT-GROUP student: 4 gen4 msexp teachers ({" ".join(GROUPS)}, latest, routed by '
                f'OBJECT via InterMimicDistillG4) -> 6-token transformer (9594), normval + adaptive LR (KL 0.06), '
                f'RAGGED, NO POSE TERM, MOTION-SCALE REWARD (ref {MS_REFERENCE_STD}), 43 gen4 bodies, num_envs=$NUM_ENVS"',
                body, "recipe echo")
    if "InterMimicDistillG3" in "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#")):
        raise SystemExit("ERROR: a non-comment reference to the g3 task survived")
    return shebang + sbatch + SH_HEADER + body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="cfg_diff the env cfg vs the g3 student and vs the chairs teacher")
    a = ap.parse_args()
    env_src, tr_src, sh_src = CFG / f"omomo_{BASE}.yaml", CFG / "train/rlg" / f"omomo_{BASE}.yaml", REPO / f"slurm_{BASE}.sh"
    for p in (env_src, tr_src, sh_src, CFG / f"omomo_teacher_{TEACHER_ARM}.yaml", CFG / "train/rlg" / f"omomo_teacher_{TEACHER_ARM}.yaml"):
        if not p.is_file():
            raise SystemExit(f"ERROR: base file missing: {p}")
    env_out, tr_out, sh_out = CFG / f"omomo_{ARM}.yaml", CFG / "train/rlg" / f"omomo_{ARM}.yaml", REPO / f"slurm_{ARM}.sh"
    env_out.write_text(rewrite_env(strip_header(env_src.read_text())))
    tr_out.write_text(rewrite_train(strip_header(tr_src.read_text())))
    sh_out.write_text(rewrite_launcher(sh_src.read_text()))
    sh_out.chmod(0o755)
    print(f"{ARM}:\n  {env_out.relative_to(REPO)}\n  {tr_out.relative_to(REPO)}\n  {sh_out.relative_to(REPO)}")
    if a.diff:
        diff = REPO / "scripts/cfg_diff.py"
        print("\n== env cfg vs the g3 OMOMO XF student ==")
        subprocess.run([sys.executable, str(diff), str(env_src), str(env_out)], check=False)
        print("\n== env cfg vs the chairs msexp TEACHER (expect only the student keys + dataObjects) ==")
        subprocess.run([sys.executable, str(diff), str(CFG / f"omomo_teacher_{TEACHER_ARM}.yaml"), str(env_out)], check=False)


if __name__ == "__main__":
    main()

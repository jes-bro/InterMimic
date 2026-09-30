#!/usr/bin/env python3
"""gen4 teacher fleet: the 13 OMOMO per-source transformer+nvadlr+nopose teachers
rebuilt on the gen4 bodies. Each arm is a COPY of its g3 nopose twin
(omomo_teacher_g3_omomo_geoall_src{S}_xf_nvadlr_nopose__f0) with exactly these
edits, nothing else (tests/test_gen4_teacher_cfgs.py pins it):

  env cfg   subjectHeightsFile   -> scripts/synthetic_heights_gen4.json
            (no betas_file: the g3 recipe has no betas conditioning, so none is needed)
            subjectBodies        -> the 13 fold-0 training reals + sub600..sub629
                                    (test bodies sub4 sub10 sub13 sub16 never train)
            humanoidAssetSuffix  -> _inertial   (NEW key: load smplx_omomo_<sub>_inertial.xml)
  train cfg full_experiment_name -> smplx_teacher_gen4_omomo_geoall_src{S}_xf_nvadlr_nopose__f0
  launcher  names/paths          -> the gen4 arm; HELDOUT hint gains sub4; two guards
                                    (inertial MJCF + retargeted refs present for every body)

Why gen4: (1) the old synthetic bodies (sub100-239) were selected with knowledge
of the test set and are retired; sub600-629 are sampled from the SMPL-X shape
prior with no subject involved. (2) sub4 is a valid subject: its "sim-crasher"
reputation was a PhysX link-frame rollover in the stock MJCF that also hits
sub11, sub606, sub607 and sub619; the *_inertial files fix all of them with
value-identical physics (47/47 clean, 2026-09-29). sub4 joins the held-out
set (Jess, 2026-09-30) so the 13 real training bodies stay the g3 roster.

Prerequisites on the cluster BEFORE launching (the launcher guards fail loudly):
  - inertial MJCFs:   python3 scripts/mjcf_add_inertials.py --all
  - retargeted refs for the new bodies, per source (one CPU job per source):
      SOURCE=sub{S} TARGETS="sub4 $(seq -f sub%g 600 629 | tr '\\n' ' ')" \\
        OUT=InterAct/OMOMO_retarget_contact_src{S} sbatch slurm_retarget_gen.sh
    (sub4 is included so it can be EVALUATED on retargeted arms later.)

  python3 scripts/make_gen4_teacher_cfgs.py            # writes 39 files
  python3 scripts/make_gen4_teacher_cfgs.py --diff     # + cfg_diff of each env cfg vs its base
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
SOURCES = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]
TRAIN_REALS = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
               "sub11", "sub12", "sub14", "sub15", "sub17"]
GEN4 = [f"sub{i}" for i in range(600, 630)]
TEST_BODIES = ["sub4", "sub10", "sub13", "sub16"]
HEIGHTS = "scripts/synthetic_heights_gen4.json"
SUFFIX = "_inertial"


def g3(s):
    return f"g3_omomo_geoall_src{s}_xf_nvadlr_nopose__f0"


def gen4(s):
    return f"gen4_omomo_geoall_src{s}_xf_nvadlr_nopose__f0"


def strip_header(text):
    """Drop the leading '#' comment block of a yaml/sh file (keeps a shebang)."""
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    if lines and lines[0].startswith("#!"):
        out.append(lines[0]); i = 1
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(out) + "".join(lines[i:])


def env_header(s):
    return (f"# gen4 TEACHER {gen4(s)}: the g3 nopose transformer teacher of OMOMO source sub{s}\n"
            f"# ({g3(s)}) on the gen4 bodies. This file is that arm's env cfg with\n"
            f"# exactly three edits (python3 scripts/cfg_diff.py the two files; pinned by\n"
            f"# tests/test_gen4_teacher_cfgs.py). No betas_file: the recipe has no betas conditioning.\n"
            f"#   subjectHeightsFile  -> {HEIGHTS}\n"
            f"#   subjectBodies       -> 13 fold-0 training reals + sub600..sub629 (43 bodies;\n"
            f"#                          held out: sub4 sub10 sub13 sub16 -- sub4 rejoined the test set)\n"
            f"#   humanoidAssetSuffix -> {SUFFIX}  (smplx_omomo_<sub>_inertial.xml: explicit link inertials;\n"
            f"#                          the stock files trip a PhysX joint-angle rollover on ~1 body in 10)\n"
            f"# Recipe otherwise unchanged: geometric_all reward, no pose term, 6 obs horizons,\n"
            f"# retargeted references (retargetedMotionDir must hold sub600-629 -- see\n"
            f"# scripts/make_gen4_teacher_cfgs.py for the rt-gen command), PSI 3, body-normalised reward.\n")


def train_header(s):
    return (f"# gen4 TEACHER train cfg for {gen4(s)}: byte-identical to the g3 nopose twin's\n"
            f"# train cfg (train/rlg/omomo_teacher_{g3(s)}.yaml) except full_experiment_name\n"
            f"# (own checkpoint dir). Transformer 6 tokens, normalize_value, adaptive LR exact-KL 0.06.\n")


def sh_header(s):
    return (f"# gen4 TEACHER launcher for {gen4(s)}: the g3 nopose launcher\n"
            f"# (slurm_teacher_{g3(s)}.sh) with the names/paths swapped to this arm, the\n"
            f"# eval hint holding out sub4 as well, and two guards (every body's *_inertial\n"
            f"# MJCF and retargeted reference tree must exist). Nothing else differs.\n"
            f"# Eval when done (hand-write omomo_eval_{gen4(s)}.yaml first, mirroring\n"
            f"# humanoidAssetSuffix):  HELDOUT=\"sub4 sub10 sub13 sub16\" sh scripts/eval_one.sh {gen4(s)}\n")


def rewrite_env(text):
    if re.search(r"^\s*betas_file:", text, flags=re.M):
        raise SystemExit("ERROR: base has betas_file -- this recipe is nobetas; check the base")
    text = re.sub(r"^(\s*)subjectHeightsFile:.*$", rf"\1subjectHeightsFile: {HEIGHTS}", text, count=1, flags=re.M)
    # subjectBodies block: replace the '- subN' items that follow the key
    m = re.search(r"^(\s*)subjectBodies:\s*\n((?:\s*- sub\d+\s*\n)+)", text, flags=re.M)
    if not m:
        raise SystemExit("ERROR: subjectBodies block not found")
    indent = m.group(1)
    block = f"{indent}subjectBodies:\n" + "".join(f"{indent}- {b}\n" for b in TRAIN_REALS + GEN4)
    text = text[:m.start()] + block + text[m.end():]
    # new key, kept in alphabetical position (before hybridInitProb / initRootHeight)
    if "humanoidAssetSuffix" in text:
        raise SystemExit("ERROR: base already has humanoidAssetSuffix")
    m = re.search(r"^(\s*)(hybridInitProb|initRootHeight|initVel|keyBodies):", text, flags=re.M)
    text = text[:m.start()] + f"{m.group(1)}humanoidAssetSuffix: {SUFFIX}\n" + text[m.start():]
    return text


def rewrite_train(text, s):
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{g3(s)}\s*$",
                     rf"\1smplx_teacher_{gen4(s)}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: full_experiment_name for src{s} not found exactly once")
    return new


GUARD = '''# gen4 guards: every body must have its *_inertial MJCF and its retargeted
# reference tree, or the run would silently load stock files / miss bodies.
RT_DIR=$(grep -E '^\\s*retargetedMotionDir:' "$CFG_ENV" | awk '{print $2}')
if ! grep -qE '^\\s*humanoidAssetSuffix:\\s*_inertial\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: gen4 arm without humanoidAssetSuffix: _inertial in $CFG_ENV" >&2; exit 1
fi
for b in $(grep -E '^\\s*- sub[0-9]+\\s*$' "$CFG_ENV" | awk '{print $2}'); do
    f="isaacgym/src/intermimic/data/assets/smplx/smplx_omomo_${b}_inertial.xml"
    [ -f "$f" ] || { echo "[teacher] ERROR: missing $f (run scripts/mjcf_add_inertials.py --all)" >&2; exit 1; }
    [ -d "$RT_DIR/$b" ] || { echo "[teacher] ERROR: no retargeted refs $RT_DIR/$b (run slurm_retarget_gen.sh for this source)" >&2; exit 1; }
done
'''


def rename_sh(text, s):
    text = text.replace(g3(s), gen4(s))          # also covers smplx_teacher_<arm> and cfg paths
    return text.replace('HELDOUT="sub10 sub13 sub16"', 'HELDOUT="sub4 sub10 sub13 sub16"')


def insert_guard(body):
    """gen4 guards go right after the existing retargetedMotionDir guard block."""
    anchor = "retarget arm without retargetedMotionDir"
    i = body.find(anchor)
    if i < 0:
        raise SystemExit("ERROR: retargetedMotionDir guard not found in launcher")
    j = body.find("\nfi\n", i) + len("\nfi\n")
    return body[:j] + GUARD + body[j:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="print cfg_diff of each env cfg vs its g3 base")
    a = ap.parse_args()
    for s in SOURCES:
        env_src = CFG / f"omomo_teacher_{g3(s)}.yaml"
        tr_src = CFG / "train/rlg" / f"omomo_teacher_{g3(s)}.yaml"
        sh_src = REPO / f"slurm_teacher_{g3(s)}.sh"
        for p in (env_src, tr_src, sh_src):
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p}")
        env_out = CFG / f"omomo_teacher_{gen4(s)}.yaml"
        tr_out = CFG / "train/rlg" / f"omomo_teacher_{gen4(s)}.yaml"
        sh_out = REPO / f"slurm_teacher_{gen4(s)}.sh"
        env_out.write_text(env_header(s) + rewrite_env(strip_header(env_src.read_text())))
        tr_out.write_text(train_header(s) + rewrite_train(strip_header(tr_src.read_text()), s))
        sh_text = sh_src.read_text()
        shebang = sh_text.splitlines(keepends=True)[0]
        body = strip_header(sh_text)[len(shebang):]
        # keep the #SBATCH lines (strip_header removes them as comments) -> re-extract
        sbatch = "".join(l for l in sh_text.splitlines(keepends=True) if l.startswith("#SBATCH"))
        sh_out.write_text(shebang + rename_sh(sbatch, s) + sh_header(s) + insert_guard(rename_sh(body, s)))
        sh_out.chmod(0o755)
        print(f"src{s:<2d}: {env_out.name}  {tr_out.name}  {sh_out.name}")
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {3 * len(SOURCES)} files for {len(SOURCES)} gen4 teacher arms")


if __name__ == "__main__":
    main()

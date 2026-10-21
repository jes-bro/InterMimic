#!/usr/bin/env python3
"""FROM-SCRATCH teacher arms with the foot weighting: every _msexp_fullep (or _flight_fullep)
teacher arm with rewardTerms.footWeight {weight 5} switched on, tagged _msexpfootw_fullep
(or _flightfootw_fullep). Jess 2026-10-20: "msexp from scratch with the foot weighting" --
the from-scratch twin of fine-tune variant D (and B), for a fleet that does not depend on
fine-tuning a trained teacher.

Each arm is DERIVED by text from its source twin: the arm renamed, the footWeight block
inserted under rewardTerms (key order: footWeight < freeFlightGate), a launcher guard for the
block, the recipe echo extended. Nothing else differs; the generator REFUSES an env cfg that
differs from its twin by anything but the two footWeight keys. 21 arms per source set.

  python3 scripts/make_gen4_footw_teacher_cfgs.py                 # --from msexp: the _msexpfootw_fullep arms
  python3 scripts/make_gen4_footw_teacher_cfgs.py --from flight   # the _flightfootw_fullep arms
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

FOOT_WEIGHT = 5.0            # default; --weight W writes the _<src>footwW_ arms (the tag carries W unless W == 5)
SOURCES = {"msexp": ("_msexp_", "_msexpfootw_"), "flight": ("_flight_", "_flightfootw_")}


def set_weight(w):
    """Pick the weight: the tag is footw for 5 (the first set), footw<W> otherwise (footw2, footw3)."""
    global FOOT_WEIGHT, SOURCES, GUARD
    FOOT_WEIGHT = float(w)
    tag = "footw" if FOOT_WEIGHT == 5.0 else "footw" + (str(int(w)) if float(w).is_integer() else str(w).replace(".", "p"))
    SOURCES = {"msexp": ("_msexp_", f"_msexp{tag}_"), "flight": ("_flight_", f"_flight{tag}_")}
    GUARD = guard_text()


def bases(src):
    return list(mkf.BASES) if src == "msexp" else [mkf.arm_name(b, "flight") for b in mkf.BASES]


def arm_name(base, src):
    frm, to = SOURCES[src]
    assert base.count(frm) == 1, base
    return base.replace(frm, to)


def foot_block(ind):
    return f"{ind}footWeight:\n{ind}  enable: true\n{ind}  weight: {FOOT_WEIGHT}\n"


def guard_text():
    return (f"# Foot-weight guard: this arm trains with the feet weighted x{FOOT_WEIGHT} in the position / rotation terms\n"
            f"# (utils/foot_terms.py); a cfg that lost the block would train the SRC arm under this arm's name.\n"
            f"if ! grep -qE '^\\s*footWeight:' \"$CFG_ENV\" || ! grep -qE '^\\s*weight:\\s*{FOOT_WEIGHT}\\s*$' \"$CFG_ENV\"; then\n"
            f"    echo \"[teacher] ERROR: footWeight arm without rewardTerms.footWeight weight {FOOT_WEIGHT} in $CFG_ENV\" >&2; exit 1\n"
            f"fi\n")


GUARD = guard_text()


def note(base, src, what):
    return (f"# FOOT-WEIGHT twin (from scratch): DERIVED from the {base} {what} by scripts/make_gen4_footw_teacher_cfgs.py\n"
            f"# -- the arm renamed and rewardTerms.footWeight switched on (weight {FOOT_WEIGHT}: the ankles' and toes' error\n"
            f"# counts x{FOOT_WEIGHT} inside the position and rotation terms, utils/foot_terms.py). Nothing else differs\n"
            f"# (pinned by tests/test_gen4_footw_teacher_cfgs.py). Never edit as-run cfgs.\n")


def derive_env(text, base, src):
    m = re.search(r"^(?P<ind>[ \t]*)rewardTerms:\n(?P<ind2>[ \t]+)(freeFlightGate|pose):", text, re.M)
    if not m:
        raise SystemExit(f"ERROR: {base}: rewardTerms block not found")
    if re.search(r"^\s*footWeight:", text, re.M):
        raise SystemExit(f"ERROR: {base}: already has footWeight")
    at = m.start("ind2")
    text = text[:at] + foot_block(m.group("ind2")) + text[at:]
    return note(base, src, "env cfg") + text.replace(base, arm_name(base, src))


def derive_train(text, base, src):
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{re.escape(base)}\s*$",
                     lambda m: m.group(1) + f"smplx_teacher_{arm_name(base, src)}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: {base}: full_experiment_name not found exactly once in the train cfg")
    return note(base, src, "train cfg") + new.replace(base, arm_name(base, src))


def derive_launcher(text, base, src):
    anchor = re.compile(r"^# Motion-scale guard:.*?\nfi\n" if src == "msexp" else r"^# Object-flight guard:.*?\nfi\n", re.M | re.S)
    m = anchor.search(text)
    if not m:
        raise SystemExit(f"ERROR: {base}: reward guard not found in the launcher")
    text = text[:m.end()] + GUARD.replace("SRC", src) + text[m.end():]
    text = text.replace(base, arm_name(base, src))
    for old in ("MOTION-SCALE EXPONENT", "OBJECT-FLIGHT EXPONENT"):
        text = text.replace(old, f"FOOT WEIGHT x{FOOT_WEIGHT} + {old}")
    lines = text.splitlines(keepends=True)
    last = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    return "".join(lines[:last + 1]) + note(base, src, "launcher") + "".join(lines[last + 1:])


def verify(base, src, env_base, env_out):
    a, b = mkf.flatten(yaml.safe_load(open(env_base))), mkf.flatten(yaml.safe_load(open(env_out)))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    want = {"env.rewardTerms.footWeight.enable", "env.rewardTerms.footWeight.weight"}
    if diff != want:
        raise SystemExit(f"ERROR: {arm_name(base, src)}: env cfg differs from its twin by {sorted(diff)}, expected {sorted(want)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", choices=sorted(SOURCES), default="msexp")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--weight", type=float, default=5.0, help="footWeight.weight; 5 -> the footw arms, W -> the footw<W> arms")
    a = ap.parse_args()
    set_weight(a.weight)
    n = 0
    for base in bases(a.src):
        env_src, tr_src, sh_src = CFG / f"omomo_teacher_{base}.yaml", CFG / "train/rlg" / f"omomo_teacher_{base}.yaml", REPO / f"slurm_teacher_{base}.sh"
        for p in (env_src, tr_src, sh_src):
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p}")
        arm = arm_name(base, a.src)
        env_out, tr_out, sh_out = CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml", REPO / f"slurm_teacher_{arm}.sh"
        env_out.write_text(derive_env(env_src.read_text(), base, a.src))
        tr_out.write_text(derive_train(tr_src.read_text(), base, a.src))
        sh_out.write_text(derive_launcher(sh_src.read_text(), base, a.src))
        sh_out.chmod(0o755)
        verify(base, a.src, env_src, env_out)
        n += 3
        print(arm)
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {n} files for {n // 3} foot-weight teacher arms (from {a.src})")


if __name__ == "__main__":
    main()

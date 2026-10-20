#!/usr/bin/env python3
"""The gen4 FOOT-SKATE teacher arms: every _flight2_fullep teacher arm with the
rewardTerms.footSkate factor switched on (utils/foot_skate_reward.py), tagged
_flight2foot. One change off the flight2 twin, so the foot term is attributable.

Jess, 2026-10-18: "why don't we try a foot / leg term" -- every OMOMO policy slides its
planted feet (renders 2026-10-16/18; the full-episode window did not fix it). The
object-flight exponent (flight2) restores the strict product on OMOMO; this arm adds,
on top of it, a direct penalty on a planted foot's horizontal speed:

    rewardTerms:
      footSkate: {enable: true, lambda: 2.0, bodies: [L_Toe, R_Toe], contactThreshold: 1.0}

Each arm is DERIVED from its _flight2_fullep twin by text: the three files copied with
the arm renamed (flight2 -> flight2foot), the footSkate block inserted under
rewardTerms (key order: before freeFlightGate), a launcher guard for the block, and the
recipe echo extended. The generator REFUSES an env cfg that differs from its twin by
anything but the four footSkate keys (yaml-level diff). 21 arms (the flight2 bases),
63 files. The chairs arm is the probe; the rest are ready for the fleet.

  python3 scripts/make_gen4_foot_teacher_cfgs.py            # writes 63 files
  python3 scripts/make_gen4_foot_teacher_cfgs.py --diff     # + cfg_diff of each env cfg vs its flight2 twin
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
_flight = importlib.util.spec_from_file_location("mkf", REPO / "scripts/make_gen4_flight_teacher_cfgs.py")
mkf = importlib.util.module_from_spec(_flight); _flight.loader.exec_module(mkf)

FOOT = {"lambda": 2.0, "bodies": ["L_Toe", "R_Toe"], "contactThreshold": 1.0}
# --from flight2 (default): the _flight2foot arms; --from flight: the _flightfoot arms
# (Jess 2026-10-20), the same block on the one-constant object-flight arms.
SOURCES = {"flight2": ("_flight2_", "_flight2foot_"), "flight": ("_flight_", "_flightfoot_")}
SRC = "flight2"


def bases(src=None):
    """The 21 _<src>_fullep arms this generator twins."""
    return [mkf.arm_name(b, src or SRC) for b in mkf.BASES]


BASES = bases("flight2")


def arm_name(base):
    for frm, to in SOURCES.values():
        if base.count(frm) == 1:
            return base.replace(frm, to)
    raise AssertionError(base)


def foot_block(ind):
    return (f"{ind}footSkate:\n{ind}  enable: true\n{ind}  lambda: {FOOT['lambda']}\n"
            f"{ind}  bodies: [{', '.join(FOOT['bodies'])}]\n{ind}  contactThreshold: {FOOT['contactThreshold']}\n")


GUARD = (f"# Foot-skate guard: this arm trains with the planted-foot speed penalty; a cfg that\n"
         f"# lost the block would train the SRCTAG arm under this arm's name.\n"
         f"if ! grep -qE '^\\s*footSkate:' \"$CFG_ENV\" || ! grep -qE '^\\s*lambda:\\s*{FOOT['lambda']}\\s*$' \"$CFG_ENV\"; then\n"
         f"    echo \"[teacher] ERROR: foot-skate arm without rewardTerms.footSkate lambda {FOOT['lambda']} in $CFG_ENV\" >&2; exit 1\n"
         f"fi\n")


def note(base, what):
    return (f"# FOOT-SKATE twin: DERIVED from the {base} {what} by scripts/make_gen4_foot_teacher_cfgs.py\n"
            f"# -- the arm renamed and rewardTerms.footSkate switched on (lambda {FOOT['lambda']}, bodies "
            f"{' '.join(FOOT['bodies'])}, contact {FOOT['contactThreshold']} N;\n"
            f"# utils/foot_skate_reward.py: exp(-lambda * mean |v_xy|^2 of a foot while planted), one more factor\n"
            f"# under the root). Nothing else differs (pinned by tests/test_gen4_foot_teacher_cfgs.py). Never edit as-run cfgs.\n")


def derive_env(text, base):
    m = re.search(r"^(?P<ind>[ \t]*)rewardTerms:\n(?P<ind2>[ \t]+)(freeFlightGate|pose):", text, re.M)
    if not m:
        raise SystemExit(f"ERROR: {base}: rewardTerms block not found")
    if re.search(r"^\s*footSkate:", text, re.M):
        raise SystemExit(f"ERROR: {base}: already has footSkate")
    # 'footSkate' sorts before 'freeFlightGate' and 'pose': first child of rewardTerms
    insert_at = m.start("ind2")
    text = text[:insert_at] + foot_block(m.group("ind2")) + text[insert_at:]
    text = text.replace(base, arm_name(base))
    return note(base, "env cfg") + text


def derive_train(text, base):
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{re.escape(base)}\s*$",
                     lambda m: m.group(1) + f"smplx_teacher_{arm_name(base)}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: {base}: full_experiment_name not found exactly once in the train cfg")
    return note(base, "train cfg") + new.replace(base, arm_name(base))


def derive_launcher(text, base):
    # the guard goes right after the object-flight guard's 'fi'
    m = re.search(r"^# Object-flight guard:.*?\nfi\n", text, re.M | re.S)
    if not m:
        raise SystemExit(f"ERROR: {base}: object-flight guard not found in the launcher")
    src = next(k for k, (frm, _) in SOURCES.items() if base.count(frm) == 1)
    text = text[:m.end()] + GUARD.replace("SRCTAG", src) + text[m.end():]
    text = text.replace(base, arm_name(base))
    text = text.replace("OBJECT-FLIGHT EXPONENT", f"FOOT-SKATE PENALTY (lambda {FOOT['lambda']}) + OBJECT-FLIGHT EXPONENT")
    lines = text.splitlines(keepends=True)
    last = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    return "".join(lines[:last + 1]) + note(base, "launcher") + "".join(lines[last + 1:])


def verify(base, env_base, env_out):
    a, b = mkf.flatten(yaml.safe_load(open(env_base))), mkf.flatten(yaml.safe_load(open(env_out)))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    want = {"env.rewardTerms.footSkate.enable", "env.rewardTerms.footSkate.lambda",
            "env.rewardTerms.footSkate.bodies", "env.rewardTerms.footSkate.contactThreshold"}
    if diff != want:
        raise SystemExit(f"ERROR: {arm_name(base)}: env cfg differs from its flight2 twin by {sorted(diff)}, expected {sorted(want)}")
    e = yaml.safe_load(open(env_out))["env"]
    if e["rewardTerms"]["footSkate"] != dict(enable=True, **FOOT):
        raise SystemExit(f"ERROR: {arm_name(base)}: bad footSkate block {e['rewardTerms']['footSkate']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="cfg_diff each env cfg vs its flight2 twin")
    ap.add_argument("--from", dest="src", choices=sorted(SOURCES), default="flight2",
                    help="which object-flight arms to twin: flight2 -> _flight2foot (default), flight -> _flightfoot")
    a = ap.parse_args()
    n = 0
    for base in bases(a.src):
        env_src, tr_src, sh_src = CFG / f"omomo_teacher_{base}.yaml", CFG / "train/rlg" / f"omomo_teacher_{base}.yaml", REPO / f"slurm_teacher_{base}.sh"
        for p in (env_src, tr_src, sh_src):
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p} (run make_gen4_flight_teacher_cfgs.py first)")
        arm = arm_name(base)
        env_out, tr_out, sh_out = CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml", REPO / f"slurm_teacher_{arm}.sh"
        env_out.write_text(derive_env(env_src.read_text(), base))
        tr_out.write_text(derive_train(tr_src.read_text(), base))
        sh_out.write_text(derive_launcher(sh_src.read_text(), base))
        sh_out.chmod(0o755)
        verify(base, env_src, env_out)
        n += 3
        print(arm)
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {n} files for {n // 3} foot-skate teacher arms")


if __name__ == "__main__":
    main()

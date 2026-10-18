#!/usr/bin/env python3
"""The gen4 OBJECT-FLIGHT teacher arms: every _msexp_fullep teacher arm with its reward
exponent rule swapped from motionScaleReward to objectFlightReward, in TWO variants.

Jess, 2026-10-17: the per-clip reward exponent must be a function of the clip, with
OMOMO and soccer graded at (or near) the product and basketball at the 4th root.
utils/object_flight.py is that function (the clip's object flight: airborne share x
mean airborne height, metres). Two settings of its one free parameter are trained
side by side, every teacher in both ("we will train all of the teachers the same way"):

    _flight   objectFlightReward {rootFlight 0.6, flatFlight 0.0}   one constant: the
              exponent falls from the first metre of flight (OMOMO product, soccer
              median 0.89, basketball at the root)
    _flight2  objectFlightReward {rootFlight 0.6, flatFlight 0.3}   flat at product up
              to 0.3 m (OMOMO and ~90% of soccer at exactly product), then the same slope

Each arm is DERIVED from its _msexp_fullep twin by text: the three files are copied
with the arm renamed (msexp -> flight / flight2), the motionScaleReward block replaced
by the objectFlightReward block in the env cfg, the launcher's motion-scale guard
replaced by the object-flight guard, and the reward notes in headers and echoes
reworded. Nothing else changes, and the generator REFUSES an output whose env cfg
differs from its twin by anything but those two blocks (a yaml-level diff). Full
episodes only (Jess 2026-10-17): the bases are the 21 _msexp_fullep arms --
the 4 OMOMO object groups, the generalist, the 13 single-object arms, bball7,
soccer15, act22 -- so 42 arms, 126 files.

  python3 scripts/make_gen4_flight_teacher_cfgs.py            # writes 126 files
  python3 scripts/make_gen4_flight_teacher_cfgs.py --diff     # + cfg_diff of each env cfg vs its msexp twin
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"

ROOT_FLIGHT = 0.6                          # metres of flight at which the 1/N root is reached
VARIANTS = {"flight": 0.0, "flight2": 0.3}  # tag -> flatFlight
OMOMO_OBJECTS = ["clothesstand", "floorlamp", "largebox", "largetable", "monitor", "plasticbox",
                 "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "whitechair", "woodchair"]
BASES = ([f"gen4_omomo_geoall_{g}_xf_nvadlr_nopose_msexp_fullep__f0" for g in ("boxes", "stands", "chairs", "tables", "all")]
         + [f"gen4_omomo_geoall_{o}_xf_nvadlr_nopose_msexp_fullep__f0" for o in OMOMO_OBJECTS]
         + [f"gen4_{a}_geoall_xf_nvadlr_nopose_msexp_fullep__f0" for a in ("bball7", "soccer15", "act22")])

MS_BLOCK = re.compile(r"^(?P<ind>[ \t]*)motionScaleReward:\n(?P=ind)[ \t]+enable: true\n(?P=ind)[ \t]+referenceStd: 0\.5157\n", re.M)
MS_GUARD = re.compile(
    r"^# Motion-scale guard: this arm trains on the per-clip reward exponent; a cfg that\n"
    r"# lost the block would train the plain (?:group arm|recipe) under this arm's name\.\n"
    r"if ! grep -qE '\^\\s\*motionScaleReward:' \"\$CFG_ENV\" \|\| ! grep -qE '\^\\s\*referenceStd:\\s\*0\.5157\\s\*\$' \"\$CFG_ENV\"; then\n"
    r"    echo \"\[teacher\] ERROR: motion-scale arm without motionScaleReward\.referenceStd 0\.5157 in \$CFG_ENV\" >&2; exit 1\n"
    r"fi\n", re.M)


def arm_name(base, tag):
    assert base.count("_msexp_") == 1, base
    return base.replace("_msexp_", f"_{tag}_")


def flight_block(ind, flat):
    return (f"{ind}objectFlightReward:\n{ind}  enable: true\n{ind}  rootFlight: {ROOT_FLIGHT}\n"
            f"{ind}  flatFlight: {flat}\n")


def flight_guard(flat):
    return (f"# Object-flight guard: this arm trains on the per-clip reward exponent from the clip's\n"
            f"# object flight; a cfg that lost the block, or carries the other variant's flatFlight,\n"
            f"# would train a different arm under this arm's name.\n"
            f"if ! grep -qE '^\\s*objectFlightReward:' \"$CFG_ENV\" || ! grep -qE '^\\s*rootFlight:\\s*{ROOT_FLIGHT}\\s*$' \"$CFG_ENV\" "
            f"|| ! grep -qE '^\\s*flatFlight:\\s*{flat}\\s*$' \"$CFG_ENV\"; then\n"
            f"    echo \"[teacher] ERROR: object-flight arm without objectFlightReward rootFlight {ROOT_FLIGHT} + flatFlight {flat} in $CFG_ENV\" >&2; exit 1\n"
            f"fi\n")


def derived_note(base, tag, flat, what):
    return (f"# OBJECT-FLIGHT twin ({tag}): DERIVED from the {base} {what} by\n"
            f"# scripts/make_gen4_flight_teacher_cfgs.py -- the arm renamed and the reward exponent rule swapped from\n"
            f"# motionScaleReward (per-clip key-body std) to objectFlightReward {{rootFlight {ROOT_FLIGHT}, flatFlight {flat}}}\n"
            f"# (per-clip object flight = airborne share x mean airborne height, utils/object_flight.py). Nothing\n"
            f"# else differs (pinned by tests/test_gen4_flight_teacher_cfgs.py). Never edit as-run cfgs.\n")


def reword(text, flat):
    """The reward notes that name the old rule, in comments and echoes."""
    text = text.replace("MOTION-SCALE EXPONENT (ref 0.5157, capped at product)",
                        f"OBJECT-FLIGHT EXPONENT (root {ROOT_FLIGHT} m, flat {flat} m)")
    text = text.replace("MOTION-SCALE EXPONENT ref 0.5157 capped at product",         # the EgoExo launchers' wording
                        f"OBJECT-FLIGHT EXPONENT root {ROOT_FLIGHT} m flat {flat} m")
    text = text.replace("reward exponent per clip from its key-body std (utils/motion_scale.py)",
                        "reward exponent per clip from its object flight (utils/object_flight.py)")
    text = text.replace("motion-scale reward exponent", "object-flight reward exponent")
    text = text.replace("motion-scale reward", "object-flight reward")
    text = text.replace("motionScaleReward, referenceStd 0.5157", f"objectFlightReward, rootFlight {ROOT_FLIGHT} flatFlight {flat}")
    text = text.replace("motionScaleReward", "objectFlightReward")
    text = text.replace("utils/motion_scale.py", "utils/object_flight.py")
    return text


def derive_env(text, base, tag, flat):
    m = MS_BLOCK.search(text)
    if not m or MS_BLOCK.search(text, m.end()):
        raise SystemExit(f"ERROR: {base}: motionScaleReward block not found exactly once")
    text = text[:m.start()] + text[m.end():]
    # objectFlightReward sorts after objectDensity and before objectMass / objectPropsFile / obsHorizons
    anchor = re.search(r"^(?P<ind>[ \t]*)(objectMass|objectPropsFile|obsHorizons):", text, re.M)
    if not anchor:
        raise SystemExit(f"ERROR: {base}: no anchor key for objectFlightReward")
    text = text[:anchor.start()] + flight_block(anchor.group("ind"), flat) + text[anchor.start():]
    text = text.replace(base, arm_name(base, tag))
    text = reword(text, flat)
    return derived_note(base, tag, flat, "env cfg") + text


def derive_train(text, base, tag, flat):
    new, n = re.subn(rf"^(\s*full_experiment_name:\s*)smplx_teacher_{re.escape(base)}\s*$",
                     lambda m: m.group(1) + f"smplx_teacher_{arm_name(base, tag)}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"ERROR: {base}: full_experiment_name not found exactly once in the train cfg")
    new = new.replace(base, arm_name(base, tag))
    return derived_note(base, tag, flat, "train cfg") + new


def derive_launcher(text, base, tag, flat):
    m = MS_GUARD.search(text)
    if not m or MS_GUARD.search(text, m.end()):
        raise SystemExit(f"ERROR: {base}: motion-scale guard not found exactly once in the launcher")
    text = text[:m.start()] + flight_guard(flat) + text[m.end():]
    text = text.replace(base, arm_name(base, tag))
    text = reword(text, flat)
    lines = text.splitlines(keepends=True)
    # the derivation note goes right after the #SBATCH block
    last = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    return "".join(lines[:last + 1]) + derived_note(base, tag, flat, "launcher") + "".join(lines[last + 1:])


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def verify(base, tag, flat, env_base, env_out):
    """Refuse unless the env cfgs differ by exactly the two reward blocks."""
    a, b = flatten(yaml.safe_load(open(env_base))), flatten(yaml.safe_load(open(env_out)))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    want = {"env.motionScaleReward.enable", "env.motionScaleReward.referenceStd",
            "env.objectFlightReward.enable", "env.objectFlightReward.rootFlight", "env.objectFlightReward.flatFlight"}
    if diff != want:
        raise SystemExit(f"ERROR: {arm_name(base, tag)}: env cfg differs from its msexp twin by {sorted(diff)}, expected {sorted(want)}")
    e = yaml.safe_load(open(env_out))["env"]
    if e["objectFlightReward"] != {"enable": True, "rootFlight": ROOT_FLIGHT, "flatFlight": flat} or "motionScaleReward" in e:
        raise SystemExit(f"ERROR: {arm_name(base, tag)}: bad reward blocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="cfg_diff each env cfg vs its msexp fullep twin")
    a = ap.parse_args()
    n = 0
    for base in BASES:
        env_src, tr_src, sh_src = CFG / f"omomo_teacher_{base}.yaml", CFG / "train/rlg" / f"omomo_teacher_{base}.yaml", REPO / f"slurm_teacher_{base}.sh"
        for p in (env_src, tr_src, sh_src):
            if not p.is_file():
                raise SystemExit(f"ERROR: base file missing: {p}")
        for tag, flat in VARIANTS.items():
            arm = arm_name(base, tag)
            env_out, tr_out, sh_out = CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml", REPO / f"slurm_teacher_{arm}.sh"
            env_out.write_text(derive_env(env_src.read_text(), base, tag, flat))
            tr_out.write_text(derive_train(tr_src.read_text(), base, tag, flat))
            sh_out.write_text(derive_launcher(sh_src.read_text(), base, tag, flat))
            sh_out.chmod(0o755)
            verify(base, tag, flat, env_src, env_out)
            n += 3
            print(f"{arm}")
            if a.diff:
                subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(env_src), str(env_out)], check=False)
    print(f"wrote {n} files for {n // 3} object-flight teacher arms ({len(BASES)} bases x {len(VARIANTS)} variants)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""InterMimic BASELINE teachers: upstream's own per-subject recipe, plus our free-flight gate.

Jess, 2026-10-20: the reviewer-proof baseline is InterMimic as published, on our data, changed
in exactly the one thing we relax -- the contact-gated termination -- so nobody can credit our
numbers to looser resets. The paper (Sec. "Imitation as Retargeting", supp. Sec. F): teacher
policies are tailored to each human subject, all sharing one base human model; on OMOMO
"Subject 9 is the base model, with teacher policies retargeting interactions from other
subjects into this base". No kinematic retargeting of the inputs -- MEASURED 2026-10-20: every
subject's clips fit only its own skeleton (sub9's clip is 4 cm off on sub2's body), and the
shipped body file omomo.xml is sub9's skeleton to 0.000 cm over all 52 bodies.

Per training source (13 arms, imbase_src<N>_ffg__f0), DERIVED by text from upstream's own recipe
files (checked in unchanged: cfg/omomo_upstream_train_env.yaml = commit 56790b1's
omomo_train.yaml; train/rlg/omomo_upstream_train.yaml = the original release's omomo.yaml):
  env     dataSub -> [sub<N>]; motion_file -> InterAct/OMOMO_new (the data our arms use: same
          body/object tracks as upstream's OMOMO to < 0.2 cm, hands fixed);
          + rewardTerms.freeFlightGate {resets: true, reward: false}   (OUR GATE, the one change)
          Everything else upstream's: robotType smplx/omomo.xml (NO subjectBodies -- upstream's
          exact body file, sub9's skeleton with upstream's joint defaults), Hybrid 0.1, PSI 3,
          300-frame window, product reward at the paper's weights (30, 2.5, 5, 0.1, 5, 5, 5, 3,
          2e-5, 2e-5, 1e-9), 4096 envs, no retargeting, no body normalisation, no pose term.
          Termination thresholds are the code defaults (= ours).
  train   full_experiment_name -> smplx_teacher_<arm>; resume_from -> None (upstream resumes from
          a shared init checkpoint we do not have; the arms train from scratch). MLP
          1024-1024-512, constant LR 2e-5, no value normalisation, horizon 32, minibatch 16384.
  launch  a minimal launcher (guards: one source, the gate block, no subjectBodies / retarget keys)
  eval    upstream's env with the eval-owned keys only; NO subjectBodies: these teachers are
          evaluated on the base body only -- render with BODY=generic (slurm_render_policy.sh).
The generator REFUSES an env cfg that differs from upstream's by anything but those keys.

  python3 scripts/make_intermimic_baseline_cfgs.py            # 13 arms, 52 files
  python3 scripts/make_intermimic_baseline_cfgs.py --diff
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
ENV_BASE = CFG / "omomo_upstream_train_env.yaml"
TRAIN_BASE = CFG / "train/rlg/omomo_upstream_train.yaml"
SOURCES = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]        # the 13 training sources (held out: 4 10 13 16)
MOTION = "InterAct/OMOMO_new"
GATE = {"resets": True, "reward": False}
EVAL_OWNED = {"numEnvs": 1024, "stateInit": '"Start"', "enableEvaluation": "True", "rolloutLength": 1000, "physicalBufferSize": 1}


def arm_name(n):
    return f"imbase_src{n}_ffg__f0"


def strip_header(text):
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(lines[i:])


def sub1(pat, rep, text, what, flags=re.M):
    new, n = re.subn(pat, lambda m: rep(m) if callable(rep) else rep, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"ERROR: {what}: expected exactly one match, got {n}")
    return new


def note(n, what):
    return (f"# INTERMIMIC BASELINE {arm_name(n)} ({what}): upstream's per-subject recipe on source sub{n} + OUR\n"
            f"# free-flight gate (rewardTerms.freeFlightGate resets true / reward false), nothing else. DERIVED from\n"
            f"# {ENV_BASE.name} / {TRAIN_BASE.name} by scripts/make_intermimic_baseline_cfgs.py; the base body is\n"
            f"# upstream's own file smplx/omomo.xml (= Subject 9, the paper's base model). Pinned by\n"
            f"# tests/test_intermimic_baseline_cfgs.py. Never edit as-run cfgs.\n")


def derive_env(text, n):
    text = sub1(r"^(\s*dataSub:\s*)\['sub2'\]\s*$", lambda m: f"{m.group(1)}['sub{n}']", text, "dataSub")
    text = sub1(r"^(\s*motion_file:\s*)InterAct/OMOMO\s*$", lambda m: f"{m.group(1)}{MOTION}", text, "motion_file")
    if re.search(r"^\s*rewardTerms:", text, re.M):
        raise SystemExit("ERROR: upstream env already has rewardTerms")
    m = re.search(r"^(\s*)rewardWeights:", text, re.M)       # 'rewardTerms' sorts before 'rewardWeights'
    if not m:
        raise SystemExit("ERROR: rewardWeights not found (anchor for rewardTerms)")
    ind = m.group(1)
    blk = (f"{ind}rewardTerms:\n{ind}  freeFlightGate:\n{ind}    resets: {str(GATE['resets']).lower()}\n"
           f"{ind}    reward: {str(GATE['reward']).lower()}\n")
    text = text[:m.start()] + blk + text[m.start():]
    return note(n, "env cfg") + text


def derive_train(text, n):
    text = sub1(r"^(\s*full_experiment_name:\s*)init\s*$", lambda m: f"{m.group(1)}smplx_teacher_{arm_name(n)}", text, "full_experiment_name")
    text = sub1(r"^(\s*resume_from:\s*)omomo/init/nn/mimic\.pth\s*$", lambda m: f"{m.group(1)}None", text, "resume_from")
    return note(n, "train cfg") + text


def derive_eval(text, n):
    body = derive_env(text, n)
    body = strip_header(body)
    for key, val in EVAL_OWNED.items():
        if re.search(rf"^\s*{key}:", body, re.M):
            body = sub1(rf"^(\s*{key}:\s*).*$", lambda m, v=val: f"{m.group(1)}{v}", body, f"eval-owned {key}")
        else:   # upstream's env has no enableEvaluation key: insert it in key order, after enableEarlyTermination
            body = sub1(r"^(\s*)enableEarlyTermination:.*$", lambda m, k=key, v=val: f"{m.group(0)}\n{m.group(1)}{k}: {v}",
                        body, f"anchor for eval-owned {key}")
    hdr = (f"# EVAL config -- {arm_name(n)}: the InterMimic baseline teacher for source sub{n} (upstream's recipe +\n"
           f"# our free-flight gate). HAND-DERIVED mirror of omomo_teacher_{arm_name(n)}.yaml with only the eval-owned\n"
           f"# keys changed (numEnvs, stateInit, enableEvaluation, rolloutLength, physicalBufferSize) -- check_eval_cfg.py\n"
           f"# proves it. NO subjectBodies: this teacher knows ONE body, upstream's smplx/omomo.xml (Subject 9), so it is\n"
           f"# scored on that body only: render with BODY=generic SOURCE=sub{n} (slurm_render_policy.sh). The dataSub\n"
           f"# placeholder is its own source; a per-pair eval of a baseline is the (generic, sub{n}) pair only.\n"
           f"evalFor:\n  - {arm_name(n)}\n\n")
    return hdr + body


LAUNCHER = """#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="tch-ARM"
#SBATCH --output=teacher-ARM-%j.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL
NOTE#
# Upstream's recipe trains at 4096 envs; override per submission: NUM_ENVS=2048 sbatch ...
source ~/.bashrc
conda deactivate
conda activate "${INTERMIMIC_ENV:-intermimic-gym2}"   # another machine: INTERMIMIC_ENV=<its env name>
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="isaacgym/src:.${PYTHONPATH:+:$PYTHONPATH}"
export REWARD_BREAKDOWN=1
export REWARD_BREAKDOWN_EVERY=1000
export TERM_REASON=1
export TERM_REASON_EVERY=2000
NUM_ENVS="${NUM_ENVS:-4096}"
CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_ARM.yaml
CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_ARM.yaml

# Baseline guards: ONE source, upstream's base body (no subjectBodies), no retargeting, and
# the gate block present -- a cfg that lost any of these would train something else under
# this arm's name.
if ! grep -qE "^\\s*dataSub:\\s*\\['subN'\\]\\s*$" "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm must train exactly source subN (dataSub) in $CFG_ENV" >&2; exit 1
fi
if grep -qE '^\\s*(subjectBodies|retargetedMotionDir|humanoidAssetSuffix|motionScaleReward|objectFlightReward|rewardShape):' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm carries a non-upstream key (subjectBodies/retarget/exponent/shape) in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\\s*robotType:\\s*"smplx/omomo\\.xml"\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm must use upstream's body file smplx/omomo.xml in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\\s*freeFlightGate:' "$CFG_ENV" || ! grep -qE '^\\s*resets:\\s*true\\s*$' "$CFG_ENV"; then
    echo "[teacher] ERROR: baseline arm without rewardTerms.freeFlightGate.resets true (OUR gate) in $CFG_ENV" >&2; exit 1
fi
if ! grep -qE '^\\s*name:\\s*intermimic\\s*$' "$CFG_TRAIN" || grep -qE '^\\s*name:\\s*intermimic_transformer' "$CFG_TRAIN"; then
    echo "[teacher] ERROR: baseline arm must use upstream's MLP network in $CFG_TRAIN" >&2; exit 1
fi
[ -d "InterAct/OMOMO_new" ] || { echo "[teacher] ERROR: InterAct/OMOMO_new not found (run from the repo root)" >&2; exit 1; }
[ -f "isaacgym/src/intermimic/data/assets/smplx/omomo.xml" ] || { echo "[teacher] ERROR: upstream body file smplx/omomo.xml missing" >&2; exit 1; }

echo "[teacher] invocation: python -u -m intermimic.run --task InterMimic --cfg_env $CFG_ENV --cfg_train $CFG_TRAIN --num_envs $NUM_ENVS --headless --output checkpoints  (slurm=$0 job=$SLURM_JOB_ID)"
echo "[teacher] INTERMIMIC BASELINE ARM (upstream per-subject recipe on source subN: base body smplx/omomo.xml = Subject 9, raw clips from InterAct/OMOMO_new, Hybrid 0.1 + PSI 3 at a 300-frame window, product reward at the paper's weights, MLP) + OUR free-flight gate (resets on no-contact frames OFF). num_envs=$NUM_ENVS"
echo "[teacher] host=$(hostname) job=$SLURM_JOB_ID -> checkpoints/smplx_teacher_ARM/nn/"

# --- auto-resume: continue from the latest checkpoint if one exists. ---
EXP=$(grep -oE 'full_experiment_name:[[:space:]]*[^[:space:]]+' "$CFG_TRAIN" | awk '{print $2}')
CKPT="checkpoints/${EXP}/nn/mimic.pth"
if [ -f "$CKPT" ]; then
    RESUME_TRAIN="/tmp/${EXP}_resume_${SLURM_JOB_ID:-$$}.yaml"
    sed -E "s|^(\\s*resume_from:)\\s*'?None'?\\s*$|\\1 '${CKPT}'|" "$CFG_TRAIN" > "$RESUME_TRAIN"
    if ! grep -qF "resume_from: '${CKPT}'" "$RESUME_TRAIN"; then
        echo "[teacher] ERROR: could not rewrite resume_from in $CFG_TRAIN -- refusing to" \\
             "start fresh over ${CKPT}" >&2; exit 1
    fi
    CFG_TRAIN="$RESUME_TRAIN"
    echo "[teacher] RESUMING from ${CKPT}"
else
    echo "[teacher] fresh start (no checkpoint at ${CKPT})"
fi

python -u -m intermimic.run \\
    --task InterMimic \\
    --cfg_env "$CFG_ENV" \\
    --cfg_train "$CFG_TRAIN" \\
    --num_envs "$NUM_ENVS" \\
    --headless \\
    --output checkpoints
"""


def derive_launcher(n):
    return LAUNCHER.replace("NOTE#\n", note(n, "launcher")).replace("ARM", arm_name(n)).replace("subN", f"sub{n}")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def verify(n, env_out):
    a, b = flatten(yaml.safe_load(open(ENV_BASE))), flatten(yaml.safe_load(open(env_out)))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    want = {"env.motion_file", "env.rewardTerms.freeFlightGate.resets", "env.rewardTerms.freeFlightGate.reward"} | ({"env.dataSub"} if n != 2 else set())
    if diff != want:
        raise SystemExit(f"ERROR: {arm_name(n)}: env cfg differs from upstream's by {sorted(diff)}, expected {sorted(want)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="cfg_diff each env cfg vs upstream's recipe")
    a = ap.parse_args()
    for p in (ENV_BASE, TRAIN_BASE):
        if not p.is_file():
            raise SystemExit(f"ERROR: base file missing: {p}")
    env_src, tr_src = strip_header(ENV_BASE.read_text()), strip_header(TRAIN_BASE.read_text())
    k = 0
    for n in SOURCES:
        arm = arm_name(n)
        env_out, tr_out = CFG / f"omomo_teacher_{arm}.yaml", CFG / "train/rlg" / f"omomo_teacher_{arm}.yaml"
        sh_out, ev_out = REPO / f"slurm_teacher_{arm}.sh", CFG / f"omomo_eval_{arm}.yaml"
        env_out.write_text(derive_env(env_src, n))
        tr_out.write_text(derive_train(tr_src, n))
        sh_out.write_text(derive_launcher(n)); sh_out.chmod(0o755)
        ev_out.write_text(derive_eval(env_src, n))
        verify(n, env_out)
        k += 4
        print(arm)
        if a.diff:
            subprocess.run([sys.executable, str(REPO / "scripts/cfg_diff.py"), str(ENV_BASE), str(env_out)], check=False)
    print(f"wrote {k} files for {k // 4} InterMimic baseline arms")


if __name__ == "__main__":
    main()

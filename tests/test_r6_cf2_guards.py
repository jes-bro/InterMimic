#!/usr/bin/env python3
"""Pin the bball-r6_cf2 arm: its slurm guards, and the one-knob claim.

r6_cf2 is r5_roll50 trained on RELABELLED contact data. The chain that justifies
it is measured, not hypothesised:

  1. r3's eval, split by reference contact state, showed ro FLAT across free/held
     (0.538 / 0.580) -- killing the free-flight-gating hypothesis -- while rcg
     collapsed to 0.141 on held frames.
  2. rcg_hand is pinned to exactly 1.0 wherever the reference says no hand
     contact, so only the held row measures grip at all.
  3. inspect_bball_clip section 8 found 21 of 53 claimed-contact frames (40%)
     with NO hand body touching the ball -- rcg_hand unearnable there for ANY
     policy -- because relabel_contact_flags.py rewrote contact_obj only and
     rcg_hand grades contact_human.
  4. _cf2 fixes that: worst gap +0.187 -> +0.012 m, channel disagreement
     15/101 -> 0/101, positions byte-identical.

The arm's value is that motion_file is the ONLY difference from r5. Two ways
that rots, both covered:
  A. the cfgs drift apart (a second knob sneaks in)   -> test_one_knob
  B. the guards fail to catch the inherited _cf path  -> test_guards

The _cf-vs-_cf2 prefix trap gets its own case: '_cf' is a prefix of '_cf2', so a
loose grep would accept the wrong dataset and silently duplicate r5.

Run:  python tests/test_r6_cf2_guards.py   (exit 0 = all green)
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGDIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
SCRIPT = os.path.join(REPO, "slurm_cari4d_bball_r6_cf2.sh")
CFG = os.path.join(CFGDIR, "omomo_cari4d_bball_r6_cf2_train.yaml")
CFG_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r6_cf2_eval.yaml")
RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r6_cf2_train.yaml")
R5 = os.path.join(CFGDIR, "omomo_cari4d_bball_r5_roll50_train.yaml")
R5_EVAL = os.path.join(CFGDIR, "omomo_cari4d_bball_r5_roll50_eval.yaml")
R5_RLG = os.path.join(CFGDIR, "train/rlg/omomo_cari4d_bball_r5_roll50_train.yaml")

CF2 = "InterAct/behave_cari4d_optj3d_cf2"

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def flat(d, p=""):
    if not isinstance(d, dict):
        return {p: d}
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else str(k)
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def load(path):
    return flat(yaml.safe_load(open(path)) or {})


def extract_guard_block():
    lines = open(SCRIPT).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Guard:"))
    end = next(i for i, l in enumerate(lines) if "invocation:" in l)
    block = "\n".join(lines[start:end])
    assert "motion_file" in block, "guard block did not survive extraction"
    return block


def run_guards(cfg_path, block, cwd=None):
    script = 'CFG_ENV="$1"\n' + block + "\nexit 0\n"
    proc = subprocess.run(["bash", "-c", script, "bash", cfg_path],
                          capture_output=True, text=True, cwd=cwd or REPO)
    return proc.returncode, proc.stderr.strip()


def test_one_knob():
    print("1. one-knob claim vs r5_roll50 (parsed yaml):")
    train, r5 = load(CFG), load(R5)
    diffs = {k for k in set(r5) | set(train)
             if r5.get(k, "<absent>") != train.get(k, "<absent>")}
    check("train cfg differs from r5 ONLY in env.motion_file",
          diffs == {"env.motion_file"}, f"(differs in: {sorted(diffs)})")
    check("train motion_file is the relabelled dir",
          train.get("env.motion_file") == CF2,
          f"(got {train.get('env.motion_file')})")

    # The eval twin MUST move with the train cfg here -- unlike the rest of the
    # ladder. An arm trained on corrected labels graded against stale ones is
    # penalised for exactly the frames the relabel fixed.
    ev, r5e = load(CFG_EVAL), load(R5_EVAL)
    ev_diffs = {k for k in set(r5e) | set(ev)
                if r5e.get(k, "<absent>") != ev.get(k, "<absent>")}
    check("eval cfg differs from r5's eval ONLY in env.motion_file",
          ev_diffs == {"env.motion_file"}, f"(differs in: {sorted(ev_diffs)})")
    check("eval reads the SAME relabelled data as train",
          ev.get("env.motion_file") == train.get("env.motion_file"),
          f"(train {train.get('env.motion_file')} vs eval {ev.get('env.motion_file')})")
    check("eval still keeps rolloutLength 300 (full-clip success)",
          ev.get("env.rolloutLength") == 300)

    # Coverage must stay at r5's value or the relabel is confounded with it.
    check("train rolloutLength stays 50 (not confounded with coverage)",
          train.get("env.rolloutLength") == 50,
          f"(got {train.get('env.rolloutLength')})")
    check("no physicalBufferSize key (PSI stays gated off)",
          "env.physicalBufferSize" not in train)

    rlg, r5rlg = load(RLG), load(R5_RLG)
    check("rlg full_experiment_name is smplx_cari4d_bball_r6_cf2",
          rlg.get("params.config.full_experiment_name") == "smplx_cari4d_bball_r6_cf2",
          f"(got {rlg.get('params.config.full_experiment_name')})")
    rlg_diffs = {k for k in set(rlg) | set(r5rlg) if rlg.get(k) != r5rlg.get(k)}
    check("rlg differs from r5's ONLY in full_experiment_name",
          rlg_diffs == {"params.config.full_experiment_name"},
          f"(differs in: {sorted(rlg_diffs)})")
    check("keeps the sub2 teacher warm start",
          rlg.get("params.config.resume_from")
          == "checkpoints/smplx_teachers_new/sub2.pth")


def test_guards():
    print("\n2. slurm guards -- positive control:")
    block = extract_guard_block()
    # The data-exists guard needs the dir; the dataset is cluster-only, so run
    # from a temp root that has an empty stand-in. This keeps the OTHER guards
    # honest without pretending the real data is here.
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, CF2))
        code, err = run_guards(CFG, block, cwd=tmp)
        check("committed r6 cfg passes all guards", code == 0, f"(exit {code}: {err})")

        print("\n3. the missing-dataset guard fires when _cf2 is absent:")
        # _cf2 is produced by relabel_contact_human.py and is not in git, so a
        # fresh clone must fail loudly rather than deep inside Isaac Gym.
        with tempfile.TemporaryDirectory() as bare:
            code, err = run_guards(CFG, block, cwd=bare)
            check("absent dataset is refused", code != 0, f"(exit {code})")
            check("the error names the build command",
                  "relabel_contact_human.py" in err, f"(said: {err[:200]})")

        print("\n4. slurm guards -- each sabotage must be REFUSED:")
        mutations = [
            # THE prefix trap: _cf is a prefix of _cf2, so a loose grep would
            # accept r5's dataset and silently duplicate that run.
            ("motion_reverted_to_cf",
             r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d_optj3d_cf"),
            ("motion_reverted_to_plain",
             r"^  motion_file: InterAct/behave_cari4d_optj3d_cf2.*$",
             "  motion_file: InterAct/behave_cari4d"),
            # Coverage drift would confound the relabel with a second knob.
            ("rollout_drifted_to_30", r"^  rolloutLength: 50.*$",
             "  rolloutLength: 30"),
            ("stateinit_start", r'^  stateInit: "Hybrid".*$', '  stateInit: "Start"'),
            ("psi_added", r"^  hybridInitProb: 0\.1.*$",
             "  hybridInitProb: 0.1\n  physicalBufferSize: 3"),
            ("human_reset_off", r"^    human: 0\.5.*$", "    human: false"),
            ("object_reset_on", r"^    object: false.*$", "    object: 0.3"),
            ("igratio_reset_on", r"^    igRatio: false.*$", "    igRatio: 0.5"),
            ("resetthresholds_removed", r"^  resetThresholds:.*$", "  # removed"),
        ]
        src = open(CFG).read()
        for label, pattern, replacement in mutations:
            out, n = re.subn(pattern, replacement, src, count=1, flags=re.MULTILINE)
            assert n == 1, f"mutation '{label}' matched {n} -- fixture stale"
            path = os.path.join(tmp, f"mutant_{label}.yaml")
            open(path, "w").write(out)
            code, _ = run_guards(path, block, cwd=tmp)
            check(f"{label} is refused", code != 0,
                  "(guard did NOT fire -- it is a no-op for this knob)")

        print("\n5. the sibling arms' cfgs must be refused:")
        for sibling in ("r5_roll50_train", "r3_roll30_train", "r4_human1m_train"):
            path = os.path.join(CFGDIR, f"omomo_cari4d_bball_{sibling}.yaml")
            if not os.path.exists(path):
                continue
            code, _ = run_guards(path, block, cwd=tmp)
            check(f"refuses {sibling}", code != 0,
                  "(guard accepted another arm's cfg)")


def test_relabel_provenance():
    """The arm is only meaningful if the dataset it names is the one the
    relabeller produces, from the source r5 uses."""
    print("\n6. provenance: the arm, the script, and r5 agree on the paths:")
    rel = open(os.path.join(REPO, "scripts/relabel_contact_human.py")).read()
    check("relabel script documents the _cf -> _cf2 build",
          "behave_cari4d_optj3d_cf2" in rel)
    r5_src = load(R5).get("env.motion_file")
    check("r5 reads the SOURCE the relabeller consumes",
          r5_src == "InterAct/behave_cari4d_optj3d_cf", f"(r5 reads {r5_src})")
    sh = open(SCRIPT).read()
    check("launcher's build hint uses r5's dir as --src-dir",
          f"--src-dir {r5_src}" in sh)
    check("launcher's build hint targets this arm's dir",
          f"--dst-dir {CF2}" in sh or "--dst-dir $MOTION_DIR" in sh)


def main():
    test_one_knob()
    test_guards()
    test_relabel_provenance()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

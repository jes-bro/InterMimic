#!/usr/bin/env python3
"""Guards for the per-arm eval configs and the machinery that resolves them.

These pin the invariants that, when they broke, produced numbers that looked fine:
a policy scored against a reference it never trained on, a no-betas arm scored
with betas, a whole generation of arms unscoreable, and every CSV recording the
first progress snapshot instead of the result.

    python3 -m pytest tests/test_eval_cfgs.py -q
"""
import glob
import os
import subprocess
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")
sys.path.insert(0, os.path.join(REPO, "scripts"))

import check_eval_cfg as cec              # noqa: E402
from eval_per_pair import parse_metrics   # noqa: E402


# --------------------------------------------------------------------------
# 1. Every committed config parses.
#
# Not a formality. generate_synladder_cfgs.py:55 line-replaced `subjectBodies:`
# with a flow-style list and left the parent's block-style items dangling under
# it, committing three g2 configs that fail yaml.safe_load outright -- arms that
# could never launch. The identical defect lived in eval_per_pair.make_temp_yaml
# and in the render/replay scripts' sed, which is why none of them could be
# pointed at a per-arm config. Nothing catches this except parsing the files.
# --------------------------------------------------------------------------
ALL_CFGS = sorted(glob.glob(os.path.join(CFG, "*.yaml")) +
                  glob.glob(os.path.join(CFG, "train/rlg/*.yaml")))


@pytest.mark.parametrize("path", ALL_CFGS, ids=lambda p: os.path.basename(p))
def test_every_committed_config_parses(path):
    with open(path) as fh:
        assert yaml.safe_load(fh) is not None, "parsed to nothing"


# --------------------------------------------------------------------------
# 2. Every eval config still mirrors the arm(s) it claims to serve.
# --------------------------------------------------------------------------
EVAL_PAIRS = [(p, arm) for p, arms in cec.eval_cfgs().items() for arm in arms]


@pytest.mark.parametrize("path,arm", EVAL_PAIRS,
                         ids=[f"{os.path.basename(p)}::{a}" for p, a in EVAL_PAIRS])
def test_eval_cfg_mirrors_its_arm(path, arm):
    problems = cec.check(path, cec.train_cfg_for(arm), arm)
    assert not problems, "\n".join(problems)


def test_check_all_exits_clean():
    r = subprocess.run([sys.executable, "scripts/check_eval_cfg.py", "--check-all"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------
# 3. Resolution is total and unambiguous for the arms we intend to score.
# --------------------------------------------------------------------------
def _g23_arms():
    out = []
    for p in sorted(glob.glob(os.path.join(CFG, "omomo_teacher_g[23]_*.yaml"))):
        out.append(os.path.basename(p)[len("omomo_teacher_"):-len(".yaml")])
    return out


# The syn-ladder arms train on a different body roster AND a different betas file
# (omomo_betas_neutral_aug2.npz), which changes the beta observation -- so they
# need their own eval configs and must NOT borrow the plain ret one. Listed here
# so "unserved" is a recorded decision rather than an oversight.
UNSERVED = {
    "g2_mlp_ret_stock_syn0__f0",
    "g2_mlp_ret_stock_syn60__f0",
    "g2_mlp_ret_stock_syn130__f0",
}


@pytest.mark.parametrize("arm", _g23_arms())
def test_every_arm_resolves_to_exactly_one_eval_cfg(arm):
    hits = [p for p, arms in cec.eval_cfgs().items() if arm in arms]
    if arm in UNSERVED:
        assert not hits, f"{arm} is listed UNSERVED but {hits} claims it"
        return
    assert len(hits) == 1, f"{arm} resolved to {len(hits)} eval cfgs: {hits}"


def test_unknown_arm_is_an_error_not_a_fallback():
    with pytest.raises(SystemExit):
        cec.resolve("g9_does_not_exist__f0")


def test_v1_configs_are_not_resolvable_as_arm_configs():
    """The renamed old template must never be handed to a gen-2/gen-3 arm."""
    served = [os.path.basename(p) for p in cec.eval_cfgs()]
    assert not [f for f in served if f.startswith(cec.V1_PREFIX)]


# --------------------------------------------------------------------------
# 3b. Scoring variants: the same checkpoint under a declared different rule.
#
# The one real case: the nogate arm scored under the base's termination rule
# (freeFlightGate.resets true). Mirroring `resets: false` into its eval lets
# the referee end every free-flight episode, which scores the rule, not the
# policy. The variant may differ from the arm ONLY in what its scoringVariant
# block declares, and that key must really differ -- anything else is drift.
# --------------------------------------------------------------------------
NOGATE = "g3_bball7_geoall_nogate__f0"
GATED = NOGATE + "+gatedscore"


def test_variant_id_splits_and_plain_id_does_not():
    assert cec.split_variant(GATED) == (NOGATE, "gatedscore")
    assert cec.split_variant(NOGATE) == (NOGATE, None)
    with pytest.raises(SystemExit):
        cec.split_variant("g3_x__f0+")


def test_gatedscore_resolves_separately_and_the_arm_still_resolves_to_its_mirror():
    mirror, variant = cec.resolve(NOGATE), cec.resolve(GATED)
    assert mirror != variant
    assert os.path.basename(variant) == "omomo_eval_g3_bball7_geoall_nogate_gatedscore__f0.yaml"
    assert os.path.basename(mirror) == "omomo_eval_g3_bball7_geoall_nogate__f0.yaml"


def test_gatedscore_is_the_mirror_plus_exactly_the_gate():
    mirror = cec.flatten(cec.load(cec.resolve(NOGATE))["env"])
    variant = cec.flatten(cec.load(cec.resolve(GATED))["env"])
    diff = {k for k in set(mirror) | set(variant) if mirror.get(k) != variant.get(k)}
    assert diff == {"rewardTerms.freeFlightGate.resets"}
    assert variant["rewardTerms.freeFlightGate.resets"] is True
    # ...and it is the BASE's rule, i.e. the same exam the base takes
    base = cec.flatten(cec.load(cec.resolve("g3_bball7_geoall__f0"))["env"])
    assert base["rewardTerms.freeFlightGate.resets"] is True
    assert not cec.check(cec.resolve(GATED), cec.train_cfg_for(GATED), GATED)


def test_variant_cfg_checked_as_plain_arm_is_rejected():
    """A variant must never pass as the arm's mirror (that is how it would get
    resolved for the arm by accident and score every arm under a tweaked rule)."""
    problems = cec.check(cec.resolve(GATED), cec.train_cfg_for(NOGATE), NOGATE)
    assert any("scoringVariant" in p for p in problems)
    assert any("freeFlightGate.resets" in p for p in problems)


def _write_variant(tmp_path, overrides_block, extra_env=""):
    """A throwaway variant cfg derived from the real gatedscore one."""
    src = open(cec.resolve(GATED)).read()
    cfg = yaml.safe_load(src)
    if overrides_block is not None:
        cfg["scoringVariant"]["overrides"] = overrides_block
    if extra_env:
        cfg["env"].update(extra_env)
    p = tmp_path / "omomo_eval_tmp_variant.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_variant_drift_on_an_undeclared_key_is_rejected(tmp_path):
    p = _write_variant(tmp_path, None, extra_env={"objectMass": 9.9})
    problems = cec.check(p, cec.train_cfg_for(GATED), GATED)
    assert any("objectMass" in q for q in problems)


def test_variant_override_that_does_not_differ_from_the_arm_is_rejected(tmp_path):
    # declare the override at the arm's own value: a no-op variant is a mislabelled duplicate
    p = _write_variant(tmp_path, {"rewardTerms.freeFlightGate.resets": False},
                       extra_env={"rewardTerms": cec.load(cec.train_cfg_for(NOGATE))["env"]["rewardTerms"]})
    problems = cec.check(p, cec.train_cfg_for(GATED), GATED)
    assert any("changes nothing" in q for q in problems)


def test_eval_one_routes_a_variant_id_to_its_cfg_and_suffixes_the_csv(tmp_path):
    """EMIT mode resolves the plan without submitting; the checkpoint only has to exist."""
    ck = tmp_path / "smplx_teacher_g3_bball7_geoall_nogate__f0" / "nn" / "mimic_00020000.pth"
    ck.parent.mkdir(parents=True); ck.write_bytes(b"")
    r = subprocess.run(["sh", "scripts/eval_one.sh", GATED, str(ck)], cwd=REPO,
                       env={**os.environ, "EMIT": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = dict(line.split("=", 1) for line in r.stdout.strip().splitlines())
    assert plan["ENV_YAML"].strip("'").endswith("omomo_eval_g3_bball7_geoall_nogate_gatedscore__f0.yaml")
    assert plan["OUT"].strip("'").endswith("__mimic_00020000__indist+heldout+syn__gatedscore.csv")
    assert plan["TRAIN_YAML"].strip("'").endswith("omomo_teacher_g3_bball7_geoall_nogate__f0.yaml")
    # the plain id still goes to the mirror and the un-suffixed CSV
    r = subprocess.run(["sh", "scripts/eval_one.sh", NOGATE, str(ck)], cwd=REPO,
                       env={**os.environ, "EMIT": "1"}, capture_output=True, text=True)
    plan = dict(line.split("=", 1) for line in r.stdout.strip().splitlines())
    assert plan["ENV_YAML"].strip("'").endswith("omomo_eval_g3_bball7_geoall_nogate__f0.yaml")
    assert plan["OUT"].strip("'").endswith("__mimic_00020000__indist+heldout+syn.csv")


# --------------------------------------------------------------------------
# 3c. g3 STUDENT evals: scored through the student path, mirrored to the
# student's own env cfg. A student's observation is built by
# InterMimicDistillG3 (its own horizons + Arm A's body dims) and handed to the
# network by the DAgger wrapper; the teacher path would feed it obs_buf.
# --------------------------------------------------------------------------
STUDENTS = ["student_g3_act_xf_ret_nvadlr__f0", "student_g3_act_xf_ret_nvadlr_bodyctr__f0",
            "student_g3_act_xf_ret_nvadlr_bodyctr_sync__f0"]


@pytest.mark.parametrize("arm", STUDENTS)
def test_student_eval_cfg_mirrors_the_student_and_uses_the_student_path(arm):
    path = cec.resolve(arm)
    assert os.path.basename(path) == f"omomo_eval_{arm}.yaml"
    assert cec.train_cfg_for(arm).endswith(f"omomo_{arm}.yaml")          # not omomo_teacher_
    assert not cec.check(path, cec.train_cfg_for(arm), arm)
    cfg = cec.load(path)
    assert cfg["evalEntry"] == "intermimic.run_distill"
    assert cfg["evalTask"] == "InterMimicDistillG3"
    env, train = cfg["env"], cec.load(cec.train_cfg_for(arm))["env"]
    assert env["rolloutLength"] == 700                                   # > cpr 691 / soccer 677
    assert env["numObsRetarget"] == train["numObsRetarget"]              # 9594 plain / 9750 Arm A
    assert env["teacherPolicy"] == train["teacherPolicy"]                # teachers load at eval too
    if "bodyctr" in arm:
        # body features are part of the student's obs, so they stay on; twins
        # only feed the contrastive TRAINING loss and cannot be constructed
        # with one body per eval pair (twin_partners refuses), so they are OFF
        assert env["studentBodyFeatures"] is True and env["twinEnvs"] is False
        assert env["numObsRetarget"] == 9594 + 156
    if "bodyctr_sync" in arm:
        assert env["twinCoReset"] is False and train["twinCoReset"] is True   # training-only, like twins


def test_student_launcher_and_log_naming():
    assert cec.launcher_for(STUDENTS[0]) == "slurm_student_g3_act_xf_ret_nvadlr__f0.sh"
    assert os.path.exists(os.path.join(REPO, cec.launcher_for(STUDENTS[0])))
    assert cec.log_glob_for(STUDENTS[0]) == "student-g3_act_xf_ret_nvadlr__f0-*.out"
    assert cec.launcher_for("g3_bball7_geoall__f0") == "slurm_teacher_g3_bball7_geoall__f0.sh"
    assert cec.log_glob_for("g3_bball7_geoall__f0") == "teacher-g3_bball7_geoall__f0-*.out"


def _emit(run, ck):
    r = subprocess.run(["sh", "scripts/eval_one.sh", run, str(ck)], cwd=REPO,
                       env={**os.environ, "EMIT": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return {k: v.strip("'") for k, v in (l.split("=", 1) for l in r.stdout.strip().splitlines())}


def test_eval_one_routes_a_student_through_run_distill(tmp_path):
    ck = tmp_path / "smplx_student_g3_act_xf_ret_nvadlr_bodyctr__f0" / "nn" / "mimic_00006000.pth"
    ck.parent.mkdir(parents=True); ck.write_bytes(b"")
    plan = _emit(STUDENTS[1], ck)
    assert plan["ENV_YAML"].endswith("omomo_eval_student_g3_act_xf_ret_nvadlr_bodyctr__f0.yaml")
    assert plan["TRAIN_YAML"].endswith("train/rlg/omomo_student_g3_act_xf_ret_nvadlr_bodyctr__f0.yaml")
    assert (plan["EVAL_ENTRY"], plan["EVAL_TASK"]) == ("intermimic.run_distill", "InterMimicDistillG3")
    assert plan["EXP"] == "smplx_student_g3_act_xf_ret_nvadlr_bodyctr__f0"
    assert plan["OUT"].endswith("smplx_student_g3_act_xf_ret_nvadlr_bodyctr__f0__mimic_00006000__indist+heldout+syn.csv")
    assert plan["BODIES"].split()[-8:-5] == ["sub10", "sub16", "sub13"] or "sub10" in plan["BODIES"]


def test_eval_one_keeps_teachers_on_the_teacher_path(tmp_path):
    ck = tmp_path / "smplx_teacher_g3_bball7_geoall__f0" / "nn" / "mimic_00020000.pth"
    ck.parent.mkdir(parents=True); ck.write_bytes(b"")
    plan = _emit("g3_bball7_geoall__f0", ck)
    assert (plan["EVAL_ENTRY"], plan["EVAL_TASK"]) == ("intermimic.run", "InterMimic")


def test_eval_per_pair_build_cmd_threads_entry_and_task():
    from eval_per_pair import build_cmd, DEFAULT_ENTRY, DEFAULT_TASK
    cmd = build_cmd("intermimic.run_distill", "InterMimicDistillG3", "e.yaml", "t.yaml",
                    "ck.pth", "sub10", "sub401")
    assert cmd[:5] == ["python", "-u", "-m", "intermimic.run_distill", "--task"]
    assert cmd[5] == "InterMimicDistillG3" and "--test" in cmd and "--num_envs" not in cmd
    assert cmd[cmd.index("--subject_bodies") + 1] == "sub10"
    assert cmd[cmd.index("--data_sub") + 1] == "sub401"
    dflt = build_cmd(DEFAULT_ENTRY, DEFAULT_TASK, "e.yaml", "t.yaml", "ck.pth", "sub2", "sub2", num_envs=512)
    assert dflt[3] == "intermimic.run" and dflt[5] == "InterMimic"
    assert dflt.count("--num_envs") == 1 and dflt[dflt.index("--num_envs") + 1] == "512"


# --------------------------------------------------------------------------
# 4. The settings that decide what a number MEANS.
# --------------------------------------------------------------------------
EVAL_CFGS = sorted(cec.eval_cfgs())


@pytest.mark.parametrize("path", EVAL_CFGS, ids=lambda p: os.path.basename(p))
def test_eval_cfg_can_actually_produce_metrics(path):
    env = cec.load(path)["env"]
    # intermimic.py:169 force-disables evaluation outside Start init.
    assert env.get("stateInit") == "Start"
    assert env.get("enableEvaluation") is True
    # subjectBodies must be a single placeholder: bodies round-robin across envs,
    # so a full roster would average over every body while the CSV names one.
    assert len(env.get("subjectBodies") or []) == 1


@pytest.mark.parametrize("path", EVAL_CFGS, ids=lambda p: os.path.basename(p))
def test_numobs_matches_its_own_horizons_and_betas(path):
    env = cec.load(path)["env"]
    arch, horizons, betas, want = cec.obs_width(env)
    assert env.get("numObs") == want, (
        f"numObs={env.get('numObs')} but arch={arch} horizons={horizons} "
        f"betas={bool(betas)} implies {want}")


def test_scoring_budget_is_uniform_across_eval_cfgs():
    """One exam for every arm: every eval cfg scores at the same numEnvs, and it
    is InterMimic's own eval-script value (isaacgym/scripts/eval_*.sh).

    numEnvs is concurrency, not the attempt budget -- the player runs 20,000
    episodes per pair regardless (intermimic_players.py:52-60,173,363,392) --
    but a value that differs between cfgs, or from upstream, is still drift.
    """
    seen = {cec.load(p)["env"].get("numEnvs") for p in EVAL_CFGS}
    assert seen == {cec.EVAL_NUM_ENVS}, f"eval cfgs numEnvs: {seen}, fleet value {cec.EVAL_NUM_ENVS}"
    for script in ("eval_teacher.sh", "eval_student.sh"):
        src = open(os.path.join(REPO, "isaacgym", "scripts", script)).read()
        assert f"--num_envs {cec.EVAL_NUM_ENVS}" in src, f"upstream {script} disagrees with EVAL_NUM_ENVS"


def test_no_arm_overrides_the_player_episode_budget():
    """The attempt budget is rl_games' games_num * n_game_life * 10 = 20,000
    episodes per pair (defaults 2000 / 1, rl_games 1.1.4 common/player.py:41-43).
    A train cfg with a player block would silently give its arm a different
    budget; check_budget refuses that, and --check-all must stay clean."""
    assert not cec.check_budget(), "\n".join(cec.check_budget())
    for p, arms in cec.eval_cfgs().items():
        for spec in arms:
            rlg = cec.train_rlg_for(spec)
            player = ((cec.load(rlg).get("params") or {}).get("config") or {}).get("player") or {}
            assert "games_num" not in player and "n_game_life" not in player, spec


@pytest.mark.parametrize("path", EVAL_CFGS, ids=lambda p: os.path.basename(p))
def test_rollout_window_can_reach_the_success_condition(path):
    """rolloutLength must exceed the clips, or success is impossible.

    humanoid.py:553 cuts the episode at rolloutLength-1; success is
    _max_execution_steps >= max_episode_length-1 (intermimic.py:1703). Inheriting
    an arm's training window (g3 trains at 50) reports 0% for every arm.
    """
    env = cec.load(path)["env"]
    train_rollout = cec.load(cec.train_cfg_for(cec.eval_cfgs()[path][0]))["env"]["rolloutLength"]
    assert env["rolloutLength"] > train_rollout, (
        "eval rollout window must be widened past the training window")
    assert env["rolloutLength"] >= 300


# --------------------------------------------------------------------------
# 5. parse_metrics reads the RESULT, not the first progress snapshot.
# --------------------------------------------------------------------------
_PROGRESS = """
EVALUATION METRICS:
  Average Execution Steps: 42.00
  Average Human Pose Error: 0.3000
  Average Object Pose Error: 0.4000
  Success Rate: 11.00% (3/27)
EVALUATION METRICS:
  Average Execution Steps: 198.00
  Average Human Pose Error: 0.0900
  Average Object Pose Error: 0.1100
  Success Rate: 74.00% (20/27)
"""
_FINAL = """
FINAL EVALUATION SUMMARY:
  Sequences Evaluated: 27/27 (100.0%)
  Average Execution Steps: 203.00
  Average Human Pose Error: 0.0850
  Average Object Pose Error: 0.1050
  Success Rate: 81.00% (22/27)
"""


def test_parse_metrics_prefers_the_final_summary():
    m = parse_metrics(_PROGRESS + _FINAL)
    assert (m["success_rate"], m["avg_steps"]) == (81.0, 203.0)


def test_parse_metrics_falls_back_to_the_last_progress_block_on_timeout():
    m = parse_metrics(_PROGRESS)
    assert (m["success_rate"], m["avg_steps"]) == (74.0, 198.0)


def test_parse_metrics_never_mixes_two_blocks():
    m = parse_metrics(_PROGRESS)
    assert (m["avg_steps"], m["human_pose_error"], m["object_pose_error"]) == \
           (198.0, 0.09, 0.11)


def test_parse_metrics_returns_none_when_absent():
    assert parse_metrics("policy crashed, no metrics") is None


# --------------------------------------------------------------------------
# 6. The per-pair keys travel as CLI overrides, not as a rewritten file.
# --------------------------------------------------------------------------
def test_config_py_applies_the_per_pair_overrides():
    """config.py must set subjectBodies/dataSub/dataObjects from the CLI.

    Asserted against the source because config.py imports isaacgym, which is not
    importable outside the cluster conda env.
    """
    src = open(os.path.join(REPO, "isaacgym/src/intermimic/utils/config.py")).read()
    for flag, key in [("--subject_bodies", "subjectBodies"),
                      ("--data_sub", "dataSub"),
                      ("--data_objects", "dataObjects")]:
        assert flag in src, f"{flag} not declared"
        assert f'cfg["env"]["{key}"]' in src, f"{flag} never applied to {key}"


def test_eval_per_pair_no_longer_rewrites_configs():
    src = open(os.path.join(REPO, "scripts/eval_per_pair.py")).read()
    assert "make_temp_yaml" not in src
    assert "tempfile" not in src
    assert "--subject_bodies" in src and "--data_sub" in src


@pytest.mark.parametrize("script", [
    "slurm_render_policy.sh", "slurm_replay.sh", "slurm_replay_xbody.sh",
])
def test_render_and_replay_do_not_patch_configs_by_sed(script):
    """A render of the wrong environment is a convincing, wrong video."""
    src = open(os.path.join(REPO, script)).read()
    assert "s|dataSub:" not in src, "still sed-patching the env config"
    assert "s|subjectBodies:" not in src

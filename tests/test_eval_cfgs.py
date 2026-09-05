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
    problems = cec.check(path, cec.train_cfg_for(arm))
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
    """success is the best attempt per CLIP, so numEnvs biases the score upward.

    Two arms scored at different budgets are not comparable and nothing in the
    CSV would show it.
    """
    seen = {cec.load(p)["env"].get("numEnvs") for p in EVAL_CFGS}
    assert len(seen) == 1, f"eval cfgs disagree on numEnvs: {seen}"


@pytest.mark.parametrize("path", EVAL_CFGS, ids=lambda p: os.path.basename(p))
def test_rollout_window_can_reach_the_success_condition(path):
    """rolloutLength must exceed the clips, or success is impossible.

    humanoid.py:553 cuts the episode at rolloutLength-1; success is
    _max_execution_steps >= max_episode_length-1 (intermimic.py:1703). Inheriting
    an arm's training window (g3 trains at 50) reports 0% for every arm.
    """
    env = cec.load(path)["env"]
    arm_cfg = os.path.join(CFG, f"omomo_teacher_{cec.eval_cfgs()[path][0]}.yaml")
    train_rollout = cec.load(arm_cfg)["env"]["rolloutLength"]
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

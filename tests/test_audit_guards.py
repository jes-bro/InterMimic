#!/usr/bin/env python3
"""Prove the no-silent-fallback guards fire ONLY on genuinely-broken configs --
never on the real configs our training/eval jobs actually use.

Runs WITHOUT Isaac Gym (pure config/logic), so it can run anywhere. Covers:
  1. _validate_env_config (env-key whitelist + rewardTerms/pose+hold + objectAug) against every
     committed omomo*.yaml AND the curriculum ENV_TMPL rendered with features
     on/off (the generated configs my first pass never saw).
  2. the body-feature-requires-subjectBodies guard: no real config trips it.
  3. the dataset guards: empty-total raises; per-subject-missing only warns.
  4. positive controls: a typo'd key / a body-feature-without-subjectBodies /
     an empty dataset DO still raise (the guards aren't no-ops).

Run:  python tests/test_audit_guards.py   (exit 0 = all green)
"""
import glob
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMIMIC = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
CURRICULUM = os.path.join(REPO, "scripts/curriculum_runner.py")
CFG_DIR = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

BODY_FEATURE_KEYS = ("betas_file", "bodyNormalizedReward",
                     "subjectPairWeightsFile", "subjectHeightsFile")


def load_known_env_keys():
    """Parse KNOWN_ENV_KEYS = frozenset({...}) out of intermimic.py source."""
    src = open(INTERMIMIC).read()
    m = re.search(r"KNOWN_ENV_KEYS = frozenset\((\{.*?\})\)", src, re.DOTALL)
    assert m, "could not find KNOWN_ENV_KEYS in intermimic.py"
    import ast
    return set(ast.literal_eval(m.group(1)))


KNOWN = load_known_env_keys()


# --- exact mirror of InterMimic._validate_env_config (intermimic.py:112) -------
def validate_env_config(env_cfg):
    unknown = sorted(k for k in env_cfg if k not in KNOWN)
    if unknown:
        raise ValueError(f"unrecognized env config key(s): {unknown}")
    rt = env_cfg.get("rewardTerms") or {}
    bad = sorted(k for k in rt if k not in ("pose", "hold"))
    if bad:
        raise ValueError(f"unknown rewardTerms key(s) {bad}")
    for term in ("pose", "hold"):
        badp = sorted(k for k in (rt.get(term) or {}) if k not in ("enable", "lambda"))
        if badp:
            raise ValueError(f"unknown rewardTerms.{term} key(s) {badp}")
    oa = env_cfg.get("objectAug") or {}
    bado = sorted(k for k in oa if k not in
                  ("enable", "scaleMin", "scaleMax", "yawRad", "translateM", "massExp", "geom"))
    if bado:
        raise ValueError(f"unknown objectAug key(s) {bado}")
    bg = sorted(k for k in (oa.get("geom") or {}) if k not in
                ("enable", "numVariants", "anisoMin", "anisoMax"))
    if bg:
        raise ValueError(f"unknown objectAug.geom key(s) {bg}")
    bc = sorted(k for k in (env_cfg.get("objectConditioning") or {}) if k != "enable")
    if bc:
        raise ValueError(f"unknown objectConditioning key(s) {bc}")


def body_feature_guard(env_cfg, has_subject_bodies):
    """Mirror of the subjectBodies-cascade guard (intermimic.py __init__)."""
    if not has_subject_bodies:
        for k in BODY_FEATURE_KEYS:
            if env_cfg.get(k):
                raise ValueError(f"'{k}' set but subjectBodies absent")


def dataset_guards(parsed, data_sub_nums):
    """Mirror of the empty/partial dataset guards. parsed = list of
    (path, src, tgt, obj). Returns list of warnings; raises on empty-total."""
    if not parsed:
        raise ValueError("no motion files matched")
    matched = {p[2] for p in parsed}
    missing = sorted(set(data_sub_nums) - matched)
    return [f"missing {missing}"] if missing else []


# --- render the curriculum ENV_TMPL exactly as curriculum_runner would --------
def render_curriculum_env(features_on):
    src = open(CURRICULUM).read()
    tmpl = re.search(r'ENV_TMPL\s*=\s*(["\']{3})(.*?)\1', src, re.DOTALL).group(2)
    # The source writes `ENV_TMPL = """\` -- that leading backslash-newline is a
    # line continuation the Python parser strips from the real string, but our
    # raw regex capture keeps it. Drop it so the render is byte-faithful.
    tmpl = re.sub(r"^\\\n", "", tmpl)
    if features_on:
        fill = dict(
            pair_weights_line="  subjectPairWeightsFile: curriculum_work/r/cfgs/w.json",
            counts_line="  pairSampleCountsFile: curriculum_work/r/cfgs/c.json",
            heights_line="  subjectHeightsFile: scripts/heights.json",
            pose_term_block="  rewardTerms:\n    pose:\n      enable: true\n      lambda: 0.02",
            body_norm="true", cpu_motion="true", use_transformer_obs="true",
            mask_dead_envs="true", num_obs=6524,
        )
    else:
        fill = dict(
            pair_weights_line="  # subjectPairWeightsFile omitted: uniform pair sampling",
            counts_line="  # pairSampleCountsFile omitted: estimated exposure",
            heights_line="  # subjectHeightsFile omitted (SUBJECT_HEIGHTS dict only)",
            pose_term_block="  # rewardTerms.pose omitted (relative joint-angle reward off)",
            body_norm="false", cpu_motion="false", use_transformer_obs="false",
            mask_dead_envs="false", num_obs=3230,
        )
    text = tmpl.format(stage="06a", active="['sub2','sub3']",
                       datasub="['sub2', 'sub3']", bodies="['sub2', 'sub3']",
                       betas_file="scripts/omomo_betas_neutral.npz",
                       num_envs=4096, **fill)
    return yaml.safe_load(text)


def main():
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
            failures.append(name)

    # 1. every committed config + any LOCAL generated config passes validation.
    #    The curriculum_work glob is empty in a fresh checkout but on the cluster
    #    it holds the real generated substage configs -- run this test THERE to
    #    validate hand-edited / generated configs that never hit git.
    cfgs = sorted(glob.glob(os.path.join(CFG_DIR, "omomo*.yaml")))
    generated = sorted(glob.glob(os.path.join(REPO, "curriculum_work/*/cfgs/env_*.yaml")))
    cfgs += generated
    print(f"[1] {len(cfgs)} configs vs _validate_env_config "
          f"({len(generated)} local/generated):")
    for f in cfgs:
        env = (yaml.safe_load(open(f)) or {}).get("env", {})
        name = os.path.basename(f)
        check(name, lambda env=env: validate_env_config(env))
        # body-feature guard must not trip a real config
        has_sb = bool(env.get("subjectBodies"))
        check(name + " [body-feature]",
              lambda env=env, hs=has_sb: body_feature_guard(env, hs))

    # 1b. rl_games injects 'seed' into cfg['env'] on the --test/player path (below
    #     our code), so a valid eval config gains a 'seed' key at runtime. The
    #     validator MUST accept it -- otherwise every eval crashes (it did).
    print("[1b] runtime-injected keys (rl_games player path):")
    base_env = (yaml.safe_load(open(os.path.join(CFG_DIR, "omomo_test_multibody_xf.yaml"))) or {}).get("env", {})
    check("test config + injected seed passes",
          lambda: validate_env_config({**base_env, "seed": 42}))

    # 2. generated curriculum configs (features on AND off) pass validation
    print("[2] curriculum ENV_TMPL (generated configs):")
    for on in (True, False):
        env = render_curriculum_env(on)["env"]
        tag = "features-on" if on else "features-off"
        check(f"curriculum {tag}", lambda env=env: validate_env_config(env))
        check(f"curriculum {tag} [body-feature]",
              lambda env=env: body_feature_guard(env, bool(env.get("subjectBodies"))))

    # 3. dataset guards on representative identity-file sets (src==tgt)
    print("[3] dataset guards (identity files, src==tgt):")
    ok = [("sub2_x_0.pt", 2, 2, "x"), ("sub3_x_0.pt", 3, 3, "x")]
    check("all-present -> no warning",
          lambda: (dataset_guards(ok, {2, 3}) == []) or (_ for _ in ()).throw(
              AssertionError("unexpected warning")))
    check("partial -> warning not raise",
          lambda: (dataset_guards(ok, {2, 3, 99}) == ["missing [99]"]) or
                  (_ for _ in ()).throw(AssertionError("expected a warning")))

    # 4. POSITIVE CONTROLS: guards still catch genuine bugs
    print("[4] positive controls (these SHOULD raise):")
    def expect_raise(fn):
        try:
            fn()
        except Exception:
            return
        raise AssertionError("guard did NOT fire on a genuinely-broken config")
    check("typo'd env key raises",
          lambda: expect_raise(lambda: validate_env_config({"subjectBody": []})))
    check("rewardTerms typo raises",
          lambda: expect_raise(lambda: validate_env_config({"rewardTerms": {"pos": {}}})))
    check("pose.lamda typo raises",
          lambda: expect_raise(lambda: validate_env_config(
              {"rewardTerms": {"pose": {"lamda": 0.02}}})))
    check("rewardTerms.held typo raises",
          lambda: expect_raise(lambda: validate_env_config(
              {"rewardTerms": {"held": {"enable": True}}})))
    check("objectAug.scaleMn typo raises",
          lambda: expect_raise(lambda: validate_env_config(
              {"objectAug": {"scaleMn": 0.9}})))
    check("objectAug.geom.nVariants typo raises",
          lambda: expect_raise(lambda: validate_env_config(
              {"objectAug": {"geom": {"nVariants": 8}}})))
    check("objectConditioning typo raises",
          lambda: expect_raise(lambda: validate_env_config(
              {"objectConditioning": {"enabled": True}})))
    check("body-feature w/o subjectBodies raises",
          lambda: expect_raise(lambda: body_feature_guard({"betas_file": "x"}, False)))
    check("empty dataset raises",
          lambda: expect_raise(lambda: dataset_guards([], {2})))

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    print("ALL GREEN -- guards fire only on genuinely-broken configs.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The 21 component-ablation arms: 3 bases x 7 ablations, each exactly ONE
key off its base.

  bases      g3_omomo_geoall_src1__f0 (OMOMO sub1, 532 clips)
             g3_omomo_geoall_srchalf7__f0 (OMOMO 7 sources, ragged)
             g3_bball7_geoall__f0 (EgoExo4D basketball, 7 people)
  ablations  nonorm   bodyNormalizedReward false
             nopose   rewardTerms.pose.enable false
             product  rewardShape product
             nogate   rewardTerms.freeFlightGate.resets false
             realonly subjectBodies = the 13 real f0 bodies
             noret    retargetedMotionDir removed
             obs2     obsHorizons [1, 16], numObs 3198

Storage is the base's in every case (the sub1 seven stay PADDED at 384G,
exactly as the teacher trained -- Jess 2026-09-13 -- so no ragged flag).
The noret arms ask 64G (one reference per clip, not 43; Jess 2026-09-13).

What the checks cover (audit 2026-09-13 mutation probes): a second env key,
an un-mirrored or drifted eval key (eval == base eval modulo evalFor + the
ablated key), a launcher pointing at the base's cfgs, a wrong --mem, a changed
NUM_ENVS default or a deleted buffer guard, and any launcher code line that is
not in the enumerated classes of allowed differences from the base.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_ablation_cfgs.py -v
"""
import difflib
import os
import re

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg")

BASES = ["g3_omomo_geoall_src1__f0", "g3_omomo_geoall_srchalf7__f0", "g3_bball7_geoall__f0"]
ABL = ["nonorm", "nopose", "product", "nogate", "realonly", "noret", "obs2"]
REAL_F0 = ["sub1", "sub2", "sub3", "sub5", "sub6", "sub7", "sub8", "sub9",
           "sub11", "sub12", "sub14", "sub15", "sub17"]
# which leaf-path(s) each ablation may change in the env block
EXPECTED = {
    "nonorm":   {"bodyNormalizedReward"},
    "nopose":   {"rewardTerms.pose.enable"},
    "product":  {"rewardShape"},
    "nogate":   {"rewardTerms.freeFlightGate.resets"},
    "realonly": {"subjectBodies"},
    "noret":    {"retargetedMotionDir"},
    "obs2":     {"obsHorizons", "numObs"},
}
PAIRS = [(b, a) for b in BASES for a in ABL]


def _arm(base, abl):
    return base[:-len("__f0")] + f"_{abl}__f0"


def _flat(d, prefix=""):
    """{leaf path: value}; lists are leaves."""
    out = {}
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, p + "."))
        else:
            out[p] = v
    return out


def _load(rel):
    return yaml.safe_load(open(os.path.join(C, rel)))


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _code_lines(text):
    """Launcher lines that do something: drop blank and comment-only lines, keep #SBATCH."""
    return [l for l in text.split("\n")
            if l.strip() and (not l.lstrip().startswith("#") or l.startswith("#SBATCH"))]


@pytest.mark.parametrize("base,abl", PAIRS)
def test_env_cfg_is_exactly_one_key_off_its_base(base, abl):
    b, n = _load(f"omomo_teacher_{base}.yaml"), _load(f"omomo_teacher_{_arm(base, abl)}.yaml")
    assert _diff(b["env"], n["env"]) == EXPECTED[abl], (base, abl)
    assert b["sim"] == n["sim"]
    e = n["env"]
    if abl == "nonorm":
        assert e["bodyNormalizedReward"] is False
    elif abl == "nopose":
        assert e["rewardTerms"]["pose"]["enable"] is False
        assert e["rewardTerms"]["pose"]["lambda"] == 0.02          # untouched, just inert
    elif abl == "product":
        assert e["rewardShape"] == "product"
    elif abl == "nogate":
        assert e["rewardTerms"]["freeFlightGate"] == {"resets": False, "reward": False}
    elif abl == "realonly":
        assert e["subjectBodies"] == REAL_F0
    elif abl == "noret":
        assert "retargetedMotionDir" not in e
        assert e["cpuMotionData"] is True                           # still streams from host
    elif abl == "obs2":
        assert e["obsHorizons"] == [1, 16] and e["numObs"] == 2 * 1599
    assert e.get("raggedMotionData") == b["env"].get("raggedMotionData")   # storage = the base's


@pytest.mark.parametrize("base,abl", PAIRS)
def test_train_cfg_is_the_base_renamed(base, abl):
    arm = _arm(base, abl)
    b = _load(f"train/rlg/omomo_teacher_{base}.yaml")
    n = _load(f"train/rlg/omomo_teacher_{arm}.yaml")
    assert n["params"]["config"]["full_experiment_name"] == f"smplx_teacher_{arm}"
    b["params"]["config"]["full_experiment_name"] = f"smplx_teacher_{arm}"
    assert n == b


@pytest.mark.parametrize("base,abl", PAIRS)
def test_eval_cfg_is_the_base_eval_plus_the_mirrored_key(base, abl):
    arm = _arm(base, abl)
    tr = _load(f"omomo_teacher_{arm}.yaml")["env"]
    bev = _load(f"omomo_eval_{base}.yaml")
    ev = _load(f"omomo_eval_{arm}.yaml")
    assert ev["evalFor"] == [arm]
    # vs the BASE eval: only the ablated key moves (subjectBodies is eval-owned, patched per
    # pair, so realonly evals are identical). This pins numEnvs, rolloutLength,
    # physicalBufferSize, resetThresholds, sim ... to the base eval's, all at once.
    assert _diff(bev["env"], ev["env"]) == EXPECTED[abl] - {"subjectBodies"}, arm
    assert bev["sim"] == ev["sim"]
    # and the moved key equals the arm's train value
    fe, ft = _flat(ev["env"]), _flat(tr)
    for k in EXPECTED[abl] - {"subjectBodies"}:
        assert fe.get(k, "<absent>") == ft.get(k, "<absent>"), (arm, k)
    # 1024 = InterMimic's own eval scripts (isaacgym/scripts/eval_*.sh); numEnvs is
    # concurrency, not the attempt budget (the player runs 20,000 episodes per pair
    # regardless). Was 2048 until 2026-09-18. check_eval_cfg.EVAL_NUM_ENVS is the
    # single source of truth; this pin just makes drift in an ablation cfg loud here.
    assert ev["env"]["numEnvs"] == 1024 and ev["env"]["stateInit"] == "Start"


# Launcher code lines an ablation may add/remove/change relative to its base.
_ALLOWED_ANY = [
    r"^#SBATCH --time=",                       # the base's old value shows up as a removed line too
    r'^#SBATCH --job-name="tch-',
    r"^#SBATCH --output=teacher-",
    r"^CFG_(ENV|TRAIN)=isaacgym/src/intermimic/data/cfg/",
    r'^echo "\[teacher\] (G3 RECIPE|host=|ABLATION |invocation:)',
]
_ALLOWED_NORET = [
    r"^#SBATCH --mem=",                        # 64G here, the base's value as the removed line
    r"^if ! grep -qE '\^\\s\*retargetedMotionDir:'",
    r'^    echo "\[teacher\] ERROR: retarget arm without retargetedMotionDir',
    r"^\s*fi$", r"^done$",
    r"^RT=\$\(grep -oE", r'^    if ! ls "\$RT"/', r'^        echo "\[teacher\] ERROR: \$RT has no',
    r'^             "(array \+ merge|merge_retarget_trees)',
    r"^MF=\$\(grep -oE", r"^for s in sub", r'^    if ! ls "\$MF"/', r'^        echo "\[teacher\] ERROR: \$MF has no',
]


@pytest.mark.parametrize("base,abl", PAIRS)
def test_launcher_differs_from_its_base_only_in_the_allowed_classes(base, abl):
    arm = _arm(base, abl)
    bsrc = open(os.path.join(REPO, f"slurm_teacher_{base}.sh")).read()
    src = open(os.path.join(REPO, f"slurm_teacher_{arm}.sh")).read()
    code = _code_lines(src)
    # points at its own files, names itself, announces the ablation
    assert "#SBATCH --time=7-00:00:00" in src
    assert f'--job-name="tch-{arm}"' in src
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{arm}.yaml" in src
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{arm}.yaml" in src
    assert f"checkpoints/smplx_teacher_{arm}/nn/" in src
    assert f"ABLATION {abl} of {base}" in src
    stray = [l for l in code if base in l and "ABLATION" not in l]
    assert not stray, (arm, stray)
    # the recipe echo tells the truth for this arm
    recipe = next(l for l in code if "G3 RECIPE" in l)
    assert ("gate resets=FALSE" in recipe) == (abl == "nogate"), recipe
    assert ("13 REAL bodies" in recipe) == (abl == "realonly"), recipe
    # invariants an ablation must never touch (audit mutation probes M5/M6)
    assert 'NUM_ENVS="${NUM_ENVS:-2048}"' in code
    assert any("default_buffer_size_multiplier" in l for l in code)
    assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src
    # memory: the base's, except noret which holds one reference per clip
    bmem = re.search(r"^#SBATCH --mem=(\S+)$", bsrc, re.M).group(1)
    assert f"#SBATCH --mem={'64G' if abl == 'noret' else bmem}" in src
    if abl == "noret":
        assert "retargetedMotionDir:'" not in "\n".join(code) and 'RT=$(' not in "\n".join(code)
        assert any(l.startswith("for s in sub") for l in code), "noret must still check source clips"
    else:
        assert "retargetedMotionDir:'" in "\n".join(code)
    assert ("raggedMotionData:\\s*[Tt]rue" in "\n".join(code)) == ("raggedMotionData:\\s*[Tt]rue" in bsrc)
    # every changed code line falls in an allowed class (after renaming base -> arm)
    allowed = _ALLOWED_ANY + (_ALLOWED_NORET if abl == "noret" else [])
    bcode = [l.replace(base, arm) for l in _code_lines(bsrc)]
    changed = [l[2:] for l in difflib.ndiff(bcode, code) if l[:2] in ("- ", "+ ")]
    bad = [l for l in changed if not any(re.search(p, l) for p in allowed)]
    assert not bad, (arm, bad)

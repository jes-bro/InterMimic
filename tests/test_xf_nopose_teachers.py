"""The 15 transformer+nvadlr+nopose teacher arms (2026-09-25), each built as
exact-once edits off its own MLP base:

  13 OMOMO per-source   g3_omomo_geoall_src{S}_xf_nvadlr_nopose__f0
                        base omomo_teacher_g3_omomo_geoall_src{S}__f0
                        (sub2's base is the plain g3_omomo_geoall__f0; the arm
                        is named _src2 so the fleet is uniform for
                        collect_g3_teachers.py --exp-suffix)
   2 activity           g3_{bball7,soccer15}_geoall_xf_nvadlr_nopose__f0

METHOD CANDIDATES, not ablations: the teacher nopose arms beat the with-pose
teachers, so the pose term is being dropped from the method and the student
recipe is the 6-token transformer.

What this pins: env = the base's body byte-for-byte except the pose flip;
train = the base's + exactly the srcall13_xf_nvadlr knobs + its own name;
launcher points at its own cfgs and teacher dir, keeps the base's --mem, runs
7 days, carries the transformer guard, says METHOD CANDIDATE and never
ABLATION, and every changed code line falls in an enumerated class.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_xf_nopose_teachers.py -q
"""
import difflib
import glob
import os
import re

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(ROOT, "isaacgym", "src", "intermimic", "data", "cfg")
RLG = os.path.join(CFG, "train", "rlg")
SFX = "_xf_nvadlr_nopose"
OMOMO_SOURCES = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]
XF_KNOBS = {"params.network.name", "params.network.transformer.num_tokens",
            "params.network.transformer.readout_token", "params.config.lr_schedule",
            "params.config.kl_threshold", "params.config.normalize_value",
            "params.config.full_experiment_name"}


def _pairs():
    out = []
    for s in OMOMO_SOURCES:
        base = "g3_omomo_geoall__f0" if s == 2 else f"g3_omomo_geoall_src{s}__f0"
        out.append((base, f"g3_omomo_geoall_src{s}{SFX}__f0"))
    for n in ("bball7", "soccer15"):
        out.append((f"g3_{n}_geoall__f0", f"g3_{n}_geoall{SFX}__f0"))
    return out


PAIRS = _pairs()


def _flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _env(n):
    return yaml.safe_load(open(os.path.join(CFG, f"omomo_teacher_{n}.yaml")))


def _train(n):
    return yaml.safe_load(open(os.path.join(RLG, f"omomo_teacher_{n}.yaml")))


def _body(path):
    """Everything from the first top-level key on (the header comment dropped)."""
    lines = open(path).read().splitlines()
    return lines[next(i for i, l in enumerate(lines) if l.startswith(("env:", "params:"))):]


def _launcher(n):
    return open(os.path.join(ROOT, f"slurm_teacher_{n}.sh")).read()


def _code_lines(text):
    """Launcher lines that do something: drop blank and comment-only lines, keep #SBATCH."""
    return [l for l in text.split("\n")
            if l.strip() and (not l.lstrip().startswith("#") or l.startswith("#SBATCH"))]


# ----------------------------------------------------------------------------- env cfgs
@pytest.mark.parametrize("base,arm", PAIRS)
def test_env_is_the_base_body_with_only_the_pose_flip(base, arm):
    b, n = _env(base), _env(arm)
    assert _diff(b, n) == {"env.rewardTerms.pose.enable"}, arm
    assert n["env"]["rewardTerms"]["pose"] == {"enable": False, "lambda": 0.02}   # lambda untouched, inert
    assert b["env"]["rewardTerms"]["pose"]["enable"] is True
    assert n["sim"] == b["sim"]
    # textual, not just semantic: the body is the base's byte-for-byte except that one line
    bl, nl = _body(os.path.join(CFG, f"omomo_teacher_{base}.yaml")), _body(os.path.join(CFG, f"omomo_teacher_{arm}.yaml"))
    changed = [(x, y) for x, y in zip(bl, nl) if x != y]
    assert len(bl) == len(nl) and changed == [("      enable: true", "      enable: false")], (arm, changed)
    # the transformer's six tokens need the recipe's six horizons
    assert len(n["env"]["obsHorizons"]) == 6 and n["env"]["obsHorizons"][0] == 1
    assert n["env"]["rewardShape"] == "geometric_all"       # with pose off the root spans 4


def test_env_headers_say_method_candidate_not_ablation():
    for _, arm in PAIRS:
        head = "\n".join(l for l in open(os.path.join(CFG, f"omomo_teacher_{arm}.yaml")).read().splitlines()
                         if l.startswith("#"))
        assert "METHOD CANDIDATE" in head and "NOT an ablation" in head, arm
        assert "ABLATION" not in head, arm


# --------------------------------------------------------------------------- train cfgs
@pytest.mark.parametrize("base,arm", PAIRS)
def test_train_is_the_base_plus_exactly_the_xf_nvadlr_knobs(base, arm):
    a, b = _flat(_train(base)), _flat(_train(arm))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == XF_KNOBS, arm
    assert b["params.network.name"] == "intermimic_transformer"
    assert b["params.network.transformer.num_tokens"] == 6 and b["params.network.transformer.readout_token"] == 0
    assert b["params.config.lr_schedule"] == "adaptive" and b["params.config.kl_threshold"] == 0.06
    assert b["params.config.normalize_value"] is True
    assert b["params.config.full_experiment_name"] == f"smplx_teacher_{arm}"
    assert a["params.network.name"] == "intermimic" and a["params.config.lr_schedule"] == "constant"


def test_train_body_equals_the_srcall13_xf_teacher_renamed():
    """One transformer recipe for the whole fleet: byte-equal below the header to
    the existing XF teacher's train cfg, modulo the experiment name."""
    ref = _body(os.path.join(RLG, "omomo_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0.yaml"))
    for _, arm in PAIRS:
        got = _body(os.path.join(RLG, f"omomo_teacher_{arm}.yaml"))
        exp = [l.replace("smplx_teacher_g3_omomo_geoall_srcall13_xf_nvadlr__f0", f"smplx_teacher_{arm}") for l in ref]
        assert got == exp, arm


def test_experiment_names_are_new_and_distinct():
    names = {}
    for p in glob.glob(os.path.join(RLG, "omomo_teacher_g3_*__f0.yaml")):
        names.setdefault(yaml.safe_load(open(p))["params"]["config"]["full_experiment_name"], []).append(os.path.basename(p))
    dup = {k: v for k, v in names.items() if len(v) > 1}
    assert not dup, dup
    for _, arm in PAIRS:
        assert f"smplx_teacher_{arm}" in names


# ---------------------------------------------------------------------------- launchers
_ALLOWED_ANY = [
    r"^#SBATCH --time=",                        # 24h -> 7-day on the per-source bases
    r'^#SBATCH --job-name="tch-',
    r"^#SBATCH --output=teacher-",
    r'^conda activate "\$\{INTERMIMIC_ENV:-intermimic-gym2\}"',
    r"^conda activate intermimic-gym2$",       # the base's line shows up as removed
    r"^CFG_(ENV|TRAIN)=isaacgym/src/intermimic/data/cfg/",
    r'^echo "\[teacher\] (G3 RECIPE|host=|METHOD CANDIDATE )',
    # the transformer guard, line by line
    r"^if ! grep -qE '\^\\s\*name:\\s\*intermimic_transformer",
    r'^    echo "\[teacher\] ERROR: XF arm without network intermimic_transformer',
    r"^if ! grep -qE '\^\\s\*num_tokens:\\s\*6",
    r'^    echo "\[teacher\] ERROR: XF arm without transformer\.num_tokens 6',
    r"^fi$",
]


@pytest.mark.parametrize("base,arm", PAIRS)
def test_launcher_points_at_its_own_files_and_differs_only_in_the_allowed_classes(base, arm):
    bsrc, src = _launcher(base), _launcher(arm)
    code = _code_lines(src)
    assert "#SBATCH --time=7-00:00:00" in src
    assert f'--job-name="tch-{arm}"' in src
    assert f"#SBATCH --output=teacher-{arm}-%j.out" in src
    assert f"CFG_ENV=isaacgym/src/intermimic/data/cfg/omomo_teacher_{arm}.yaml" in src
    assert f"CFG_TRAIN=isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_{arm}.yaml" in src
    assert f"checkpoints/smplx_teacher_{arm}/nn/" in src
    assert f"METHOD CANDIDATE xf_nvadlr_nopose of {base}" in src
    assert "rewardTerms.pose.enable true -> false" in src
    assert "ABLATION" not in src, "these are method candidates, not ablations (Jess 2026-09-25)"
    # memory = the base's (storage = the base's)
    bmem = re.search(r"^#SBATCH --mem=(\S+)$", bsrc, re.M).group(1)
    assert f"#SBATCH --mem={bmem}" in src
    # the transformer guard is present and reads the train cfg
    assert "intermimic_transformer" in "\n".join(code) and "num_tokens" in "\n".join(code)
    # the recipe echo tells the truth for this arm
    recipe = next(l for l in code if "G3 RECIPE" in l)
    assert f"G3 RECIPE {arm} (" in recipe and "6-TOKEN TRANSFORMER" in recipe and "NO POSE TERM" in recipe
    # no code line still names the base experiment (the METHOD CANDIDATE echo may)
    stray = [l for l in code if f"{base}" in l and "METHOD CANDIDATE" not in l]
    assert not stray, (arm, stray)
    # invariants a one-key arm never touches
    assert 'NUM_ENVS="${NUM_ENVS:-2048}"' in code
    assert any("default_buffer_size_multiplier" in l for l in code)
    assert "resume_from:)\\s*'?None'?" in src and "refusing to" in src
    assert "-m intermimic.run " in src and "--task InterMimic " in src
    assert ("raggedMotionData:\\s*[Tt]rue" in "\n".join(code)) == ("raggedMotionData:\\s*[Tt]rue" in bsrc)
    # every changed code line falls in an allowed class (after renaming base -> arm)
    bcode = [l.replace(base, arm) for l in _code_lines(bsrc)]
    changed = [l[2:] for l in difflib.ndiff(bcode, code) if l[:2] in ("- ", "+ ")]
    bad = [l for l in changed if not any(re.search(p, l) for p in _ALLOWED_ANY)]
    assert not bad, (arm, bad)


def test_fleet_is_complete():
    """13 OMOMO sources (= the tfinal student's teacher set) + bball7 + soccer15, no cpr."""
    arms = sorted(os.path.basename(p)[len("slurm_teacher_"):-3]
                  for p in glob.glob(os.path.join(ROOT, f"slurm_teacher_g3_*{SFX}__f0.sh")))
    assert arms == sorted(a for _, a in PAIRS)
    assert not any("cpr" in a for a in arms)
    srcs = sorted(int(re.search(r"_src(\d+)_", a).group(1)) for a in arms if "_src" in a)
    assert srcs == sorted(OMOMO_SOURCES) and len(srcs) == 13

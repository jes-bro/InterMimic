#!/usr/bin/env python3
"""Generate objectAug experiment env configs from the multibody base.

Single source of truth for the object-augmentation reward experiments (mirrors
scripts/generate_crosspair_cfgs.py). Emits a 2-axis matrix of env YAMLs, and
exposes render() so scripts/objectaug_runner.py builds per-stage configs the
same way (curriculum widens the perturbation ranges across warm-started stages).

Every config has objectAug ON -- each env's object is perturbed (per-env fixed
scale with mass held constant; per-episode yaw + XY translate).

Axis 1 -- stock object terms (ro*rig*rcg, which track the EXACT source object):
  drop : objectTermsEnable=false. Perturbed object can't match the source, so
         these are dropped; base reward is rb (humanoid tracking).
  keep : objectTermsEnable=true.  Kept -- under mild perturbation they give a
         graded "stay near the source" pressure (best at the gentle end of the
         curriculum; the larger the perturbation, the more they drag reward down).

Axis 2 -- extra layered factors:
  base : none   pose : + Term1 relative-joint-angle   hold : + Term2 contact/hold
  both : pose + hold

  omomo_objectaug_{drop,keep}_{base,pose,hold,both}.yaml   (8 files)

Object/ig/contact RESETS are relaxed to human-only whenever objectAug is on
(gated on objectAug in code), so 'keep' runs aren't killed on object divergence.

Pair with train/rlg/omomo_multibody.yaml as --cfg_train and warm-start from the
curriculum fold-in checkpoint. dataSub / subjectBodies inherited from the base
MUST match that checkpoint's env layout (numObs 3230, betas).

Run from repo root:  python3 scripts/generate_objectaug_cfgs.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "isaacgym/src/intermimic/data/cfg"
BASE = CFG_DIR / "omomo_train_multibody.yaml"

# --- default single-stage perturbation ranges. The batch matrix below uses
# these; the curriculum runner OVERRIDES them per stage via render(). --------
SCALE_MIN, SCALE_MAX = 0.8, 1.25      # per-env fixed scale, sampled U(min,max)
YAW_RAD = 0.52                        # +/- radians (~30 deg), per episode
TRANSLATE_M = 0.10                    # +/- metres (XY), per episode
HOLD_MASS = "true"                    # rescale density so mass ~constant

# --- extra reward-term weights ---------------------------------------------
POSE_LAMBDA = 0.02   # sum over 153 DOFs => ~ the rotation term's 2.5 per-DOF
HOLD_LAMBDA = 5.0    # hand->object-surface proximity weight

# Axis 1: keep vs drop the stock object terms. (slug, objectTermsEnable value)
OBJTERMS = [("drop", "false"), ("keep", "true")]
# Axis 2: extra layered factors. (slug, pose_enable, hold_enable, description)
TERMS = [
    ("base", "false", "false", "no extra terms"),
    ("pose", "true",  "false", "+ Term1 relative-joint-angle pose"),
    ("hold", "false", "true",  "+ Term2 relaxed contact/hold"),
    ("both", "true",  "true",  "+ pose + hold"),
]


def _env_blocks(object_terms, pose_enable, hold_enable,
                scale_min, scale_max, yaw_rad, translate_m,
                hold_mass, pose_lambda, hold_lambda):
    """The env-section keys appended for objectAug + the reward toggles."""
    return f"""
  # --- Object augmentation (one curriculum stage) ----------------------------
  objectTermsEnable: {object_terms}   # keep/drop stock ro*rig*rcg
  objectAug:
    enable: true
    scaleMin: {scale_min}
    scaleMax: {scale_max}
    yawRad: {yaw_rad}        # +/- radians (~{int(round(yaw_rad * 57.2958))} deg), per episode
    translateM: {translate_m}     # +/- metres (XY), per episode
    holdMass: {hold_mass}
  rewardTerms:
    pose: {{ enable: {pose_enable}, lambda: {pose_lambda} }}   # Term 1: parent-relative joint angles
    hold: {{ enable: {hold_enable}, lambda: {hold_lambda} }}    # Term 2: relaxed contact / hold
"""


def split_base():
    """Return (env_part, sim_part): the multibody base split at top-level sim:."""
    base_text = BASE.read_text()
    assert "\nsim:" in base_text, "unexpected base config layout (no top-level sim:)"
    env_part, sim_part = base_text.split("\nsim:", 1)
    return env_part, sim_part


def render(object_terms, pose_enable, hold_enable, *,
           scale_min=SCALE_MIN, scale_max=SCALE_MAX, yaw_rad=YAW_RAD,
           translate_m=TRANSLATE_M, hold_mass=HOLD_MASS,
           pose_lambda=POSE_LAMBDA, hold_lambda=HOLD_LAMBDA, header=""):
    """Full env-YAML text: multibody base + objectAug/reward blocks.

    object_terms / pose_enable / hold_enable are YAML literals ('true'/'false').
    The perturbation ranges and lambdas default to the module constants; the
    curriculum runner passes per-stage values to widen diversity across stages.
    """
    env_part, sim_part = split_base()
    blocks = _env_blocks(object_terms, pose_enable, hold_enable,
                         scale_min, scale_max, yaw_rad, translate_m,
                         hold_mass, pose_lambda, hold_lambda)
    return header + env_part + blocks + "\nsim:" + sim_part


def main():
    n = 0
    for obj_slug, object_terms in OBJTERMS:
        kept = "KEPT" if object_terms == "true" else "DROPPED"
        for term_slug, pose_enable, hold_enable, term_desc in TERMS:
            slug = f"{obj_slug}_{term_slug}"
            header = (
                f"# objectAug experiment -- stock object terms {kept}; {term_desc}\n"
                f"# GENERATED by scripts/generate_objectaug_cfgs.py -- do not edit by hand.\n"
                f"# Warm-start from the curriculum fold-in policy; dataSub / subjectBodies\n"
                f"# (inherited from omomo_train_multibody.yaml) MUST match that checkpoint.\n"
            )
            dst = CFG_DIR / f"omomo_objectaug_{slug}.yaml"
            dst.write_text(render(object_terms, pose_enable, hold_enable, header=header))
            print(f"wrote {dst.relative_to(REPO)}")
            n += 1

    print(f"\n{n} configs written. Stage ranges: scale [{SCALE_MIN},{SCALE_MAX}], "
          f"yaw +/-{YAW_RAD}rad, translate +/-{TRANSLATE_M}m.")


if __name__ == "__main__":
    raise SystemExit(main())

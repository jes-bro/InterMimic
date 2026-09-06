"""Fixture tests for the takeover eval (learning/takeover.py).

The property that has to hold above all others: with TAKEOVER_K unset or 0, the
instrument is a NO-OP. A takeover run is only interpretable against the ordinary
eval, so if merely wiring this in moved the k=0 numbers, every comparison built
on it would be measuring the harness. That is the acceptance gate, and most of
these tests exist to pin it down from several directions.

The second property: envs PAST the takeover point get the teacher's action
bit-for-bit. If residual noise leaked past step k, the back half of the episode
would no longer be a clean measurement of the teacher.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "isaacgym" / "src" / "intermimic" / "learning"))

import takeover  # noqa: E402


def cfg(k=10, noise=0.1, seed=0):
    return takeover.TakeoverConfig(k=k, noise=noise, seed=seed)


# ---------------------------------------------------------------- the no-op gate

def test_unset_env_is_off():
    assert takeover.TakeoverConfig.from_env({}) is None


def test_empty_string_is_off():
    assert takeover.TakeoverConfig.from_env({"TAKEOVER_K": ""}) is None


def test_k_zero_is_off_not_a_zero_length_wander():
    """k=0 must return None, so the acceptance run exercises the real no-op path.

    A k=0 TakeoverConfig would take the masking branch with an all-False mask and
    "agree" with a normal eval by coincidence. Collapsing it to None means the
    k=0 comparison actually tests the path an ordinary eval takes.
    """
    assert takeover.TakeoverConfig.from_env({"TAKEOVER_K": "0"}) is None


def test_noise_alone_does_nothing_without_k():
    assert takeover.TakeoverConfig.from_env({"TAKEOVER_NOISE": "0.5"}) is None


# -------------------------------------------------------- loud failure, no guess

def test_unparseable_k_raises_rather_than_falling_back():
    """A typo'd budget must not quietly become an ordinary eval.

    Silently reading TAKEOVER_K=1O (letter O) as "off" would file a plain eval
    as a takeover result -- the exact class of silent-fallback bug this repo
    keeps stamping out.
    """
    with pytest.raises(ValueError, match="not an integer"):
        takeover.TakeoverConfig.from_env({"TAKEOVER_K": "1O"})


def test_negative_k_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        takeover.TakeoverConfig(k=-1, noise=0.1)


def test_negative_noise_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        takeover.TakeoverConfig(k=10, noise=-0.1)


def test_unparseable_noise_raises():
    with pytest.raises(ValueError, match="not a number"):
        takeover.TakeoverConfig.from_env({"TAKEOVER_K": "50", "TAKEOVER_NOISE": "loud"})


def test_env_parsing_reads_all_three():
    c = takeover.TakeoverConfig.from_env(
        {"TAKEOVER_K": "50", "TAKEOVER_NOISE": "0.25", "TAKEOVER_SEED": "7"})
    assert (c.k, c.noise, c.seed) == (50, 0.25, 7)


def test_noise_defaults_when_only_k_given():
    c = takeover.TakeoverConfig.from_env({"TAKEOVER_K": "50"})
    assert c.k == 50 and c.noise == 0.1 and c.seed is None


# ------------------------------------------------------------------ the masking

def test_only_envs_before_k_are_perturbed():
    """Envs past the takeover point must be bit-identical to the teacher."""
    prog = torch.tensor([0, 5, 9, 10, 11, 200])
    act = torch.randn(6, 4)
    out = takeover.apply(act, prog, cfg(k=10, noise=1.0))
    wandering = prog < 10
    # perturbed where wandering...
    assert not torch.equal(out[wandering], act[wandering])
    # ...and untouched, exactly, where not
    assert torch.equal(out[~wandering], act[~wandering])


def test_boundary_is_progress_lt_k_not_le():
    """progress == k is the first TEACHER step, not the last noisy one."""
    prog = torch.tensor([9, 10])
    act = torch.zeros(2, 3)
    out = takeover.apply(act, prog, cfg(k=10, noise=5.0))
    assert not torch.equal(out[0], act[0]), "progress=9 (<k) should be perturbed"
    assert torch.equal(out[1], act[1]), "progress=10 (==k) must be the teacher"


def test_all_past_k_returns_untouched_action():
    prog = torch.tensor([50, 60, 70])
    act = torch.randn(3, 4)
    out = takeover.apply(act, prog, cfg(k=10, noise=1.0))
    assert torch.equal(out, act)


def test_zero_noise_is_identity_even_while_wandering():
    """The deliberate control: same masking path, zero perturbation.

    If a noise=0 run does NOT reproduce the k=0 numbers, the harness moved them
    and no teacher comparison from it is trustworthy.
    """
    prog = torch.zeros(8, dtype=torch.long)
    act = torch.randn(8, 5)
    out = takeover.apply(act, prog, cfg(k=100, noise=0.0))
    assert torch.equal(out, act)


def test_input_action_is_not_modified_in_place():
    prog = torch.zeros(4, dtype=torch.long)
    act = torch.randn(4, 3)
    before = act.clone()
    takeover.apply(act, prog, cfg(k=10, noise=1.0))
    assert torch.equal(act, before)


def test_shape_and_dtype_preserved():
    prog = torch.zeros(6, dtype=torch.long)
    act = torch.randn(6, 28, dtype=torch.float32)
    out = takeover.apply(act, prog, cfg(k=5, noise=0.3))
    assert out.shape == act.shape and out.dtype == act.dtype


def test_seeded_draws_are_reproducible():
    """Two arms must be comparable under the SAME perturbations, not two draws."""
    prog = torch.zeros(16, dtype=torch.long)
    act = torch.randn(16, 4)
    g1 = torch.Generator(); g1.manual_seed(1234)
    g2 = torch.Generator(); g2.manual_seed(1234)
    a = takeover.apply(act, prog, cfg(k=10, noise=0.5), generator=g1)
    b = takeover.apply(act, prog, cfg(k=10, noise=0.5), generator=g2)
    assert torch.equal(a, b)


def test_noise_scale_tracks_sigma():
    prog = torch.zeros(20000, dtype=torch.long)
    act = torch.zeros(20000, 2)
    g = torch.Generator(); g.manual_seed(0)
    out = takeover.apply(act, prog, cfg(k=10, noise=0.25), generator=g)
    assert abs(out.std().item() - 0.25) < 0.02


# --------------------------------------------------------- wander-death counting

def test_wander_deaths_counts_only_deaths_during_the_noise():
    """Deaths under the noise are not teacher failures and must stay separate."""
    done = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
    wandering = torch.tensor([True, False, True, True])
    # env0 and env3 died while wandering; env1 died AFTER takeover (a real
    # teacher failure); env2 is alive.
    assert takeover.count_wander_deaths(done, wandering) == 2


def test_wander_deaths_accepts_bool_done():
    done = torch.tensor([True, False, True])
    wandering = torch.tensor([True, True, False])
    assert takeover.count_wander_deaths(done, wandering) == 1


def test_wander_deaths_zero_when_nothing_died():
    done = torch.zeros(5, dtype=torch.bool)
    wandering = torch.ones(5, dtype=torch.bool)
    assert takeover.count_wander_deaths(done, wandering) == 0


# ------------------------------------------------- the submission wrapper (sh)

import subprocess  # noqa: E402

WRAPPER = REPO / "scripts" / "takeover_eval.sh"


def dry(arm, k, **env):
    """DRY=1 run of the wrapper: prints its plan, submits nothing."""
    import os
    e = {**os.environ, "DRY": "1", **{k2: str(v) for k2, v in env.items()}}
    return subprocess.run(["sh", str(WRAPPER), arm, str(k)],
                          cwd=REPO, capture_output=True, text=True, env=e)


def test_wrapper_scores_in_distribution_bodies_only():
    """No held-out and no synthetic bodies.

    Under the current split the student holds out the same trio the teachers do,
    so the teacher is never asked to label sub10/13/16 and its scores there
    cannot speak to its fitness as a teacher. sub100+ are training bodies, not a
    teaching signal.
    """
    r = dry("g3_omomo_geoall__f0", 50)
    assert r.returncode == 0, r.stderr
    line = next(l for l in r.stdout.splitlines() if l.strip().startswith("bodies"))
    bodies = line.split(":", 1)[1].split()
    assert bodies, "no bodies resolved"
    assert not ({"sub10", "sub13", "sub16"} & set(bodies)), f"held-out leaked in: {bodies}"
    assert not [b for b in bodies if int(b[3:]) >= 100], f"synthetics leaked in: {bodies}"
    assert "sub4" not in bodies, "sub4 crashes the sim and must never be scored"


def test_wrapper_pairs_the_two_candidate_arms_on_identical_bodies():
    """geoall vs nobetas must be scored on the same bodies or it is not an A/B."""
    a = dry("g3_omomo_geoall__f0", 50)
    b = dry("g3_omomo_nobetas__f0", 50)
    assert a.returncode == 0 and b.returncode == 0
    grab = lambda out: next(l for l in out.splitlines()
                            if l.strip().startswith("bodies")).split(":", 1)[1].split()
    assert grab(a.stdout) == grab(b.stdout)


def test_k_zero_is_labelled_acceptance_and_gets_its_own_csv():
    r = dry("g3_omomo_geoall__f0", 0)
    assert r.returncode == 0, r.stderr
    assert "ACCEPTANCE" in r.stdout
    assert "takeover_k0_ACCEPTANCE.csv" in r.stdout


def test_csv_name_separates_k_and_sigma():
    """Runs at different k or sigma must never collide on one output path."""
    a = dry("g3_omomo_geoall__f0", 50)
    b = dry("g3_omomo_geoall__f0", 100)
    c = dry("g3_omomo_geoall__f0", 50, NOISE="0.3")
    outs = {next(l for l in r.stdout.splitlines() if "-> csv" in l) for r in (a, b, c)}
    assert len(outs) == 3, f"csv paths collided: {outs}"


def test_non_integer_k_is_rejected_loudly():
    r = dry("g3_omomo_geoall__f0", "fifty")
    assert r.returncode != 0
    assert "k must be a non-negative integer" in r.stderr


def test_unknown_arm_is_rejected_loudly():
    r = dry("g3_no_such_arm__f0", 50)
    assert r.returncode != 0
    assert "no env config for arm" in r.stderr

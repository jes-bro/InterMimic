#!/usr/bin/env python3
"""Tests for scripts/generate_synthetic_bodies.py -- prior sampling, no subject involved.

The generator must be a function of the seed alone. The tests pin that from
the outside: run the script twice with the same seed against two betas files
whose REAL bodies (training and test alike) are completely different, and
demand bit-identical synthetic bodies. If anyone ever re-adds blending from
training subjects or a distance check against test subjects, that test fails.
A second group checks the Latin hypercube spread: one body per 1/n slice on
every axis.

Uses the REAL SMPL-X neutral model (the generator's default --models-dir,
~/Downloads/models/smplx) and the real OMOMO neutral betas. Missing = hard failure.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_generate_synthetic_bodies.py -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr            # standard normal CDF, to undo ndtri

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "generate_synthetic_bodies.py"
REAL_BETAS = REPO / "scripts" / "omomo_betas_neutral.npz"
MODELS_DIR = Path.home() / "Downloads" / "models" / "smplx"   # generator default

REALS = [f"sub{i}" for i in range(1, 18)]          # the 17 OMOMO subjects
N_BETAS = 16
N, START = 30, 600                                  # ids 600-629: a free block


def test_real_inputs_present():
    """Fail loudly up front if the real model or betas are missing."""
    assert (MODELS_DIR / "SMPLX_NEUTRAL.npz").is_file(), \
        f"real SMPL-X model required at {MODELS_DIR}/SMPLX_NEUTRAL.npz"
    assert REAL_BETAS.is_file(), f"missing {REAL_BETAS}"


# ------------------------------------------------------------------ helpers
def write_betas(path, override=None):
    """Copy of the real neutral betas file, with chosen subjects' betas replaced."""
    z = np.load(REAL_BETAS, allow_pickle=True)
    betas = {k: z[k] for k in z.files}
    if override:
        betas.update(override)
    np.savez(path, **betas)


def run_generator(tmp, betas_path, tag, extra=()):
    """Invoke the script as a subprocess (the way it is run for real) and return
    (stdout, syn npz dict, combined npz dict, heights json)."""
    out = tmp / f"syn_{tag}.npz"
    combined = tmp / f"combined_{tag}.npz"
    heights = tmp / f"heights_{tag}.json"
    cmd = [sys.executable, str(SCRIPT),
           "--betas", str(betas_path), "--models-dir", str(MODELS_DIR),
           "--n", str(N), "--start-id", str(START), "--seed", "0",
           "--out", str(out), "--combined-out", str(combined),
           "--heights-out", str(heights), *extra]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert res.returncode == 0, f"generator failed:\nSTDOUT\n{res.stdout}\nSTDERR\n{res.stderr}"
    syn = dict(np.load(out, allow_pickle=True))
    comb = dict(np.load(combined, allow_pickle=True))
    return res.stdout, syn, comb, json.load(open(heights))


def syn_names(z):
    return [k for k in z if k.startswith("sub")]


def stack(syn):
    return np.stack([syn[n] for n in syn_names(syn)])


def assert_same_synthetics(syn_a, syn_b, h_a, h_b):
    assert syn_names(syn_a) == syn_names(syn_b) == [f"sub{i}" for i in range(START, START + N)]
    for n in syn_names(syn_a):
        np.testing.assert_array_equal(syn_a[n], syn_b[n])
    assert list(syn_a["_kinds"]) == list(syn_b["_kinds"])
    assert h_a == h_b


# --------------------------------------------------------------------- tests
def test_independent_of_every_real_body(tmp_path):
    """THE invariant: replace ALL 17 real subjects' betas (training and test)
    with garbage -> the synthetic bodies are bit-identical. The set depends on
    the seed only."""
    clean = tmp_path / "betas_clean.npz"
    write_betas(clean)
    rng = np.random.default_rng(99)
    garbage = tmp_path / "betas_garbage.npz"
    write_betas(garbage, override={s: (50.0 * rng.standard_normal(N_BETAS)).astype(np.float32)
                                   for s in REALS})
    _, syn_a, _, h_a = run_generator(tmp_path, clean, "a")
    _, syn_b, _, h_b = run_generator(tmp_path, garbage, "b")
    assert_same_synthetics(syn_a, syn_b, h_a, h_b)


def test_seed_is_the_only_input(tmp_path):
    """Same seed twice -> identical; a different seed -> different bodies."""
    _, syn_a, _, h_a = run_generator(tmp_path, REAL_BETAS, "a")
    _, syn_b, _, h_b = run_generator(tmp_path, REAL_BETAS, "b")
    assert_same_synthetics(syn_a, syn_b, h_a, h_b)
    _, syn_c, _, _ = run_generator(tmp_path, REAL_BETAS, "c", extra=["--seed", "1"])
    assert any(not np.array_equal(syn_a[n], syn_c[n]) for n in syn_names(syn_a))


def test_real_betas_are_copied_untouched(tmp_path):
    """The combined file carries every real subject exactly as given (the env
    looks them up), followed by the synthetics."""
    _, syn, comb, _ = run_generator(tmp_path, REAL_BETAS, "a")
    z = np.load(REAL_BETAS, allow_pickle=True)
    assert [k for k in comb if k.startswith("sub")] == REALS + syn_names(syn)
    for s in REALS:
        np.testing.assert_array_equal(comb[s], z[s])
    for n in syn_names(syn):
        np.testing.assert_array_equal(comb[n], syn[n])
    assert list(comb["_genders"]) == [f"{k}:neutral" for k in REALS + syn_names(syn)]


def test_latin_hypercube_covers_every_axis(tmp_path):
    """Undo the normal quantile map and check the Latin hypercube property:
    on EVERY axis the n bodies occupy n distinct equal-probability slices,
    i.e. the population is covered end to end with no clumping."""
    _, syn, _, _ = run_generator(tmp_path, REAL_BETAS, "a")
    S = stack(syn).astype(np.float64)
    assert S.shape == (N, N_BETAS)
    slices = np.floor(ndtr(S) * N).astype(int)        # which 1/N slice each value is in
    for axis in range(N_BETAS):
        assert sorted(slices[:, axis]) == list(range(N)), \
            f"axis {axis}: slices {sorted(slices[:, axis])} are not one-per-slice"


def test_clip(tmp_path):
    """Every beta within +/-4 by default; a tighter --clip is honoured and binds."""
    _, syn, _, _ = run_generator(tmp_path, REAL_BETAS, "a")
    S = stack(syn)
    assert S.dtype == np.float32 and np.abs(S).max() <= 4.0
    _, syn_t, _, _ = run_generator(tmp_path, REAL_BETAS, "t", extra=["--clip", "1.0"])
    St = stack(syn_t)
    assert np.abs(St).max() <= 1.0
    assert (np.abs(St) == 1.0).any()                # something actually got clipped


def test_legacy_flags_are_gone(tmp_path):
    """No blending, no test-distance filtering, no held-out list: the knobs that
    produced the invalid sets must not parse."""
    for flag in (["--min-heldout-dist", "2.0"], ["--held-out", "sub10"],
                 ["--frac-extrap", "0.3"], ["--proportion-margin", "0.3"],
                 ["--min-pairwise-dist", "2.0"]):
        res = subprocess.run([sys.executable, str(SCRIPT), "--betas", str(REAL_BETAS),
                              "--models-dir", str(MODELS_DIR),
                              "--out", str(tmp_path / "x.npz"),
                              "--combined-out", str(tmp_path / "c.npz"), *flag],
                             cwd=REPO, capture_output=True, text=True)
        assert res.returncode != 0 and "unrecognized arguments" in res.stderr, flag


def test_id_clash_fails_loudly(tmp_path):
    """A --start-id that collides with an existing entry in --betas is an error."""
    res = subprocess.run([sys.executable, str(SCRIPT), "--betas", str(REAL_BETAS),
                          "--models-dir", str(MODELS_DIR), "--start-id", "1",
                          "--out", str(tmp_path / "x.npz"),
                          "--combined-out", str(tmp_path / "c.npz")],
                         cwd=REPO, capture_output=True, text=True)
    assert res.returncode != 0 and "already exist" in res.stderr


def test_file_layout_and_heights(tmp_path):
    """ids, tags and heights keys match what the env and generate_per_subject_mjcfs.py
    expect; heights come from the real mesh and are human-sized."""
    _, syn, _, heights = run_generator(tmp_path, REAL_BETAS, "a")
    names = syn_names(syn)
    assert names == [f"sub{i}" for i in range(START, START + N)]
    assert list(syn["_kinds"]) == ["prior"] * N
    assert list(syn["_genders"]) == [f"{n}:neutral" for n in names]
    assert set(heights) == {str(i) for i in range(START, START + N)}
    assert all(1.30 <= v <= 2.20 for v in heights.values()), heights

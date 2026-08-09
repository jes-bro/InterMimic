#!/usr/bin/env python3
"""Tests for scripts/generate_kfold_cfgs.py -- the fold-assignment invariants
and the generated files. These are the properties that would silently corrupt
the CV if wrong (a test body left in training, a leaky synthetic retained).

Run:  python tests/test_generate_kfold_cfgs.py   (exit 0 = all green)
"""
import os
import re
import sys
import tempfile

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import generate_kfold_cfgs as gk  # noqa: E402

raw = np.load(gk.BETAS)
BETAS = {k: raw[k] for k in raw.files if k.startswith("sub")}


def test_fold_invariants():
    folds, near = gk.assign_folds(BETAS)
    flat = [b for f in folds for b in f]
    assert len(folds) == 3 and all(len(f) == 3 for f in folds), folds
    assert len(set(flat)) == 9, "test sets overlap"
    for banned in ("sub9", "sub4", "sub2", "sub10", "sub13", "sub16"):
        assert banned not in flat, f"{banned} must never be a test body"
    assert all(int(b[3:]) < 100 for b in flat), "synthetic body in a test set"
    # the 2 never-test bodies really are the nearest to the source
    pool = flat + near
    d = {b: gk.beta_dist(BETAS, b, gk.SOURCE) for b in pool}
    assert sorted(near, key=d.get) == sorted(pool, key=d.get)[:2], (near, d)
    print("ok: fold invariants (3x3, disjoint, exclusions honored)")


def test_generated_files():
    with tempfile.TemporaryDirectory() as tmp:
        written = gk.main(["--out-root", tmp])
        assert len(written) == 9, written  # 3 folds x (env, train, slurm)
        folds, _ = gk.assign_folds(BETAS)
        for i, test in enumerate(folds, start=1):
            env = yaml.safe_load(open(os.path.join(
                tmp, f"isaacgym/src/intermimic/data/cfg/omomo_teacher_kfold{i}_src2_mlp.yaml")))
            bodies = env["env"]["subjectBodies"]
            assert not set(test) & set(bodies), f"fold{i}: test body in training"
            assert "sub121" not in bodies and "sub4" not in bodies
            assert "sub2" in bodies and "sub9" in bodies  # excluded from TEST only
            # fold0's bodies return to training in every new fold
            assert {"sub10", "sub13", "sub16"} <= set(bodies), f"fold{i}"
            # EVERY retained training body (synthetic AND real) clears the
            # computed threshold = smallest real-real subject distance.
            thr = gk.real_human_floor(BETAS)
            assert abs(thr - 2.106) < 0.01, thr   # pin the calibration
            for b in bodies:
                dmin = min(gk.beta_dist(BETAS, b, t) for t in test)
                assert dmin >= thr - 1e-9, f"fold{i}: {b} kept at {dmin:.3f} < {thr:.3f}"
            tr = open(os.path.join(
                tmp, f"isaacgym/src/intermimic/data/cfg/train/rlg/omomo_teacher_kfold{i}_src2_mlp.yaml")).read()
            assert f"full_experiment_name: smplx_teacher_kfold{i}_src2_mlp" in tr
            sl = open(os.path.join(tmp, f"slurm_teacher_kfold{i}_src2_mlp.sh")).read()
            assert f'HELDOUT="{" ".join(test)}"' in sl and "REWARD_BREAKDOWN=1" in sl
        # at least one synthetic was leak-dropped somewhere (sub107/sub109 today)
        all_syn = {b for b in yaml.safe_load(open(gk.BASE_ENV))["env"]["subjectBodies"]
                   if int(b[3:]) >= 100}
        kept_everywhere = set.intersection(*[
            {b for b in yaml.safe_load(open(os.path.join(
                tmp, f"isaacgym/src/intermimic/data/cfg/omomo_teacher_kfold{i}_src2_mlp.yaml")))
             ["env"]["subjectBodies"] if int(b[3:]) >= 100}
            for i in range(1, 4)])
        assert kept_everywhere < all_syn, "leak guard never fired -- suspicious"
    print("ok: generated files (bodies, leak guard, names, slurm eval line)")


if __name__ == "__main__":
    test_fold_invariants()
    test_generated_files()
    print("ALL GREEN")

#!/usr/bin/env python3
"""Tests for scripts/generate_gen2_arms.py -- the 8 methods x 2 folds grid.
Each cell must carry exactly its own axes; these are the properties that would
silently corrupt the sweep: a test body left in training, a leaky synthetic
retained, a transformer flag on an MLP arm, a recipe knob crossing cells, or a
non-uniform env count (batch confound).

Run:  python tests/test_generate_gen2_arms.py   (exit 0 = all green)
"""
import os
import sys
import tempfile

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import generate_gen2_arms as g2  # noqa: E402
from generate_kfold_cfgs import beta_dist, real_human_floor  # noqa: E402

raw = np.load(g2.BETAS)
BETAS = {k: raw[k] for k in raw.files if k.startswith("sub")}
FLOOR = real_human_floor(BETAS)


def main():
    assert abs(FLOOR - 2.106) < 0.01, FLOOR          # pin the calibration
    with tempfile.TemporaryDirectory() as tmp:
        written = g2.main(["--out-root", tmp])
        assert len(written) == 48, len(written)      # 16 cells x (env, train, slurm)
        names = set()
        roster = {}      # the shared synthetic roster, pinned by the first cell
        for name, arch, refs, recipe, fold in g2.cells():
            cfgd = os.path.join(tmp, "isaacgym/src/intermimic/data/cfg")
            env = yaml.safe_load(open(os.path.join(cfgd, f"omomo_teacher_{name}.yaml")))["env"]
            tr = yaml.safe_load(open(os.path.join(cfgd, f"train/rlg/omomo_teacher_{name}.yaml")))
            cfgtr = tr["params"]["config"]
            slurm = open(os.path.join(tmp, f"slurm_teacher_{name}.sh")).read()
            test = g2.FOLDS[fold]
            bodies = env["subjectBodies"]

            # fold axis: no test body trains; sub4 never trains; 13 reals; and
            # every SYNTHETIC clears the floor vs EVERY fold's trio (shared
            # roster = identical synthetics and equal size in both folds).
            assert not set(test) & set(bodies), name
            assert "sub4" not in bodies, name
            reals = [b for b in bodies if int(b[3:]) < 100]
            syns = frozenset(b for b in bodies if int(b[3:]) >= 100)
            assert len(reals) == 13, (name, len(reals))
            for s in syns:
                for f2, trio in g2.FOLDS.items():
                    dmin = min(beta_dist(BETAS, s, t) for t in trio)
                    assert dmin >= FLOOR - 1e-9, f"{name}: {s} at {dmin:.3f} < floor vs f{f2}"
            roster.setdefault("syn", syns)
            assert syns == roster["syn"], f"{name}: synthetic roster differs between cells"
            assert len(bodies) == 43, (name, len(bodies))

            # refs axis
            if refs == "ret":
                assert env.get("retargetedMotionDir") and env.get("cpuMotionData") is True, name
                assert "max_split_size_mb" in slurm, name
            else:
                assert "retargetedMotionDir" not in env, name
                assert "max_split_size_mb" not in slurm, name

            # arch axis
            net = tr["params"]["network"]["name"]
            if arch == "xf":
                assert env.get("useTransformerObs") is True and env["numObs"] == 6524, name
                assert net == "intermimic_transformer", (name, net)
            else:
                assert "useTransformerObs" not in env and env["numObs"] == 3230, name
                assert net == "intermimic", (name, net)

            # recipe axis
            if recipe == "nvadlr":
                assert cfgtr["normalize_value"] is True and cfgtr["lr_schedule"] == "adaptive"
                assert abs(cfgtr["kl_threshold"] - 0.06) < 1e-9, name
            else:
                assert cfgtr["normalize_value"] is False and cfgtr["lr_schedule"] == "constant"
                assert "kl_threshold" not in cfgtr, name

            # uniform batch: every cell defaults to 2048 envs; 24h walltime
            assert 'NUM_ENVS="${NUM_ENVS:-2048}"' in slurm, name
            assert "--time=24:00:00" in slurm and "REWARD_BREAKDOWN=1" in slurm
            exp = cfgtr["full_experiment_name"]
            assert exp == f"smplx_teacher_{name}" and exp not in names
            names.add(exp)
        assert len(names) == 16
    print("ALL GREEN: 16 cells, axes independent, floor-clean folds, uniform 2048 envs")


if __name__ == "__main__":
    main()

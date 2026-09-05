#!/usr/bin/env python3
"""Generate the synthetic-body-count ladder: ret_stock fold0 cells that differ
ONLY in subjectBodies.

    syn0    13 reals                                  (does the current 30 help at all?)
    syn30   = the existing g2_mlp_ret_stock__f0      (already trained -- no new cell)
    syn60   43-roster + sub140-169                    (+30 out-of-hull, strict spacing)
    syn130  43-roster + sub140-239                    (+100)

New bodies come from synthetic_bodies_neutral_v2.npz: out-of-hull, proportion-
banded, >=2.106 from every fold0 test body AND from every other training body
(see generate_synthetic_bodies.py --min-pairwise-dist / --frac-extrap-dir).

Each cell = env cfg + rlg train cfg + slurm script, cloned from the trained
g2_mlp_ret_stock__f0 cell so the recipe/envs/PSI/lowbuf/cpuMotion setup is
byte-identical -- subjectBodies, betas_file (aug2 superset) and
subjectHeightsFile (v2, covers sub140+) are the only functional edits.

Before TRAINING syn60/syn130 the cluster needs, for sub140-239:
  1. MJCFs:            scripts/generate_per_subject_mjcfs.py (gender=neutral)
  2. retargeted refs:  SRC=sub2 TARGETS='sub140 ... ' OUT=InterAct/OMOMO_retarget_contact_src2 \
                           sbatch slurm_retarget_gen.sh
     (additive: writes new per-body files beside the existing ones; syn0 needs nothing)
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"
BASE = "g2_mlp_ret_stock__f0"

REALS13 = ['sub1', 'sub2', 'sub3', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9',
           'sub11', 'sub12', 'sub14', 'sub15', 'sub17']
G2_SYN30 = ['sub100', 'sub101', 'sub102', 'sub103', 'sub104', 'sub105', 'sub106',
            'sub107', 'sub108', 'sub110', 'sub111', 'sub112', 'sub113', 'sub114',
            'sub115', 'sub116', 'sub117', 'sub118', 'sub119', 'sub120', 'sub124',
            'sub127', 'sub128', 'sub130', 'sub131', 'sub132', 'sub133', 'sub136',
            'sub137', 'sub139']
NEW30 = [f"sub{i}" for i in range(140, 170)]
NEW100 = [f"sub{i}" for i in range(140, 240)]

CELLS = {
    "syn0":   REALS13,
    "syn60":  REALS13 + G2_SYN30 + NEW30,
    "syn130": REALS13 + G2_SYN30 + NEW100,
}


def main():
    for cell, roster in CELLS.items():
        name = f"g2_mlp_ret_stock_{cell}__f0"
        body_line = "  subjectBodies: [" + ", ".join(f"'{b}'" for b in roster) + "]"

        env = (CFG / f"omomo_teacher_{BASE}.yaml").read_text()
        # Replace the subjectBodies KEY *and the block-style items under it*.
        #
        # This used to be r"^  subjectBodies:.*$", which matches only the key line.
        # The base cfg writes the roster as a block list --
        #     subjectBodies:
        #     - sub1
        #     ... 43 items
        # -- so swapping the key line for a flow-style one left 43 orphaned
        # `- subN` entries dangling under it, and all three generated cells failed
        # yaml.safe_load outright. They were committed broken and could never have
        # launched. Worse, tests/test_generate_synladder_cfgs.py runs this script,
        # so every `pytest tests/` rewrote the corruption back into the repo --
        # repairing the output files could never stick while this line was wrong.
        env, n = re.subn(r"^  subjectBodies:.*(?:\n  - .*)*$", body_line,
                         env, count=1, flags=re.M)
        assert n == 1
        # Belt and braces: the whole point is that the result PARSES. Assert it
        # here rather than discovering it at sbatch time.
        import yaml as _yaml
        _bodies = _yaml.safe_load(env)["env"]["subjectBodies"]
        assert _bodies == roster, (
            f"{cell}: emitted roster {len(_bodies)} bodies, expected {len(roster)}")
        env, n = re.subn(r"betas_file:.*", "betas_file: scripts/omomo_betas_neutral_aug2.npz", env, count=1)
        assert n == 1
        env, n = re.subn(r"subjectHeightsFile:.*",
                         "subjectHeightsFile: scripts/synthetic_heights_v2.json", env, count=1)
        assert n == 1
        env = (f"# SYN-LADDER cell {cell}: {len(roster)} bodies "
               f"({len([b for b in roster if int(b[3:]) < 100])} reals + "
               f"{len([b for b in roster if int(b[3:]) >= 100])} synthetic). Differs from the "
               f"trained {BASE} ONLY in subjectBodies/betas_file/heights.\n") + env
        (CFG / f"omomo_teacher_{name}.yaml").write_text(env)

        train = (CFG / f"train/rlg/omomo_teacher_{BASE}.yaml").read_text()
        train, n = re.subn(r"full_experiment_name:\s*\S+",
                           f"full_experiment_name: smplx_teacher_{name}", train, count=1)
        assert n == 1
        (CFG / f"train/rlg/omomo_teacher_{name}.yaml").write_text(train)

        slurm = (REPO / f"slurm_teacher_{BASE}.sh").read_text().replace(BASE, name)
        out = REPO / f"slurm_teacher_{name}.sh"
        out.write_text(slurm)
        print(f"{name}: {len(roster)} bodies -> env/train cfgs + {out.name}")


if __name__ == "__main__":
    main()

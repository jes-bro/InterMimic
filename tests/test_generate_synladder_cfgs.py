"""Checks on the generated syn-ladder cells: roster sizes, nesting, the three
functional edits, and that nothing else drifted from the trained base cell."""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "isaacgym/src/intermimic/data/cfg"


def setup_module():
    subprocess.run([sys.executable, str(REPO / "scripts/generate_synladder_cfgs.py")], check=True)


def bodies(name):
    txt = (CFG / f"omomo_teacher_{name}.yaml").read_text()
    line = next(l for l in txt.splitlines() if l.strip().startswith("subjectBodies:"))
    return re.findall(r"'(sub\d+)'", line)


def test_roster_sizes_and_nesting():
    b0, b60, b130 = (bodies(f"g2_mlp_ret_stock_{c}__f0") for c in ("syn0", "syn60", "syn130"))
    assert (len(b0), len(b60), len(b130)) == (13, 73, 143)
    assert set(b0) < set(b60) < set(b130)          # strictly nested ladder
    base = bodies("g2_mlp_ret_stock__f0")
    assert set(base) < set(b60)                    # the trained 43-roster is the syn30 rung
    for trio in ("sub10", "sub13", "sub16"):       # fold0 test bodies never in training
        assert all(trio not in b for b in (b0, b60, b130))


def test_functional_edits_and_no_drift():
    base = (CFG / "omomo_teacher_g2_mlp_ret_stock__f0.yaml").read_text()
    for cell in ("syn0", "syn60", "syn130"):
        txt = (CFG / f"omomo_teacher_g2_mlp_ret_stock_{cell}__f0.yaml").read_text()
        assert "omomo_betas_neutral_aug2.npz" in txt
        assert "synthetic_heights_v2.json" in txt
        assert "retargetedMotionDir: InterAct/OMOMO_retarget_contact_src2" in txt
        assert "cpuMotionData: true" in txt
        # every line outside the 4 intended edits matches the base cell exactly
        skip = ("subjectBodies:", "betas_file:", "subjectHeightsFile:", "# SYN-LADDER")
        left = [l for l in txt.splitlines() if not any(k in l for k in skip)]
        right = [l for l in base.splitlines() if not any(k in l for k in skip)]
        assert left == right, f"{cell}: unexpected drift from base cfg"


def test_train_cfg_and_slurm():
    for cell in ("syn0", "syn60", "syn130"):
        name = f"g2_mlp_ret_stock_{cell}__f0"
        train = (CFG / f"train/rlg/omomo_teacher_{name}.yaml").read_text()
        assert f"full_experiment_name: smplx_teacher_{name}" in train
        assert "resume_from: 'None'" in train      # fresh start, never a warm start
        slurm = (REPO / f"slurm_teacher_{name}.sh").read_text()
        assert f"omomo_teacher_{name}.yaml" in slurm
        assert "g2_mlp_ret_stock__f0.yaml" not in slurm   # no dangling base references

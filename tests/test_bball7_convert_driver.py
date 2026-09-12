#!/usr/bin/env python3
"""scripts/slurm_cari4d_bball7_convert.sh against STUB pipeline steps.

The driver's job is orchestration: call the per-clip wrapper for every manifest
row, keep partial results out of the dataset, resume without redoing finished
work, and run the two relabels only on a complete set. Each of those is a way to
silently ship a wrong dataset (the audit's HIGH-2 / MEDIUM-4 / MEDIUM-5), so
they are exercised here with a fake INTERMIMIC tree whose scripts/ are stubs
that record their calls and write the files the real ones would.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_bball7_convert_driver.py -v
"""
import os
import stat
import subprocess
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(REPO, "scripts", "slurm_cari4d_bball7_convert.sh")

MANIFEST = textwrap.dedent("""\
    clip,subject,subject_id,clip_idx,object,take,gender,n_frames,lo,hi,export,bundle,mesh
    Date03_Sub01_bball_rev003b,Sub01,401,000,bballd03s01rev003b,t,male,68,232,299,x.tar.gz,a/a.pth,a/a.obj
    Date03_Sub01_bball_rev009c,Sub01,401,001,bballd03s01rev009c,t,male,70,529,598,y.tar.gz,b/b.pth,b/b.obj
    Date08_Sub12_bball_t012a,Sub12,412,000,bballd08s12t012a,t,male,118,1042,1159,z.tar.gz,c/c.pth,c/c.obj
    """)


def _write_exec(path, body):
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def make_tree(tmp_path, wrapper_behaviour="ok"):
    """A fake InterMimic checkout: stub wrapper + stub relabels + assets."""
    root = tmp_path / "InterMimic"
    (root / "scripts").mkdir(parents=True)
    (root / "InterAct").mkdir()
    assets = root / "isaacgym/src/intermimic/data/assets"
    (assets / "smplx").mkdir(parents=True)
    (assets / "objects/objects").mkdir(parents=True)
    for sid in (401, 412):
        (assets / "smplx" / f"smplh_behave_sub{sid}.xml").write_text("<mujoco/>")
    bundles = tmp_path / "bundles"
    for d in ("a", "b", "c"):
        (bundles / d).mkdir(parents=True)
        (bundles / d / f"{d}.pth").write_bytes(b"x")
        (bundles / d / f"{d}.obj").write_text("v 0 0 0\n")
    (bundles / "manifest.csv").write_text(MANIFEST)
    (root / "scripts" / "betas.npz").write_bytes(b"npz")
    log = root / "calls.log"
    # Stub wrapper: records its env, writes the .pt (step 3), then either
    # succeeds, fails AFTER writing (the upside-down case), or eats stdin.
    _write_exec(root / "scripts" / "slurm_cari4d_to_mimic.sh", textwrap.dedent(f"""\
        #!/bin/bash
        echo "wrapper SUBJECT_ID=$SUBJECT_ID OBJECT_NAME=$OBJECT_NAME CLIP_IDX=$CLIP_IDX ROTATE_AXIS=$ROTATE_AXIS REPLAY=$REPLAY BETAS_NPZ=$BETAS_NPZ BUNDLE=$BUNDLE INTERMIMIC=$INTERMIMIC" >> "{log}"
        mkdir -p "$INTERMIMIC/InterAct/$DATASET_TAG"
        echo pt > "$INTERMIMIC/InterAct/$DATASET_TAG/sub${{SUBJECT_ID}}_${{OBJECT_NAME}}_${{CLIP_IDX}}.pt"
        case "{wrapper_behaviour}" in
          fail_after_install) [ "$CLIP_IDX" = "001" ] && exit 3 ;;
          eat_stdin) cat > /dev/null ;;
        esac
        exit 0
        """))
    # Stub relabels: copy src -> dst clip by clip, like the real ones.
    for name in ("relabel_contact_flags.py", "relabel_contact_human.py"):
        _write_exec(root / "scripts" / name, textwrap.dedent(f"""\
            #!/usr/bin/env python3
            import sys, shutil, os
            a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
            open("{log}", "a").write("{name} " + " ".join(sys.argv[1:]) + "\\n")
            if os.path.exists(a["--dst-dir"]): sys.exit("dst exists")
            os.makedirs(a["--dst-dir"])
            for f in sorted(os.listdir(a["--src-dir"])):
                if f.endswith(".pt"): shutil.copy(os.path.join(a["--src-dir"], f), a["--dst-dir"])
            """))
    return root, bundles, log


def run(root, bundles, **env):
    e = dict(os.environ, MANIFEST=str(bundles / "manifest.csv"), BUNDLES_ROOT=str(bundles),
             INTERMIMIC=str(root), BETAS_NPZ=str(root / "scripts" / "betas.npz"),
             DATASET_TAG="behave_test")
    e.update(env)                                   # per-test overrides win
    return subprocess.run(["bash", DRIVER], env=e, capture_output=True, text=True)


def test_happy_path_converts_all_and_relabels(tmp_path):
    root, bundles, log = make_tree(tmp_path)
    r = run(root, bundles)
    assert r.returncode == 0, r.stdout + r.stderr
    pts = sorted(os.listdir(root / "InterAct/behave_test"))
    assert pts == ["sub401_bballd03s01rev003b_000.pt", "sub401_bballd03s01rev009c_001.pt",
                   "sub412_bballd08s12t012a_000.pt"]
    calls = log.read_text()
    assert calls.count("wrapper ") == 3
    assert "ROTATE_AXIS=x REPLAY=0" in calls and "BETAS_NPZ=" in calls
    assert f"INTERMIMIC={root}" in calls
    assert "relabel_contact_flags.py" in calls and "--ball-radius-from-mesh" in calls
    assert len(os.listdir(root / "InterAct/behave_test_cf2")) == 3
    assert "converted 3, skipped 0, of 3" in r.stdout


def test_wrapper_failure_after_install_removes_partial_and_stops(tmp_path):
    """HIGH-2: the .pt exists but the rotate step failed -- must NOT be kept."""
    root, bundles, log = make_tree(tmp_path, "fail_after_install")
    r = run(root, bundles)
    assert r.returncode == 1
    assert "FAILED -- removing partial" in r.stderr
    pts = sorted(os.listdir(root / "InterAct/behave_test"))
    assert pts == ["sub401_bballd03s01rev003b_000.pt"]          # the failed 001 is gone
    assert not (root / "InterAct/behave_test_cf").exists()      # no relabel on a partial set


def test_resume_skips_done_and_does_not_lose_rows_to_stdin(tmp_path):
    """MEDIUM-4/5: a child eating stdin must not shrink the row count, and a
    partial relabel dir from a killed run must be rebuilt, not accepted."""
    root, bundles, log = make_tree(tmp_path, "eat_stdin")
    (root / "InterAct/behave_test").mkdir()
    (root / "InterAct/behave_test/sub401_bballd03s01rev003b_000.pt").write_text("done")
    (root / "InterAct/behave_test_cf").mkdir()                   # partial: 0 of 3 clips
    (root / "InterAct/behave_test_cf/stale.txt").write_text("")
    r = run(root, bundles)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log.read_text().count("wrapper ") == 2               # only the two missing clips
    assert "converted 2, skipped 1, of 3" in r.stdout
    assert "is partial -- rebuilding" in r.stdout
    assert sorted(os.listdir(root / "InterAct/behave_test_cf")) == [
        "sub401_bballd03s01rev003b_000.pt", "sub401_bballd03s01rev009c_001.pt",
        "sub412_bballd08s12t012a_000.pt"]


def test_refuses_without_inputs(tmp_path):
    root, bundles, log = make_tree(tmp_path)
    r = run(root, bundles, BETAS_NPZ=str(root / "nope.npz"))
    assert r.returncode == 1 and "no shared betas" in r.stderr

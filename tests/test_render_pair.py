"""render_pair.sh: the plan it prints, and the guards, via DRY=1.

A render that runs but shows an empty court, or tracks a different clip than the
camera frames, wastes the GPU and looks like a method failure. What is pinned:

  * the clip framed is the FIRST clip of that source under the body's own
    reference tree -- the one env 0 actually plays
  * the tree comes from the eval config, not a guess
  * a missing checkpoint / config / pair fails before Isaac Gym starts
  * ARM selects the arm's own eval+train configs, so a video matches its numbers
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("scripts", "render_pair.sh")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(os.path.join(REPO, SCRIPT), tmp_path / SCRIPT)
    # a cam_for_clip stub: the real one needs torch and a real clip
    (tmp_path / "scripts" / "cam_for_clip.py").write_text(
        'print("export RECORD_VIDEO_CAM_POS=1,2,3")\n'
        'print("export RECORD_VIDEO_CAM_TARGET=0,0,1")\n')
    cfg = tmp_path / "isaacgym/src/intermimic/data/cfg"
    (cfg / "train/rlg").mkdir(parents=True)
    (cfg / "omomo_eval_arm__f0.yaml").write_text(
        "env:\n  retargetedMotionDir: InterAct/tree\n")
    (cfg / "train/rlg/omomo_arm__f0.yaml").write_text("params: {}\n")
    (tmp_path / "InterAct/tree/sub16").mkdir(parents=True)
    for n in ["sub458_ball_000.pt", "sub458_ball_001.pt", "sub401_ball_000.pt"]:
        (tmp_path / "InterAct/tree/sub16" / n).write_text("")
    ck = tmp_path / "checkpoints/x/nn"
    ck.mkdir(parents=True)
    (ck / "mimic.pth").write_text("")
    return tmp_path


def run(repo, **env):
    e = dict(os.environ)
    e.update(DRY="1", BODY="sub16", SOURCE="sub458", OUT="/tmp/o.mp4",
             ARM="arm__f0", CKPT="checkpoints/x/nn/mimic.pth")
    e.update(env)                                   # caller's values win
    return subprocess.run(["sh", SCRIPT], cwd=repo, env=e, capture_output=True, text=True)


def test_frames_the_first_clip_of_that_source(repo):
    out = run(repo).stdout
    assert "InterAct/tree/sub16/sub458_ball_000.pt" in out
    assert "sub401" not in out                      # another source's clip


def test_reads_the_tree_from_the_eval_cfg(repo):
    out = run(repo).stdout
    assert "clip (env0): InterAct/tree/" in out


def test_camera_comes_from_cam_for_clip(repo):
    out = run(repo).stdout
    assert "RECORD_VIDEO_CAM_POS=1,2,3" in out


def test_arm_selects_its_own_cfgs(repo):
    out = run(repo).stdout
    assert "omomo_eval_arm__f0.yaml" in out


def test_missing_pair_fails_loudly(repo):
    r = run(repo, SOURCE="sub999")
    assert r.returncode == 2 and "no sub999_*.pt" in r.stderr


def test_missing_checkpoint_fails_loudly(repo):
    r = run(repo, CKPT="checkpoints/nope.pth")
    assert r.returncode == 2 and "missing checkpoints/nope.pth" in r.stderr


def test_baseline_mode_without_arm(repo):
    out = run(repo, ARM="",
              ENV_YAML="isaacgym/src/intermimic/data/cfg/omomo_eval_arm__f0.yaml",
              TRAIN_YAML="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_arm__f0.yaml",
              TASK="InterMimic_All").stdout
    assert "InterMimic_All" in out


def test_dump_is_optional(repo):
    assert "-> dump:" not in run(repo).stdout
    assert "-> dump: /tmp/d.npz" in run(repo, DUMP="/tmp/d.npz").stdout

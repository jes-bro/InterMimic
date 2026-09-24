"""render_pair.sh: the plan it prints, and the guards, via DRY=1.

A render that runs but shows an empty court, or tracks a different clip than the
camera frames, wastes the GPU and looks like a method failure. What is pinned:

  * the clip framed is the FIRST clip of that source under the body's own
    reference tree -- the one env 0 actually plays
  * the tree comes from the eval config, not a guess
  * a missing checkpoint / config / pair fails before Isaac Gym starts
  * ARM selects the arm's own eval+train configs, so a video matches its numbers

Also pinned: ONE clip per video. An env that resets moves to another clip, so an
unpinned render is a montage of half-attempts at different motions -- which is
what the first demo batch turned out to be. The render points --motion_file at a
staging dir holding exactly that clip, so a reset re-attempts THE SAME motion.
The video is deliberately NOT shortened to the clip's length: seeing the retries
is wanted.
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
        "env:\n  motion_file: InterAct/src\n  retargetedMotionDir: InterAct/tree\n")
    (cfg / "train/rlg/omomo_arm__f0.yaml").write_text("params: {}\n")
    (tmp_path / "InterAct/tree/sub16").mkdir(parents=True)
    (tmp_path / "InterAct/src").mkdir(parents=True)
    for n in ["sub458_ball_000.pt", "sub458_ball_001.pt", "sub401_ball_000.pt"]:
        (tmp_path / "InterAct/tree/sub16" / n).write_text("")
        # the SOURCE clips the staging dir links to; the tree holds the per-body
        # retargeted reference of the same basename
        (tmp_path / "InterAct/src" / n).write_text("")
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


# --- one clip per video -------------------------------------------------------

def test_plan_names_the_pinned_clip(repo):
    out = run(repo).stdout
    assert "pinned to  : sub458_ball_000.pt" in out


def test_clip_selects_that_clip_and_frames_it(repo):
    """CLIP= must move BOTH what plays and what the camera frames; moving only
    one of them is the failure that renders an empty court."""
    out = run(repo, CLIP="sub458_ball_001.pt").stdout
    assert "pinned to  : sub458_ball_001.pt" in out
    assert "clip (env0): InterAct/tree/sub16/sub458_ball_001.pt" in out


def test_unknown_clip_fails_before_the_gpu(repo):
    r = run(repo, CLIP="sub458_ball_099.pt")
    assert r.returncode == 2 and "CLIP=sub458_ball_099.pt not found" in r.stderr


def test_clip_present_in_tree_but_not_in_source_dir_fails_loudly(repo):
    """The staging dir links the SOURCE clip; a tree-only basename would
    otherwise produce an empty staging dir and an env with no motions."""
    (repo / "InterAct/tree/sub16/sub458_ball_007.pt").write_text("")
    r = run(repo, CLIP="sub458_ball_007.pt")
    assert r.returncode == 2 and "not in the source motion dir" in r.stderr


def test_missing_motion_file_key_fails_loudly(repo):
    cfg = repo / "isaacgym/src/intermimic/data/cfg/omomo_eval_arm__f0.yaml"
    cfg.write_text("env:\n  retargetedMotionDir: InterAct/tree\n")
    r = run(repo)
    assert r.returncode == 2 and "no motion_file in" in r.stderr


def test_list_shows_only_that_pairs_clips_and_runs_nothing(repo):
    r = run(repo, LIST="1")
    assert r.returncode == 0
    assert "sub458_ball_000.pt" in r.stdout and "sub458_ball_001.pt" in r.stdout
    assert "sub401" not in r.stdout                 # another source
    assert "camera" not in r.stdout                 # listing, not a plan


def test_staging_dir_holds_exactly_one_clip(repo):
    """The real guard. DRY=1 exits before staging, so this runs for real -- the
    python launch then fails (no Isaac Gym here), which is fine: staging happens
    first and is what is being checked."""
    run(repo, DRY="0", CLIP="sub458_ball_001.pt")
    stage = repo / ".render_clip/sub16__sub458_ball_001"
    staged = sorted(p.name for p in stage.iterdir())
    assert staged == ["sub458_ball_001.pt"]
    assert (stage / "sub458_ball_001.pt").is_symlink()


def test_staging_dir_is_rebuilt_not_added_to(repo):
    """A stale link from an earlier CLIP= would quietly restore the montage."""
    stage = repo / ".render_clip/sub16__sub458_ball_001"
    stage.mkdir(parents=True)
    (stage / "sub458_ball_000.pt").write_text("")    # leftover from a past run
    run(repo, DRY="0", CLIP="sub458_ball_001.pt")
    assert sorted(p.name for p in stage.iterdir()) == ["sub458_ball_001.pt"]


def test_the_staging_dir_is_what_is_handed_to_the_env(repo):
    """Read from the script itself: the staging dir is only effective if it is
    passed as --motion_file, and a plan print cannot show that."""
    src = (repo / "scripts/render_pair.sh").read_text()
    assert '--motion_file "$STAGE"' in src

#!/usr/bin/env python3
"""utils/rollout_pick.RolloutRecorder: the best-of-N selection behind DUMP_PICK=best.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_rollout_pick.py -q
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
PKG = os.path.join(ROOT, "isaacgym", "src", "intermimic")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


rp = _load(os.path.join(PKG, "utils", "rollout_pick.py"), "rollout_pick")


def _frame(n, t):
    """Distinguishable per-env, per-step content: body_pos[e, :, 0] == e + t/1000."""
    body_rot = np.zeros((n, 52, 4), np.float32); body_rot[..., 3] = 1
    body_pos = np.zeros((n, 52, 3), np.float32); body_pos[:, :, 0] = (np.arange(n) + t / 1000.0)[:, None]
    obj_pos = np.full((n, 3), t, np.float32)
    obj_rot = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    return body_rot, body_pos, obj_pos, obj_rot


def _run(n, cap, done_at, terminated):
    """Simulate: env e ends at step done_at[e] (None = never within cap); terminated[e]
    says whether that end was a failure."""
    r = rp.RolloutRecorder(n, cap)
    t = 0
    while not r.all_ended() and not r.full():
        r.record(*_frame(n, t))
        done = np.array([done_at[e] is not None and done_at[e] == t for e in range(n)])
        term = np.array([bool(terminated[e]) for e in range(n)])
        r.step_done(done, term)
        t += 1
    return r


def test_success_beats_survival_and_lowest_index_wins():
    # env 0 fails at step 5, env 1 reaches the end at 9, env 2 reaches the end at 9, env 3 survives longest but fails at 20
    r = _run(4, 100, [5, 9, 9, 20], [True, False, False, True])
    e, why = r.pick()
    assert e == 1 and "reached the clip's last frame (2 of 4 envs did)" in why
    assert r.n_frames.tolist() == [6, 10, 10, 21] and r.succeeded.tolist() == [False, True, True, False]
    assert r.all_ended()


def test_no_success_keeps_the_farthest_survivor():
    r = _run(3, 100, [4, 12, 7], [True, True, True])
    e, why = r.pick()
    assert e == 1 and "NO env reached the clip's end" in why and "13 frames" in why
    s = r.summary()
    assert s["chosen_succeeded"] is False and s["n_succeeded"] == 0 and s["chosen_frames"] == 13


def test_arrays_are_the_chosen_envs_frames_before_its_first_done():
    r = _run(3, 100, [2, 5, 3], [True, False, True])
    e, _ = r.pick()
    assert e == 1
    a = r.arrays_for(e)
    assert a["body_pos"].shape == (6, 52, 3) and a["body_rot"].shape == (6, 52, 4)
    assert a["obj_pos"].shape == (6, 3) and a["obj_rot"].shape == (6, 4)
    np.testing.assert_allclose(a["body_pos"][:, 0, 0], 1 + np.arange(6) / 1000.0, rtol=1e-5)   # env 1's rows
    assert a["obj_pos"][:, 0].tolist() == list(range(6))                                     # steps 0..5


def test_cap_stops_recording_and_unended_envs_rank_by_length():
    r = _run(2, 10, [None, 3], [False, True])
    assert r.full() and not r.all_ended()
    assert r.n_frames.tolist() == [10, 4]
    e, why = r.pick()
    assert e == 0 and "farthest survivor, 10 frames" in why


def test_no_termination_info_means_no_success_only_length():
    r = rp.RolloutRecorder(2, 10)
    for t in range(4):
        r.record(*_frame(2, t))
        r.step_done(np.array([t == 1, t == 3]), None)
    assert r.succeeded.tolist() == [False, False] and r.pick()[0] == 1


def test_an_env_that_ends_twice_is_counted_once():
    r = rp.RolloutRecorder(2, 10)
    r.record(*_frame(2, 0)); assert r.step_done(np.array([True, False]), np.array([False, False])) == 1
    r.record(*_frame(2, 1)); assert r.step_done(np.array([True, False]), np.array([True, False])) == 0
    assert r.n_frames.tolist() == [1, 2] and r.succeeded.tolist() == [True, False]


def test_shape_mismatches_are_refused():
    r = rp.RolloutRecorder(3, 10)
    with pytest.raises(ValueError, match="leading axis 2 != num_envs 3"):
        r.record(*_frame(2, 0))
    with pytest.raises(ValueError, match="done: 2 != num_envs 3"):
        r.step_done(np.array([True, False]))


def test_player_wires_dump_pick_best(tmp_path):
    s = open(os.path.join(PKG, "learning", "intermimic_players.py")).read()
    assert 'os.environ.get("DUMP_PICK", "env0")' in s
    assert "RolloutRecorder(self.env.task.num_envs, cap=_dump_max)" in s
    assert "_best_record(done, info)" in s and "if _best.all_ended() or _best.full():" in s
    assert 'term = info.get("terminate")' in s                    # success = done without termination
    assert "chosen_env=s[\"chosen_env\"]" in s and "frames_per_env" in s
    # the original env-0 path is untouched for DUMP_PICK unset
    assert "elif _traj is not None:\n                        if len(_traj[\"body_rot\"]) < _dump_max:" in s


def test_task_teacher_skip_flag_is_test_only():
    s = open(os.path.join(PKG, "env", "tasks", "intermimic_distill_g3.py")).read()
    assert 'os.environ.get("DISTILL_SKIP_TEACHERS", "0") == "1"' in s
    assert "is for --test inference only" in s
    assert s.count("if self._g3_ready and not self._skip_teachers:") == 2      # post_physics_step + reset
    assert "self._g3_ready = True\n            return" in s                    # skips the teacher block entirely
    g4 = open(os.path.join(PKG, "env", "tasks", "intermimic_distill_g4.py")).read()
    assert "_skip_teachers" not in g4                                          # inherited, not re-implemented

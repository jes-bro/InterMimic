"""Best-of-N rollout recording for DUMP_TRAJ (no torch / Isaac Gym imports, so
tests/test_rollout_pick.py runs anywhere).

DUMP_TRAJ records env 0 for a fixed number of frames whether or not that env
succeeded. For a qualitative rollout that is a coin flip. DUMP_PICK=best
(intermimic_players.py) runs N envs on the SAME pinned clip, records every env
until its episode ends, and keeps ONE:

  1. an env that reached the clip's last frame without a termination
     (the eval's own success criterion: reset by progress, not by failure);
     the lowest-index such env, so the choice is reproducible;
  2. otherwise the env that survived the most frames (Jess, 2026-10-14: never
     fail loudly here -- "use the best run out of what we have").

An env's record stops at its first done: the player resets done envs on the
next step, and frames after that belong to a different attempt.
"""
import numpy as np


class RolloutRecorder:
    """Per-env frame buffers plus the done/success bookkeeping that picks one."""

    def __init__(self, num_envs, cap):
        self.num_envs = int(num_envs)
        self.cap = int(cap)
        self.n_frames = np.zeros(self.num_envs, dtype=np.int64)
        self.ended = np.zeros(self.num_envs, dtype=bool)
        self.succeeded = np.zeros(self.num_envs, dtype=bool)
        self.frames = {"body_rot": [], "body_pos": [], "obj_pos": [], "obj_rot": []}
        self.n_steps = 0

    def record(self, body_rot, body_pos, obj_pos, obj_rot):
        """Append one step for ALL envs (arrays with a leading num_envs axis).
        Ended envs' slots are stored too (cheap) but never read back."""
        for k, v in (("body_rot", body_rot), ("body_pos", body_pos), ("obj_pos", obj_pos), ("obj_rot", obj_rot)):
            v = np.asarray(v, dtype=np.float32)
            if v.shape[0] != self.num_envs:
                raise ValueError(f"{k}: leading axis {v.shape[0]} != num_envs {self.num_envs}")
            self.frames[k].append(v.copy())
        self.n_frames[~self.ended] += 1
        self.n_steps += 1

    def step_done(self, done, terminated=None):
        """Mark envs whose episode ended THIS step. `terminated` (same shape) says
        the episode ended by failure; an env done WITHOUT termination reached the
        clip's end = success. None = no termination info: nothing counts as a
        success, only survival length ranks."""
        done = np.asarray(done, dtype=bool).reshape(-1)
        if done.shape[0] != self.num_envs:
            raise ValueError(f"done: {done.shape[0]} != num_envs {self.num_envs}")
        new = done & ~self.ended
        if terminated is not None:
            terminated = np.asarray(terminated, dtype=bool).reshape(-1)
            self.succeeded[new & ~terminated] = True
        self.ended |= new
        return int(new.sum())

    def all_ended(self):
        return bool(self.ended.all())

    def full(self):
        return self.n_steps >= self.cap

    def pick(self):
        """(env index, reason). Success beats survival; ties by lowest index."""
        if self.succeeded.any():
            e = int(np.flatnonzero(self.succeeded)[0])
            return e, f"reached the clip's last frame ({int(self.succeeded.sum())} of {self.num_envs} envs did)"
        e = int(np.argmax(self.n_frames))
        return e, (f"NO env reached the clip's end; farthest survivor, {int(self.n_frames[e])} frames "
                   f"(max over {self.num_envs} envs)")

    def arrays_for(self, e):
        """The chosen env's frames, trimmed to the frames before its first done."""
        n = int(self.n_frames[e])
        return {k: np.stack([f[e] for f in v[:n]], axis=0) for k, v in self.frames.items()}

    def summary(self):
        e, why = self.pick()
        return {
            "chosen_env": e, "reason": why, "chosen_frames": int(self.n_frames[e]),
            "chosen_succeeded": bool(self.succeeded[e]), "num_envs": self.num_envs,
            "n_succeeded": int(self.succeeded.sum()), "n_ended": int(self.ended.sum()),
            "frames_per_env": self.n_frames.copy(), "succeeded_per_env": self.succeeded.copy(),
        }

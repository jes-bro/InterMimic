"""Takeover eval: let a DEGRADED policy wander for k steps, then hand the
teacher the wheel and see whether it can still finish the clip.

WHY THIS EXISTS. Distillation trains the student with
`(student_mu - teacher_mu)**2` (intermimic_agent_distill.py:215) over states
visited during the rollout, and the DAgger beta
(`beta_t = max(0.7 - max((epoch-1500)/1000, 0), 0)`) decays to 0 by ~epoch 2200.
After that the STUDENT drives and the teacher only labels. So the property that
decides whether a teacher is a good teacher is the quality of its action mean on
states IT DID NOT CHOOSE -- and the standard eval measures the exact opposite,
because `stateInit: Start` puts the teacher on its own reference trajectory
every episode.

Success rate at k=0 is that standard eval. The SLOPE as k grows is the new
information: a teacher at 95% for k=0 and 60% at k=100 cannot recover from
states the student will routinely produce; one at 92%/88% can.

THE WANDERER. You do not need a trained student to generate off-reference
states -- which matters, because the point is to pick a teacher BEFORE
distilling. Gaussian noise on the teacher's own action mean produces
off-reference states of roughly the right character for a fraction of the cost.
TAKEOVER_NOISE is the sigma, in action units, added for the first k steps only.

CONFOUND, HANDLED. An env can die DURING the noisy phase, which is the noise
killing it rather than the teacher failing to recover. That would make any
sufficiently large sigma look like a bad teacher. `wander_deaths` counts those
separately so the two are never silently added together; the runner prints the
count and a high one invalidates the comparison rather than quietly biasing it.

k=0 (or TAKEOVER_K unset) MUST be byte-identical to a normal eval -- that is the
acceptance test for the whole instrument, since it is the only way to know the
harness itself did not move the numbers.
"""
import os

import torch


class TakeoverConfig:
    """How long to wander, and how hard.

    Attributes:
        k: steps from each env's own reset during which the DEGRADED policy
            drives. Counted per-env off the task's `progress_buf`, not off a
            global step counter -- envs reset at different times, so a global
            counter would perturb some envs mid-episode and others at their
            start, which is a different experiment on every env.
        noise: standard deviation of the Gaussian added to the teacher's action
            mean while wandering. In action units.
        seed: optional; makes the wander reproducible so two arms can be
            compared under the SAME perturbations rather than two draws.
    """

    def __init__(self, k, noise, seed=None):
        if k < 0:
            raise ValueError(f"TAKEOVER_K must be >= 0, got {k}")
        if noise < 0:
            raise ValueError(f"TAKEOVER_NOISE must be >= 0, got {noise}")
        self.k = int(k)
        self.noise = float(noise)
        self.seed = seed

    def __repr__(self):
        return (f"TakeoverConfig(k={self.k}, noise={self.noise}, "
                f"seed={self.seed})")

    @classmethod
    def from_env(cls, environ=None):
        """Build from TAKEOVER_K / TAKEOVER_NOISE / TAKEOVER_SEED.

        Returns None when the instrument is off, so the caller's fast path is a
        plain `is None` check and an ordinary eval runs code that is byte-
        identical to before this module existed.

        k=0 returns None for the same reason: "wander for zero steps" and "do
        not wander" are the same run, and collapsing them here means the k=0
        acceptance test exercises the real no-op path rather than a k=0 special
        case that merely happens to agree.

        Raises:
            ValueError: if TAKEOVER_K is set but unparseable, or negative. A
                typo'd budget must not silently become an ordinary eval that
                gets filed as a takeover result.
        """
        environ = os.environ if environ is None else environ
        raw = environ.get("TAKEOVER_K")
        if raw is None or raw == "":
            return None
        try:
            k = int(raw)
        except ValueError:
            raise ValueError(
                f"TAKEOVER_K={raw!r} is not an integer. Unset it for a normal "
                f"eval; a typo must not be read as one.")
        if k == 0:
            return None

        raw_noise = environ.get("TAKEOVER_NOISE", "0.1")
        try:
            noise = float(raw_noise)
        except ValueError:
            raise ValueError(f"TAKEOVER_NOISE={raw_noise!r} is not a number")

        raw_seed = environ.get("TAKEOVER_SEED")
        seed = int(raw_seed) if raw_seed not in (None, "") else None
        return cls(k=k, noise=noise, seed=seed)


def wander_mask(progress_buf, cfg):
    """Which envs are still in their degraded phase: progress < k.

    `progress_buf` is the task's per-env step count since that env's last reset,
    so this is per-episode and survives staggered resets.
    """
    return progress_buf < cfg.k


def apply(action, progress_buf, cfg, generator=None):
    """Return `action` with the wander noise added, but only where progress < k.

    Envs past the takeover point are returned BIT-IDENTICAL: the teacher's own
    deterministic action, untouched. That is what makes the second half of the
    episode a clean measurement of the teacher rather than of the teacher plus
    residual perturbation.

    Args:
        action: (num_envs, act_dim) the teacher's deterministic action mean.
        progress_buf: (num_envs,) per-env steps since reset.
        cfg: a TakeoverConfig.
        generator: optional torch.Generator for reproducible draws.

    Returns:
        A new tensor; `action` is not modified in place, because the caller may
        still want the clean action for logging.
    """
    if cfg.noise == 0:
        # A deliberate control: same code path, same masking, zero perturbation.
        # Should reproduce the k=0 numbers, and if it does not, the harness --
        # not the teacher -- is what moved them.
        return action.clone()

    mask = wander_mask(progress_buf, cfg)
    if not bool(mask.any()):
        return action.clone()

    noise = torch.randn(action.shape, device=action.device,
                        dtype=action.dtype, generator=generator) * cfg.noise
    # mask is (num_envs,); action is (num_envs, act_dim) -> broadcast on dim 1.
    return torch.where(mask.unsqueeze(-1), action + noise, action)


def count_wander_deaths(done, was_wandering):
    """How many envs terminated while still being driven by the degraded policy.

    These are NOT teacher failures -- the noise killed them before the teacher
    ever got the wheel -- so they must be reported separately rather than folded
    into the success rate. A large count means the sigma is too high and the
    comparison is measuring the noise, not the teachers.
    """
    if done.dtype != torch.bool:
        done = done != 0
    return int((done & was_wandering).sum().item())

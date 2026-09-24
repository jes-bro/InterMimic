"""_sanitize_teacher: a NaN teacher output must not kill the job.

An env whose observation goes numerically invalid feeds NaN into every teacher,
so Normal(mus, sigma).sample() raises "normal expects all elements of std >= 0.0"
and all 16 envs of an eval die over a handful of bad ones. That cost 4 of the 169
xf@29k in-dist pairs (sub8x sub8, sub12x sub8, sub14x sub8, sub17x sub8 -- every
crash had source sub8), and the teacher action is DISCARDED at eval anyway.

What these pin, in order of what would hurt if it broke:

  * a healthy teacher output is returned BIT-IDENTICAL -- the repair must not be
    able to move a finished run's numbers
  * only the bad entries change; good entries beside them are untouched
  * after the repair, sampling actually works (the point of the exercise)
  * the repair is LOUD once, not silent

HOW THIS IS TESTED. intermimic_distill_g3 imports isaacgym and functorch, which
do not load on the laptop, so the method's SOURCE is extracted from the file with
ast and bound to a bare object. That runs the real code -- if the method is
edited, this test sees the edit -- without importing the simulator.
"""
import ast
import io
import os
from contextlib import redirect_stdout

import pytest

torch = pytest.importorskip("torch")

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "isaacgym", "src", "intermimic", "env", "tasks",
                   "intermimic_distill_g3.py")


class _Bare:
    """Stands in for the task: _sanitize_teacher touches only torch and self."""


@pytest.fixture(scope="module")
def sanitize():
    """The real _sanitize_teacher, bound to a bare object."""
    tree = ast.parse(open(SRC).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "InterMimicDistillG3")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "_sanitize_teacher")
    floor = next(n.value.value for n in cls.body
                 if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", None) == "_TEACHER_SIGMA_FLOOR")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)
    _Bare._TEACHER_SIGMA_FLOOR = floor
    return lambda obj, mu, sd: ns["_sanitize_teacher"](obj, mu, sd)


def _obj():
    return _Bare()


def test_healthy_output_is_returned_unchanged(sanitize):
    """The load-bearing one: no finished run can move."""
    mu = torch.randn(3, 4, 5)
    sd = torch.rand(3, 4, 5) + 0.1
    out_mu, out_sd = sanitize(_obj(), mu.clone(), sd.clone())
    assert torch.equal(out_mu, mu)
    assert torch.equal(out_sd, sd)


def test_healthy_output_prints_nothing(sanitize):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sanitize(_obj(), torch.zeros(2, 2), torch.ones(2, 2))
    assert buf.getvalue() == ""


def test_nan_sigma_becomes_samplable(sanitize):
    """The actual crash: sample() raised on a NaN std."""
    mu = torch.zeros(2, 3)
    sd = torch.ones(2, 3)
    sd[0, 1] = float("nan")
    # RuntimeError on ikura (torch 2.0.1, raised inside torch.normal); ValueError
    # where torch validates distribution args up front. Same bug either way.
    with pytest.raises((RuntimeError, ValueError)):
        torch.distributions.Normal(mu, sd).sample()      # the bug, reproduced
    out_mu, out_sd = sanitize(_obj(), mu, sd)
    assert torch.isfinite(out_sd).all() and (out_sd > 0).all()
    torch.distributions.Normal(out_mu, out_sd).sample()  # must not raise


def test_negative_sigma_is_floored(sanitize):
    sd = torch.ones(4)
    sd[2] = -3.0
    _, out_sd = sanitize(_obj(), torch.zeros(4), sd)
    assert out_sd[2] > 0
    torch.distributions.Normal(torch.zeros(4), out_sd).sample()


def test_nonfinite_mu_is_zeroed(sanitize):
    mu = torch.tensor([1.5, float("nan"), float("inf"), float("-inf")])
    out_mu, _ = sanitize(_obj(), mu, torch.ones(4))
    assert torch.isfinite(out_mu).all()
    assert out_mu[0] == 1.5                              # the good one survives
    assert (out_mu[1:] == 0).all()


def test_good_entries_beside_bad_ones_are_untouched(sanitize):
    """Element-wise, not all-or-nothing: repairing one env must not perturb the
    other fifteen, whose actions are the ones that matter during training."""
    mu = torch.arange(6, dtype=torch.float32)
    sd = torch.full((6,), 0.7)
    mu[3] = float("nan")
    sd[3] = float("nan")
    out_mu, out_sd = sanitize(_obj(), mu.clone(), sd.clone())
    keep = [0, 1, 2, 4, 5]
    assert torch.equal(out_mu[keep], mu[keep])
    assert torch.equal(out_sd[keep], sd[keep])


def test_the_repair_is_loud(sanitize):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sanitize(_obj(), torch.zeros(2), torch.tensor([float("nan"), 1.0]))
    out = buf.getvalue()
    assert "WARNING" in out and "teacher" in out
    assert "1" in out                                    # how many were bad


def test_the_warning_fires_once_per_env_not_per_step(sanitize):
    """A per-step warning would bury the log of a 7200s eval."""
    obj = _obj()
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(5):
            sanitize(obj, torch.zeros(2), torch.tensor([float("nan"), 1.0]))
    assert buf.getvalue().count("WARNING") == 1


def test_sigma_floor_is_small_enough_to_be_a_repair_not_a_policy(sanitize):
    """The floor stands in for 'this teacher said nothing usable', so it must be
    negligible next to a real sigma rather than a distribution anyone samples."""
    assert 0 < _Bare._TEACHER_SIGMA_FLOOR <= 1e-3

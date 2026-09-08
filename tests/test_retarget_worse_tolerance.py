"""The --allow-worse-cm gate on retarget_contact.regression_exceeds.

The guard's job is to refuse a solve that came out WORSE than not retargeting at
all, because that would hand training a reference worse than the original. It was
built for UNDER-CONVERGED solves: 25 iters took sub16 from 2.72 cm to 4.86 cm.

With no tolerance it also refused converged solves that miss by numerical noise:
12 (body, clip) pairs across src5/src15/src17 regressed by 0.01-0.02 cm, did not
improve at 900 iters, and left holes that blocked three teacher arms. --allow-
worse-cm admits those WITHOUT weakening the real check -- and defaults to 0.0, so
every run that does not ask for it behaves exactly as before.

Numbers below are the measured ones from rt-gen-17328164/17328372/17328374/17328377.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from retarget_contact import regression_exceeds  # noqa: E402

# (before_cm, after_cm) exactly as the failing pairs reported them.
NOISE_REGRESSIONS = [
    (1.58, 1.58),   # sub7/sub5_largetable_038
    (1.51, 1.51),   # sub7/sub5_largetable_039
    (6.55, 6.56),   # sub3/sub17_smalltable_033
    (2.05, 2.07),   # sub131/sub15_woodchair_038  -- largest observed, +0.02
    (1.67, 1.68),   # sub9/sub15_woodchair_034
    (2.73, 2.74),   # sub132/sub15_woodchair_034
]
REAL_UNDER_CONVERGENCE = (2.72, 4.86)   # the case the guard exists for


class TestDefaultIsUnchanged:
    """Default tol=0.0 must reproduce the historical `after > before` exactly."""

    @pytest.mark.parametrize("before,after", NOISE_REGRESSIONS)
    def test_noise_regression_refused_by_default(self, before, after):
        # 1.58 -> 1.58 only reads as equal at 2dp; the stored floats differ, but
        # the contract at tol=0 is a plain >, so equal values are NOT a regression.
        assert regression_exceeds(before, after) == (after > before)

    def test_under_convergence_refused_by_default(self):
        assert regression_exceeds(*REAL_UNDER_CONVERGENCE) is True

    def test_improvement_always_accepted(self):
        assert regression_exceeds(8.01, 1.13) is False

    def test_exactly_equal_is_not_a_regression(self):
        assert regression_exceeds(2.00, 2.00) is False


class TestAllowWorseAdmitsOnlyNoise:
    TOL = 0.05   # 2.5x the largest observed noise, ~40x below a real failure

    @pytest.mark.parametrize("before,after", NOISE_REGRESSIONS)
    def test_every_observed_near_miss_is_admitted(self, before, after):
        assert regression_exceeds(before, after, self.TOL) is False

    def test_real_under_convergence_still_refused(self):
        assert regression_exceeds(*REAL_UNDER_CONVERGENCE, self.TOL) is True

    def test_boundary_is_exclusive(self):
        # exactly at the tolerance passes; a hair beyond it does not
        assert regression_exceeds(2.00, 2.05, self.TOL) is False
        assert regression_exceeds(2.00, 2.0501, self.TOL) is True

    def test_tolerance_does_not_change_improving_clips(self):
        # the arms that already work must be unaffected by the flag
        for before, after in [(8.01, 1.13), (2.91, 0.27), (0.96, 0.20)]:
            assert regression_exceeds(before, after) is False
            assert regression_exceeds(before, after, self.TOL) is False

    def test_a_large_tolerance_would_admit_the_real_failure(self):
        # documents WHY 0.05 and not something loose: at 2.5 the guard is useless
        assert regression_exceeds(*REAL_UNDER_CONVERGENCE, 2.5) is False

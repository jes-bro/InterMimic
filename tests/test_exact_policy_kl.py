"""Verify exact_policy_kl (learning/common_agent.py) against INDEPENDENT ground
truth: torch.distributions.kl_divergence for diagonal Gaussians. Also documents
the rl_games policy_kl bias that motivated the replacement.

common_agent.py imports rl_games/isaacgym (unavailable locally), so the test
extracts the function's source from the file and execs it -- testing the code
as shipped, without the module's import graph.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_exact_policy_kl.py -q
"""

import os
import re

import pytest
import torch
from torch.distributions import Normal, kl_divergence

AGENT_FILE = os.path.join(os.path.dirname(__file__), "..",
                          "isaacgym", "src", "intermimic", "learning", "common_agent.py")


def _load_exact_policy_kl():
    src = open(AGENT_FILE).read()
    m = re.search(r"(def exact_policy_kl.*?return kl\.mean\(\) if reduce else kl\n)", src, re.S)
    assert m, "exact_policy_kl not found in common_agent.py"
    ns = {"torch": torch}
    exec(m.group(1), ns)
    return ns["exact_policy_kl"]


exact_policy_kl = _load_exact_policy_kl()


def torch_reference_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """Ground truth from torch.distributions (code we did not write)."""
    return kl_divergence(Normal(p0_mu, p0_sigma), Normal(p1_mu, p1_sigma)).sum(dim=-1)


def test_identical_policies_give_exactly_zero():
    mu = torch.randn(64, 153)
    sigma = torch.full((64, 153), float(torch.exp(torch.tensor(-2.9))))
    kl = exact_policy_kl(mu, sigma, mu.clone(), sigma.clone(), reduce=True)
    assert kl.item() == 0.0  # exact zero -- the whole point vs the biased version


def test_matches_torch_distributions_random():
    torch.manual_seed(0)
    for _ in range(5):
        mu0, mu1 = torch.randn(32, 153), torch.randn(32, 153)
        s0 = torch.rand(32, 153) * 0.5 + 0.01   # arbitrary positive sigmas
        s1 = torch.rand(32, 153) * 0.5 + 0.01
        ours = exact_policy_kl(mu0, s0, mu1, s1, reduce=False)
        ref = torch_reference_kl(mu0, s0, mu1, s1)
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_matches_at_training_operating_point():
    # fixed sigma exp(-2.9) as in the teacher configs (sigma_init val -2.9)
    sigma = torch.full((16, 153), float(torch.exp(torch.tensor(-2.9))))
    mu0 = torch.randn(16, 153) * 0.1
    mu1 = mu0 + torch.randn(16, 153) * 1e-3     # small policy step, like one epoch
    ours = exact_policy_kl(mu0, sigma, mu1, sigma, reduce=True)
    ref = torch_reference_kl(mu0, sigma, mu1, sigma).mean()
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-8)
    assert ours.item() > 0                      # KL is non-negative, always


def test_documents_rl_games_bias():
    # The stock estimator's epsilons bias each dim ~-0.0016 at sigma=exp(-2.9);
    # over 153 dims that's ~-0.25. This test pins the number the kl_threshold
    # calibration (0.06 = observed -0.19 median + 0.25 bias) rests on.
    def rl_games_policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):  # verbatim from 1.1.4 wheel
        c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
        c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
        c3 = -1.0 / 2.0
        return (c1 + c2 + c3).sum(dim=-1).mean()

    mu = torch.zeros(8, 153)
    sigma = torch.full((8, 153), float(torch.exp(torch.tensor(-2.9))))
    biased = rl_games_policy_kl(mu, sigma, mu, sigma)
    assert biased.item() == pytest.approx(-0.25, abs=0.01)   # identical policies != 0
    assert exact_policy_kl(mu, sigma, mu, sigma).item() == 0.0

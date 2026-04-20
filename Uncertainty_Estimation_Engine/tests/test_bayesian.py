"""Tests for Bayesian inference components."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bayesian.prior_distributions import GaussianPrior, LaplacePrior
from bayesian.mcmc_sampler import MetropolisHastings
from bayesian.variational_inference import ELBO, VariationalInference


def test_gaussian_prior_log_prob():
    prior = GaussianPrior(mu=0.0, sigma=1.0)
    x = np.zeros(5)
    lp = prior.log_prob(x)
    assert np.isfinite(lp)


def test_laplace_prior_sample_shape():
    prior = LaplacePrior()
    s = prior.sample((10,))
    assert s.shape == (10,)


def test_mh_sampler_output_shape():
    log_prob = lambda x: -0.5 * np.sum(x**2)
    sampler = MetropolisHastings(n_samples=100, burn_in=50)
    samples = sampler.sample(log_prob, init=np.zeros(3))
    assert samples.shape == (100, 3)


def test_mh_acceptance_rate_in_range():
    log_prob = lambda x: -0.5 * np.sum(x**2)
    sampler = MetropolisHastings(n_samples=200, burn_in=50)
    sampler.sample(log_prob, np.zeros(2))
    assert 0.0 < sampler.acceptance_rate < 1.0


def test_elbo_gaussian_kl_zero():
    kl = ELBO.gaussian_kl(np.zeros(4), np.ones(4))
    assert kl == pytest.approx(0.0, abs=1e-6)

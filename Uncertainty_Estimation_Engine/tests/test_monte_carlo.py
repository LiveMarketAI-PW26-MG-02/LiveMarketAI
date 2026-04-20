"""Tests for MonteCarloEstimator."""

import numpy as np
import pytest
from core.monte_carlo import MonteCarloEstimator


def test_sample_uncertainty_normal():
    mc = MonteCarloEstimator(n_samples=5000, random_state=0)
    samples, stats = mc.sample_uncertainty("normal", {"loc": 0.0, "scale": 1.0}, size=5000)
    assert len(samples) == 5000
    assert abs(stats["mean"]) < 0.1
    assert abs(stats["std"] - 1.0) < 0.1


def test_sample_uncertainty_unsupported():
    mc = MonteCarloEstimator(n_samples=100)
    with pytest.raises(ValueError):
        mc.sample_uncertainty("not_a_dist", {}, size=100)


def test_confidence_interval_ordering():
    mc = MonteCarloEstimator(n_samples=2000, confidence_level=0.9, random_state=1)
    _, stats = mc.sample_uncertainty("normal", {}, size=2000)
    assert stats["ci_low"] < stats["ci_high"]


def test_propagate_uncertainty_linear():
    """For a linear function f(x) = x, output std ≈ input std."""
    mc = MonteCarloEstimator(n_samples=10000, random_state=2)
    mean, std = mc.propagate_uncertainty(
        func=lambda x: float(x[0]),
        inputs=np.array([1.0]),
        input_stds=np.array([0.5]),
    )
    assert abs(std - 0.5) < 0.05


def test_integrate_unit_square():
    """Integral of f(x,y)=1 over [0,1]^2 should be ~1."""
    mc = MonteCarloEstimator(n_samples=20000, random_state=3)
    est, se = mc.integrate(lambda x: np.ones(x.shape[0]), [(0, 1), (0, 1)])
    assert abs(est - 1.0) < 0.05
    assert se < 0.01


def test_mc_dropout_predict_shapes():
    def stochastic_model(X, training=True):
        rng = np.random.default_rng()
        return X.mean(axis=1) + rng.normal(0, 0.1, X.shape[0])

    mc = MonteCarloEstimator(n_samples=50)
    mean, epi, alea = mc.mc_dropout_predict(
        model_fn=lambda X: stochastic_model(X),
        X=np.random.default_rng(0).standard_normal((10, 3)),
        n_forward=50,
    )
    assert mean.shape == (10,)
    assert epi.shape == (10,)
    assert alea.shape == (10,)

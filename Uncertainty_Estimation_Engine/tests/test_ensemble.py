"""Tests for ensemble uncertainty methods."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ensemble.deep_ensemble import DeepEnsembleEstimator
from models.dropout_model import MCDropoutModel


def make_ensemble(n=3):
    estimators = [MCDropoutModel(input_dim=4, hidden_dims=[8], n_samples=5) for _ in range(n)]
    return DeepEnsembleEstimator(estimators)


def test_ensemble_fit(make_data):
    X, y = make_data
    ens = make_ensemble()
    ens.fit(X, y)
    assert ens._is_fitted


def test_ensemble_predict_shape(make_data):
    X, y = make_data
    ens = make_ensemble()
    ens.fit(X, y)
    mean, ep, al = ens.predict_with_uncertainty(X)
    assert mean.shape == (len(X),)
    assert ep.shape == (len(X),)


def test_ensemble_diversity_positive(make_data):
    X, y = make_data
    ens = make_ensemble()
    ens.fit(X, y)
    div = ens.diversity(X)
    assert div >= 0.0


@pytest.fixture
def make_data():
    return np.random.randn(30, 4), np.random.randn(30)

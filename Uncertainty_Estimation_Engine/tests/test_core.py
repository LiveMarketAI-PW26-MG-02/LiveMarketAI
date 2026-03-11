"""Tests for the core UncertaintyEngine."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.uncertainty_engine import UncertaintyEngine, UncertaintyResult
from models.dropout_model import MCDropoutModel


@pytest.fixture
def engine():
    return UncertaintyEngine(config={"mc_samples": 10})


@pytest.fixture
def model():
    m = MCDropoutModel(input_dim=4, hidden_dims=[8], n_samples=10)
    m._is_fitted = True
    return m


@pytest.fixture
def X():
    return np.random.randn(20, 4)


def test_register_model(engine, model):
    engine.register_model("test", model)
    assert "test" in engine.list_models()


def test_unregister_model(engine, model):
    engine.register_model("test", model)
    engine.unregister_model("test")
    assert "test" not in engine.list_models()


def test_predict_returns_result(engine, model, X):
    engine.register_model("test", model)
    result = engine.predict("test", X)
    assert isinstance(result, UncertaintyResult)
    assert result.predictions.shape == (20,)
    assert result.epistemic_uncertainty.shape == (20,)
    assert result.aleatoric_uncertainty.shape == (20,)


def test_predict_unknown_model_raises(engine, X):
    with pytest.raises(KeyError):
        engine.predict("nonexistent", X)


def test_total_uncertainty_nonnegative(engine, model, X):
    engine.register_model("test", model)
    result = engine.predict("test", X)
    assert np.all(result.total_uncertainty >= 0)


def test_result_to_dict(engine, model, X):
    engine.register_model("test", model)
    result = engine.predict("test", X)
    d = result.to_dict()
    assert "predictions" in d
    assert "mean_epistemic" in d

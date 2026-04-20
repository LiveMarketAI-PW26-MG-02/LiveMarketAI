"""Tests for inflation forecasting models."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.phillips_curve import PhillipsCurveModel
from models.arima_model import ARIMAInflationModel
from models.lstm_model import LSTMInflationModel


@pytest.fixture
def macro_data():
    rng = np.random.default_rng(0)
    n = 60
    u = rng.normal(5.0, 0.5, n)
    pi = 2.0 + 0.5*(4.5 - u) + rng.normal(0, 0.2, n)
    return u, pi

@pytest.fixture
def inflation_series():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    values = 3.0 + np.cumsum(rng.normal(0, 0.15, 60))
    return pd.Series(values, index=dates)

def test_phillips_curve_fit(macro_data):
    u, pi = macro_data
    m = PhillipsCurveModel()
    m.fit(u, pi)
    assert m.is_fitted

def test_phillips_curve_predict(macro_data):
    u, pi = macro_data
    m = PhillipsCurveModel()
    m.fit(u, pi)
    preds = m.predict(u)
    assert preds.shape == u.shape

def test_arima_forecast_length(inflation_series):
    m = ARIMAInflationModel()
    m.fit(inflation_series)
    fc = m.forecast(steps=12)
    assert len(fc) == 12

def test_lstm_fit_and_forecast(inflation_series):
    m = LSTMInflationModel(input_size=1, hidden_size=16)
    df = inflation_series.to_frame("cpi")
    m.fit(df)
    fc = m.forecast()
    assert len(fc) == m.forecast_horizon

def test_arima_confidence_intervals(inflation_series):
    m = ARIMAInflationModel()
    m.fit(inflation_series)
    lo, hi = m.confidence_intervals(steps=6)
    assert len(lo) == 6
    assert np.all(hi >= lo)

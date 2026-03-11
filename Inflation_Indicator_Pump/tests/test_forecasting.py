"""Tests for the forecast engine and evaluator."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forecasting.forecast_engine import ForecastEngine
from forecasting.forecast_evaluator import ForecastEvaluator
from models.arima_model import ARIMAInflationModel


@pytest.fixture
def series():
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    return pd.Series(3.0 + np.random.default_rng(1).normal(0, 0.2, 60), index=dates)


def test_forecast_engine_run(series):
    fe = ForecastEngine()
    fe.add_model("arima", ARIMAInflationModel())
    fc_dict = fe.run(series, steps=12)
    assert "arima" in fc_dict
    assert len(fc_dict["arima"]) == 12

def test_forecast_engine_combined(series):
    fe = ForecastEngine()
    fe.add_model("arima", ARIMAInflationModel())
    fe.run(series, steps=6)
    combined = fe.combined_forecast(steps=6)
    assert len(combined) == 6

def test_evaluator_mae():
    actual   = np.array([3.0, 3.1, 3.2, 3.0, 2.9])
    forecast = np.array([2.9, 3.2, 3.1, 3.1, 3.0])
    ev = ForecastEvaluator()
    mae = ev.mae(actual, forecast)
    assert mae >= 0.0

def test_evaluator_all_metrics():
    actual   = np.random.default_rng(9).normal(3.0, 0.3, 20)
    forecast = actual + np.random.default_rng(10).normal(0, 0.1, 20)
    ev = ForecastEvaluator()
    m = ev.evaluate_all(actual, forecast)
    for key in ["mae","rmse","mape","directional_accuracy","bias"]:
        assert key in m

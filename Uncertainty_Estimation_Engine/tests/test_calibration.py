"""Tests for calibration methods."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration.calibration_evaluator import CalibrationEvaluator
from calibration.temperature_scaling import TemperatureScaling
from calibration.platt_scaling import PlattScaling


def test_ece_perfect_calibration():
    probs = np.linspace(0.0, 1.0, 100)
    y = (np.random.rand(100) < probs).astype(float)
    evaluator = CalibrationEvaluator(n_bins=10)
    ece = evaluator.expected_calibration_error(probs, y)
    assert 0.0 <= ece <= 1.0


def test_platt_scaling_fit():
    scores = np.random.randn(200)
    y = (scores > 0).astype(float)
    ps = PlattScaling()
    ps.fit(scores, y)
    cal = ps.calibrate(scores)
    assert np.all((cal >= 0) & (cal <= 1))


def test_calibration_evaluator_reliability_data():
    probs = np.random.rand(100)
    y = (np.random.rand(100) > 0.5).astype(float)
    ev = CalibrationEvaluator(n_bins=10)
    data = ev.reliability_data(probs, y)
    assert "accuracies" in data and len(data["accuracies"]) == 10

"""Tests for trend and regime analysis."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.trend_analysis import TrendAnalyser
from analysis.regime_detector import RegimeDetector
from analysis.decomposition import InflationDecomposer


@pytest.fixture
def inflation_series():
    dates = pd.date_range("2010-01-01", periods=120, freq="MS")
    values = 2.0 + np.sin(np.linspace(0, 4*np.pi, 120)) + np.random.default_rng(5).normal(0, 0.1, 120)
    return pd.Series(values, index=dates)


def test_trend_hp_filter(inflation_series):
    ta = TrendAnalyser()
    trend, cycle = ta.hp_filter(inflation_series)
    assert len(trend) > 0

def test_trend_is_trending_up(inflation_series):
    ta = TrendAnalyser()
    result = ta.is_trending_up(inflation_series)
    assert isinstance(result, bool)

def test_regime_classify():
    rd = RegimeDetector()
    assert rd.classify(-0.5)  == "deflation"
    assert rd.classify(1.0)   == "low"
    assert rd.classify(3.0)   == "moderate"
    assert rd.classify(5.5)   == "high"
    assert rd.classify(10.0)  == "hyperinflation"

def test_regime_series(inflation_series):
    rd = RegimeDetector()
    rs = rd.regime_series(inflation_series)
    assert len(rs) == len(inflation_series)
    assert set(rs.unique()).issubset(set(RegimeDetector.LABELS))

def test_decomposer_seasonal(inflation_series):
    dec = InflationDecomposer()
    result = dec.seasonal_decompose(inflation_series)
    assert "trend" in result and "seasonal" in result and "residual" in result

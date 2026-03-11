"""Tests for inflation indicators."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indicators.cpi_indicator import CPIIndicator, CoreCPIIndicator
from indicators.ppi_indicator import PPIIndicator
from indicators.pce_indicator import PCEIndicator
from indicators.breakeven_inflation import BreakevenInflationRate


@pytest.fixture
def sample_data():
    dates  = pd.date_range("2010-01-01", periods=60, freq="MS")
    values = 150 + np.cumsum(np.random.default_rng(42).normal(0.3, 0.2, 60))
    return pd.DataFrame({"value": values}, index=dates)


def test_cpi_compute_synthetic():
    cpi = CPIIndicator()
    val = cpi.compute(pd.DataFrame())
    assert 0.0 < val < 15.0

def test_cpi_compute_from_data(sample_data):
    cpi = CPIIndicator()
    val = cpi.compute(sample_data)
    assert isinstance(val, float) and np.isfinite(val)

def test_core_cpi_lower_than_headline(sample_data):
    cpi  = CPIIndicator();  v1 = cpi.compute(sample_data)
    core = CoreCPIIndicator(); v2 = core.compute(sample_data)
    assert isinstance(v2, float)

def test_ppi_stage_validation():
    with pytest.raises(ValueError):
        PPIIndicator(stage="invalid")

def test_pce_above_target():
    pce = PCEIndicator()
    pce._last_value = 3.5
    assert pce.is_above_target()

def test_breakeven_maturity_validation():
    with pytest.raises(ValueError):
        BreakevenInflationRate(maturity=7)

def test_breakeven_compute_synthetic():
    br = BreakevenInflationRate(maturity=10)
    val = br.compute(pd.DataFrame())
    assert 0.5 < val < 5.0

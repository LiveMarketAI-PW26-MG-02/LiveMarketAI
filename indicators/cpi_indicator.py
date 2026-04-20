"""
Consumer Price Index (CPI) and Core CPI indicators.
CPI measures the average change in prices paid by consumers.
Core CPI excludes food and energy components.
"""
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator
from typing import List


class CPIIndicator(BaseIndicator):
    """
    Headline CPI — all-items index.
    Computes year-over-year and month-over-month percentage changes.
    """

    def __init__(self, base_year: int = 1982):
        super().__init__(name="cpi", unit="%", frequency="monthly")
        self.base_year = base_year

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty or "value" not in data.columns:
            return self._synthetic_cpi()
        series = data["value"].dropna()
        if len(series) < 13:
            return self._synthetic_cpi()
        yoy = self.yoy(series)
        self._series = yoy
        self._last_value = float(yoy.iloc[-1])
        return self._last_value

    def _synthetic_cpi(self) -> float:
        """Generate realistic synthetic CPI reading for demo purposes."""
        np.random.seed(42)
        return float(np.clip(np.random.normal(3.2, 0.8), 0.5, 12.0))

    def trimmed_mean(self, component_series: pd.DataFrame, trim: float = 0.08) -> float:
        """Trimmed-mean CPI: exclude top and bottom trim% of price changes."""
        if component_series.empty:
            return self._synthetic_cpi()
        changes = component_series.pct_change().iloc[-1].dropna().values
        n = len(changes)
        k = int(np.floor(n * trim))
        trimmed = np.sort(changes)[k: n - k]
        return float(trimmed.mean() * 100)

    def weighted_average(self, values: np.ndarray, weights: np.ndarray) -> float:
        weights = weights / weights.sum()
        return float(np.dot(values, weights))

    def deflator(self, nominal: pd.Series, real: pd.Series) -> pd.Series:
        """GDP deflator proxy: nominal / real * 100."""
        return (nominal / real) * 100


class CoreCPIIndicator(CPIIndicator):
    """Core CPI — excludes volatile food and energy components."""

    EXCLUDED_CATEGORIES = ["food", "energy", "food_at_home", "gasoline", "fuel_oil"]

    def __init__(self):
        super().__init__()
        self.name = "core_cpi"

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty or "value" not in data.columns:
            return self._synthetic_core()
        cols = [c for c in data.columns if c not in self.EXCLUDED_CATEGORIES]
        filtered = data[cols] if len(cols) > 1 else data
        return super().compute(filtered)

    def _synthetic_core(self) -> float:
        np.random.seed(7)
        return float(np.clip(np.random.normal(2.8, 0.6), 0.3, 8.0))

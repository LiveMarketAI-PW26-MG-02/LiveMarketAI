"""
Trend analysis for inflation time series.
HP filter, moving averages, and structural break detection.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TrendAnalyser:
    """Extracts trend, cycle, and seasonal components from inflation series."""

    def hp_filter(self, series: pd.Series, lamb: float = 1600) -> Tuple[pd.Series, pd.Series]:
        """Hodrick-Prescott filter. Returns (trend, cycle)."""
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(series.dropna(), lamb=lamb)
            return trend, cycle
        except Exception:
            trend = series.rolling(12, min_periods=1).mean()
            return trend, series - trend

    def exponential_smoothing(self, series: pd.Series, alpha: float = 0.3) -> pd.Series:
        smoothed = [series.iloc[0]]
        for v in series.iloc[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return pd.Series(smoothed, index=series.index)

    def rolling_statistics(self, series: pd.Series, window: int = 12) -> pd.DataFrame:
        return pd.DataFrame({
            "rolling_mean": series.rolling(window).mean(),
            "rolling_std":  series.rolling(window).std(),
            "rolling_min":  series.rolling(window).min(),
            "rolling_max":  series.rolling(window).max(),
        })

    def structural_breaks(self, series: pd.Series, min_size: int = 12) -> list:
        """Detect structural break points using Chow test approximation."""
        breaks = []
        n = len(series)
        for i in range(min_size, n - min_size):
            s1 = series.iloc[:i].values
            s2 = series.iloc[i:].values
            if abs(s1.mean() - s2.mean()) > 1.5 * series.std():
                breaks.append(series.index[i])
        return breaks

    def acceleration(self, series: pd.Series) -> pd.Series:
        """Second derivative — rate of change of inflation rate."""
        return series.diff().diff()

    def is_trending_up(self, series: pd.Series, window: int = 6) -> bool:
        recent = series.dropna().iloc[-window:]
        return bool(recent.iloc[-1] > recent.iloc[0])

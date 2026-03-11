"""Abstract base class for all inflation indicators."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class BaseIndicator(ABC):
    """Every indicator must implement compute() returning a scalar float."""

    def __init__(self, name: str, unit: str = "%", frequency: str = "monthly"):
        self.name = name
        self.unit = unit
        self.frequency = frequency
        self._last_value: Optional[float] = None
        self._series: Optional[pd.Series] = None

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> float:
        """Compute the indicator value from raw data and return a scalar."""
        ...

    def yoy(self, series: pd.Series) -> pd.Series:
        """Year-over-year percentage change."""
        return series.pct_change(periods=12) * 100

    def mom(self, series: pd.Series) -> pd.Series:
        """Month-over-month percentage change."""
        return series.pct_change(periods=1) * 100

    def rolling_average(self, series: pd.Series, window: int = 3) -> pd.Series:
        return series.rolling(window=window).mean()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name, "unit": self.unit,
            "frequency": self.frequency, "last_value": self._last_value,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', last={self._last_value})"

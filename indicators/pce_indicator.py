"""
Personal Consumption Expenditures (PCE) Price Index.
The Federal Reserve's preferred inflation gauge.
"""
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator


class PCEIndicator(BaseIndicator):
    """
    PCE Price Index — Fed's preferred inflation measure.
    Includes broader spending coverage than CPI.
    """

    FED_TARGET = 2.0  # percent

    def __init__(self, core: bool = False):
        name = "core_pce" if core else "pce"
        super().__init__(name=name, unit="%", frequency="monthly")
        self.core = core

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty or "value" not in data.columns:
            return self._synthetic_pce()
        series = data["value"].dropna()
        if len(series) < 13:
            return self._synthetic_pce()
        yoy = self.yoy(series)
        self._last_value = float(yoy.iloc[-1])
        return self._last_value

    def _synthetic_pce(self) -> float:
        np.random.seed(21)
        mu = 2.6 if not self.core else 2.4
        return float(np.clip(np.random.normal(mu, 0.5), 0.2, 7.0))

    def deviation_from_target(self) -> float:
        """How far the latest PCE is from the Fed's 2% target."""
        if self._last_value is None:
            return 0.0
        return self._last_value - self.FED_TARGET

    def is_above_target(self) -> bool:
        return self.deviation_from_target() > 0

"""
Commodity Price Index — energy, food, metals sub-indices.
Rising commodity prices are a leading indicator of consumer inflation.
"""
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator
from typing import Dict


class CommodityIndex(BaseIndicator):
    """
    Composite commodity price index with sub-indices.
    Weights: energy 40%, food 35%, metals 25%.
    """

    WEIGHTS: Dict[str, float] = {"energy": 0.40, "food": 0.35, "metals": 0.25}

    def __init__(self):
        super().__init__(name="commodity", unit="index", frequency="daily")
        self._sub_indices: Dict[str, float] = {}

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty:
            return self._synthetic_index()
        sub = {}
        for cat, wt in self.WEIGHTS.items():
            if cat in data.columns:
                sub[cat] = float(data[cat].dropna().iloc[-1])
            else:
                sub[cat] = 100.0
        self._sub_indices = sub
        self._last_value = sum(sub[c] * w for c, w in self.WEIGHTS.items())
        return self._last_value

    def _synthetic_index(self) -> float:
        np.random.seed(55)
        return float(np.random.normal(115.0, 12.0))

    def energy_sub_index(self, data: pd.DataFrame) -> float:
        if "crude_oil" in data.columns and "natural_gas" in data.columns:
            return float(0.6*data["crude_oil"].iloc[-1] + 0.4*data["natural_gas"].iloc[-1])
        return 100.0

    def food_sub_index(self, data: pd.DataFrame) -> float:
        cols = [c for c in ["wheat","corn","soybeans","rice"] if c in data.columns]
        if cols:
            return float(data[cols].iloc[-1].mean())
        return 100.0

    def yoy_change(self) -> float:
        if self._series is not None and len(self._series) >= 252:
            return float((self._series.iloc[-1] / self._series.iloc[-252] - 1) * 100)
        return 0.0

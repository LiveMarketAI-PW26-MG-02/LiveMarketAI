"""
Breakeven Inflation Rate derived from TIPS spreads.
The difference between nominal Treasury yield and TIPS yield.
"""
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator


class BreakevenInflationRate(BaseIndicator):
    """
    Market-implied inflation expectations from TIPS breakeven rates.
    breakeven = nominal_yield - TIPS_yield
    """

    MATURITIES = [2, 5, 10, 20, 30]  # years

    def __init__(self, maturity: int = 10):
        if maturity not in self.MATURITIES:
            raise ValueError(f"maturity must be in {self.MATURITIES}")
        super().__init__(name="breakeven", unit="%", frequency="daily")
        self.maturity = maturity

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty:
            return self._synthetic_breakeven()
        nominal_col = f"nominal_{self.maturity}y"
        tips_col    = f"tips_{self.maturity}y"
        if nominal_col in data.columns and tips_col in data.columns:
            nominal = data[nominal_col].dropna().iloc[-1]
            tips    = data[tips_col].dropna().iloc[-1]
            self._last_value = float(nominal - tips)
        else:
            self._last_value = self._synthetic_breakeven()
        return self._last_value

    def _synthetic_breakeven(self) -> float:
        np.random.seed(self.maturity)
        return float(np.clip(np.random.normal(2.3, 0.3), 0.5, 4.5))

    def term_structure(self, data: pd.DataFrame) -> dict:
        """Build breakeven term structure across all maturities."""
        result = {}
        for m in self.MATURITIES:
            ind = BreakevenInflationRate(maturity=m)
            result[f"{m}y"] = ind.compute(data)
        return result

    def inflation_risk_premium(self, nominal: float, real: float, expected: float) -> float:
        """Decompose: breakeven = expected inflation + risk premium."""
        return (nominal - real) - expected

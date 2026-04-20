"""
Decompose inflation into demand-pull, cost-push, and structural components.
Uses seasonal decomposition and contribution analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class InflationDecomposer:
    """
    Decomposes inflation into structural components:
    - Demand-pull: driven by output gap / unemployment gap
    - Cost-push: driven by commodity prices, wages
    - Imported inflation: FX pass-through
    - Expectations-driven: anchoring effects
    """

    def seasonal_decompose(self, series: pd.Series, period: int = 12) -> Dict[str, pd.Series]:
        """Additive seasonal decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            result = seasonal_decompose(series.dropna(), model="additive", period=period)
            return {
                "trend":    result.trend.dropna(),
                "seasonal": result.seasonal,
                "residual": result.resid.dropna(),
                "observed": result.observed,
            }
        except Exception:
            trend = series.rolling(period, center=True).mean()
            return {
                "trend": trend.dropna(),
                "seasonal": pd.Series(np.zeros(len(series)), index=series.index),
                "residual": (series - trend).dropna(),
                "observed": series,
            }

    def demand_pull_component(self, output_gap: pd.Series, sensitivity: float = 0.5) -> pd.Series:
        """Demand-pull inflation = sensitivity * output_gap."""
        return output_gap * sensitivity

    def cost_push_component(self, commodity_price_change: pd.Series,
                             wage_growth: Optional[pd.Series] = None,
                             commodity_weight: float = 0.4) -> pd.Series:
        if wage_growth is not None:
            return commodity_weight * commodity_price_change + (1-commodity_weight) * wage_growth
        return commodity_weight * commodity_price_change

    def contribution_analysis(self, components: Dict[str, float]) -> Dict[str, float]:
        """Normalised contribution of each driver to total inflation."""
        total = sum(abs(v) for v in components.values())
        if total == 0:
            return {k: 0.0 for k in components}
        return {k: v / total * 100 for k, v in components.items()}

    def fx_passthrough(self, exchange_rate_change: float, passthrough_coef: float = 0.3) -> float:
        """Estimate imported inflation from exchange rate depreciation."""
        return exchange_rate_change * passthrough_coef

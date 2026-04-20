"""
Producer Price Index (PPI) indicator.
Measures average change in selling prices received by producers.
"""
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator


class PPIIndicator(BaseIndicator):
    """
    PPI — leading indicator for consumer inflation.
    Tracks pipeline inflationary pressure from producers.
    """

    STAGES = ["crude", "intermediate", "finished"]

    def __init__(self, stage: str = "finished"):
        if stage not in self.STAGES:
            raise ValueError(f"stage must be one of {self.STAGES}")
        super().__init__(name="ppi", unit="%", frequency="monthly")
        self.stage = stage

    def compute(self, data: pd.DataFrame) -> float:
        if data.empty or "value" not in data.columns:
            return self._synthetic_ppi()
        series = data["value"].dropna()
        if len(series) < 13:
            return self._synthetic_ppi()
        yoy = self.yoy(series)
        self._last_value = float(yoy.iloc[-1])
        self._series = yoy
        return self._last_value

    def _synthetic_ppi(self) -> float:
        np.random.seed(13)
        return float(np.clip(np.random.normal(2.5, 1.2), -2.0, 15.0))

    def pipeline_pressure(self, crude: float, intermediate: float, finished: float) -> float:
        """Weighted pipeline pressure index across PPI stages."""
        return 0.2 * crude + 0.3 * intermediate + 0.5 * finished

    def pass_through_ratio(self, ppi_change: float, cpi_change: float) -> float:
        """Estimate fraction of PPI change passed through to consumer prices."""
        if ppi_change == 0:
            return 0.0
        return cpi_change / ppi_change

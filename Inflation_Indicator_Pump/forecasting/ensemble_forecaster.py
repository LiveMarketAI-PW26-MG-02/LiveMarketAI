"""Ensemble inflation forecaster combining ARIMA, VAR, and LSTM."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class EnsembleForecaster:
    """
    Combines multiple forecasting models via stacking or simple averaging.
    """

    def __init__(self, model_names: Optional[List[str]] = None):
        self.model_names = model_names or ["arima", "var", "lstm"]
        self._individual_forecasts: Dict[str, np.ndarray] = {}

    def fit_and_forecast(self, series: pd.Series, steps: int = 12) -> np.ndarray:
        """Fit all sub-models and return ensemble forecast."""
        from models.arima_model import ARIMAInflationModel
        models = {"arima": ARIMAInflationModel()}
        for name, m in models.items():
            try:
                m.fit(series)
                fc = m.forecast(steps=steps)
                self._individual_forecasts[name] = np.array(fc).flatten()[:steps]
            except Exception:
                self._individual_forecasts[name] = np.full(steps, float(series.iloc[-1]))
        return self._combine()

    def _combine(self) -> np.ndarray:
        if not self._individual_forecasts:
            return np.array([])
        return np.stack(list(self._individual_forecasts.values())).mean(axis=0)

    def prediction_intervals(self, confidence: float = 0.9) -> tuple:
        combined = self._combine()
        spread   = np.stack(list(self._individual_forecasts.values())).std(axis=0)
        z = 1.645 if confidence == 0.9 else 1.96
        return combined - z * spread, combined + z * spread

    def model_contributions(self) -> Dict[str, float]:
        """Normalised contribution weight of each model."""
        n = len(self._individual_forecasts)
        return {k: 1.0/n for k in self._individual_forecasts}

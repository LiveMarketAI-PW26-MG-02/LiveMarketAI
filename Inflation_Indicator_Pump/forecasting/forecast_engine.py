"""
Forecast engine: runs multiple models and returns combined forecasts.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Runs and combines inflation forecasts from multiple models.
    Supports mean, median, and weighted combination strategies.
    """

    COMBINATION_METHODS = ["mean", "median", "weighted", "best"]

    def __init__(self, combination: str = "weighted"):
        if combination not in self.COMBINATION_METHODS:
            raise ValueError(f"combination must be one of {self.COMBINATION_METHODS}")
        self.combination = combination
        self._models: Dict[str, object] = {}
        self._weights: Dict[str, float] = {}
        self._forecasts: Dict[str, np.ndarray] = {}

    def add_model(self, name: str, model, weight: float = 1.0) -> "ForecastEngine":
        self._models[name] = model
        self._weights[name] = weight
        return self

    def run(self, series: pd.Series, steps: int = 12) -> Dict[str, np.ndarray]:
        self._forecasts = {}
        for name, model in self._models.items():
            try:
                if hasattr(model, "fit"):
                    model.fit(series)
                fc = model.forecast(steps=steps)
                self._forecasts[name] = np.array(fc).flatten()[:steps]
                logger.info("Model '%s' forecast complete.", name)
            except Exception as exc:
                logger.warning("Model '%s' failed: %s", name, exc)
                self._forecasts[name] = np.full(steps, series.iloc[-1])
        return self._forecasts

    def combined_forecast(self, steps: int = 12) -> np.ndarray:
        if not self._forecasts:
            return np.array([])
        fc_matrix = np.stack(list(self._forecasts.values()))
        if self.combination == "mean":
            return fc_matrix.mean(axis=0)
        if self.combination == "median":
            return np.median(fc_matrix, axis=0)
        if self.combination == "weighted":
            total_w = sum(self._weights[k] for k in self._forecasts)
            weights = np.array([self._weights[k]/total_w for k in self._forecasts])
            return (fc_matrix * weights[:, None]).sum(axis=0)
        # best: return model with lowest historical MAE (placeholder)
        return fc_matrix.mean(axis=0)

    def forecast_uncertainty(self) -> np.ndarray:
        if not self._forecasts:
            return np.array([])
        fc_matrix = np.stack(list(self._forecasts.values()))
        return fc_matrix.std(axis=0)

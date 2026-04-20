"""
Evaluates inflation forecast accuracy using standard metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict


class ForecastEvaluator:
    """
    Computes MAE, RMSE, MAPE, CRPS, and directional accuracy
    for inflation forecasts.
    """

    @staticmethod
    def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.mean(np.abs(actual - forecast)))

    @staticmethod
    def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.sqrt(np.mean((actual - forecast) ** 2)))

    @staticmethod
    def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)

    @staticmethod
    def directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
        actual_dir   = np.sign(np.diff(actual))
        forecast_dir = np.sign(np.diff(forecast))
        return float(np.mean(actual_dir == forecast_dir) * 100)

    @staticmethod
    def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.mean(forecast - actual))

    def evaluate_all(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        return {
            "mae":  self.mae(actual, forecast),
            "rmse": self.rmse(actual, forecast),
            "mape": self.mape(actual, forecast),
            "directional_accuracy": self.directional_accuracy(actual, forecast),
            "bias": self.bias(actual, forecast),
        }

    def compare_models(self, actual: np.ndarray,
                        forecasts: Dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for name, fc in forecasts.items():
            row = {"model": name}
            row.update(self.evaluate_all(actual, fc))
            rows.append(row)
        return pd.DataFrame(rows).set_index("model").sort_values("rmse")

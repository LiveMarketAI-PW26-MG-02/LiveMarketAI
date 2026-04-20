"""Temperature scaling calibration."""
import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaling:
    """Calibrates classifier confidence by learning a single temperature parameter."""

    def __init__(self):
        self.temperature = 1.0

    def _nll(self, T, logits, y):
        scaled = logits / T
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        n = len(y)
        idx = np.arange(n)
        return -np.mean(np.log(probs[idx, y] + 1e-10))

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaling":
        result = minimize_scalar(
            lambda T: self._nll(T, logits, y),
            bounds=(0.05, 10.0), method="bounded"
        )
        self.temperature = float(result.x)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def calibrate_uncertainty(self, uncertainty: np.ndarray) -> np.ndarray:
        return uncertainty * self.temperature

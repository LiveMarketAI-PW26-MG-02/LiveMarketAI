"""
Conformal Prediction for distribution-free uncertainty intervals.
"""
import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """
    Split conformal prediction for valid coverage guarantees.
    Provides prediction intervals with 1-alpha coverage.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._calibration_scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None

    def calibrate(self, residuals: np.ndarray) -> None:
        """Fit conformal quantile from calibration residuals."""
        self._calibration_scores = np.abs(residuals)
        n = len(residuals)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(self._calibration_scores, level))

    def get_intervals(
        self, predictions: np.ndarray, uncertainty: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals. Falls back to 2-sigma if not calibrated."""
        if self._quantile is not None:
            margin = self._quantile
        else:
            margin = 2.0 * uncertainty
        lower = predictions - margin
        upper = predictions + margin
        return lower, upper

    def coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """Compute empirical coverage of intervals."""
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    def efficiency(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """Average interval width (smaller = more efficient)."""
        return float(np.mean(upper - lower))

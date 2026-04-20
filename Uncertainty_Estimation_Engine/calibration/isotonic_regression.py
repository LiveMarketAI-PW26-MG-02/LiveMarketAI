"""Isotonic regression calibration."""
import numpy as np
from sklearn.isotonic import IsotonicRegression as SKLearnIR


class IsotonicCalibrator:
    """Non-parametric monotonic calibration via isotonic regression."""

    def __init__(self):
        self._iso = SKLearnIR(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(probs, y)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return self._iso.predict(probs)

    def calibration_error(self, probs, y):
        calibrated = self.calibrate(probs)
        return float(np.mean(np.abs(calibrated - y)))

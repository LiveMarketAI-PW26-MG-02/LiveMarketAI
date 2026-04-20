"""Platt scaling for binary classifiers."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class PlattScaling:
    """Learns affine transformation of raw scores via sigmoid calibration."""

    def __init__(self):
        self.A = 1.0
        self.B = 0.0

    def _nll(self, params, scores, y):
        A, B = params
        p = expit(A * scores + B)
        return -np.mean(y * np.log(p+1e-10) + (1-y) * np.log(1-p+1e-10))

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattScaling":
        res = minimize(self._nll, x0=[self.A, self.B], args=(scores, y), method="L-BFGS-B")
        self.A, self.B = res.x
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        return expit(self.A * scores + self.B)

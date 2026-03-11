"""
Model Calibration
==================
Post-hoc calibration methods: Temperature Scaling, Platt Scaling,
Isotonic Regression, and Beta Calibration.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature Scaling calibration for neural network classifiers.
    Learns a single scalar T that divides logits before softmax.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 1000):
        self.lr = lr
        self.max_iter = max_iter
        self.T: float = 1.0
        self._loss_history: list = []

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """Optimise T via gradient descent on NLL."""
        T = 1.0
        for step in range(self.max_iter):
            scaled = logits / T
            scaled -= scaled.max(axis=-1, keepdims=True)
            exp_s = np.exp(scaled)
            probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
            n = len(y_true)
            idx = np.arange(n), y_true.astype(int)
            nll = -np.mean(np.log(probs[idx] + 1e-15))
            self._loss_history.append(nll)
            # Gradient of NLL w.r.t. T
            log_probs = scaled - np.log(exp_s.sum(axis=-1, keepdims=True))
            grad = np.mean(
                -logits[idx].sum() / T ** 2 + np.sum(probs * logits, axis=-1) / T ** 2
            )
            T -= self.lr * grad
            T = max(T, 0.05)
            if step > 10 and abs(self._loss_history[-1] - self._loss_history[-2]) < 1e-7:
                break
        self.T = float(T)
        logger.info("Temperature Scaling converged. T* = %.4f", self.T)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature and return calibrated probabilities."""
        scaled = logits / self.T
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum(axis=-1, keepdims=True)

    def __repr__(self) -> str:
        return f"TemperatureScaling(T={self.T:.4f})"


class PlattScaling:
    """
    Platt Scaling (sigmoid calibration) for binary classifiers.
    Fits logistic regression on (score, label) pairs.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 500):
        self.lr = lr
        self.max_iter = max_iter
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        """Fit a and b via gradient descent on binary cross-entropy."""
        a, b = 1.0, 0.0
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            grad_a = -np.mean((y_true - p) * scores)
            grad_b = -np.mean(y_true - p)
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a, self.b = float(a), float(b)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))

    def __repr__(self) -> str:
        return f"PlattScaling(a={self.a:.4f}, b={self.b:.4f})"


class IsotonicCalibration:
    """
    Isotonic Regression calibration via pool adjacent violators (PAV).
    """

    def __init__(self):
        self._calibration_table: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        """Fit isotonic regression."""
        order = np.argsort(scores)
        y_sorted = y_true[order].astype(float)
        calibrated = self._pav(y_sorted)
        self._calibration_table = (scores[order], calibrated)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Interpolate calibration table."""
        if self._calibration_table is None:
            raise RuntimeError("Call fit() first.")
        return np.interp(scores, *self._calibration_table)

    @staticmethod
    def _pav(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators algorithm."""
        n = len(y)
        target = y.copy()
        i = 0
        while i < n:
            j = i + 1
            while j < n and target[j] < target[i]:
                target[i:j+1] = target[i:j+1].mean()
                j += 1
            i = j
        return target

    def __repr__(self) -> str:
        return "IsotonicCalibration()"

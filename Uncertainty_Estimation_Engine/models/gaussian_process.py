"""
Gaussian Process model for uncertainty estimation.
Provides exact posterior predictive distribution.
"""
import numpy as np
from typing import Tuple, Optional, Callable
from core.base_estimator import BaseUncertaintyEstimator


class RBFKernel:
    """Radial Basis Function (squared exponential) kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sq_dist / self.length_scale ** 2)


class GaussianProcessModel(BaseUncertaintyEstimator):
    """
    Exact Gaussian Process regression.
    Provides analytic mean and variance predictions.
    """

    def __init__(
        self,
        kernel: Optional[Callable] = None,
        noise: float = 1e-3,
        name: str = "gp",
    ):
        super().__init__(name=name)
        self.kernel = kernel or RBFKernel()
        self.noise = noise
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GaussianProcessModel":
        self._X_train = X
        self._y_train = y
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self._L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        K_star = self.kernel(X, self._X_train)
        mean = K_star @ self._alpha
        v = np.linalg.solve(self._L, K_star.T)
        K_ss = self.kernel(X, X)
        variance = np.diag(K_ss - v.T @ v)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        return mean, std, np.full_like(std, self.noise)

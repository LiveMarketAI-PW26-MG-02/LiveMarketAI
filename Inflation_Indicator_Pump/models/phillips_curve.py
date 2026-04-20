"""
Phillips Curve model — relationship between inflation and unemployment.
Estimates both traditional and expectations-augmented versions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class PhillipsCurveModel:
    """
    Estimates the Phillips Curve: π = α + β·(u* - u) + ε
    Supports standard, expectations-augmented, and non-linear variants.
    """

    def __init__(self, nairu: float = 4.5, alpha: float = 1.0):
        self.nairu = nairu
        self.alpha_reg = alpha
        self._model = Ridge(alpha=alpha)
        self._scaler = StandardScaler()
        self._coefs: Optional[np.ndarray] = None
        self._intercept: Optional[float] = None
        self.is_fitted = False

    def fit(self, unemployment: np.ndarray, inflation: np.ndarray,
            inflation_expectations: Optional[np.ndarray] = None) -> "PhillipsCurveModel":
        gap = self.nairu - unemployment
        features = gap.reshape(-1, 1)
        if inflation_expectations is not None:
            features = np.column_stack([features, inflation_expectations])
        X_scaled = self._scaler.fit_transform(features)
        self._model.fit(X_scaled, inflation)
        self._coefs = self._model.coef_
        self._intercept = float(self._model.intercept_)
        self.is_fitted = True
        return self

    def predict(self, unemployment: np.ndarray,
                inflation_expectations: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        gap = self.nairu - unemployment
        features = gap.reshape(-1, 1)
        if inflation_expectations is not None:
            features = np.column_stack([features, inflation_expectations])
        X_scaled = self._scaler.transform(features)
        return self._model.predict(X_scaled)

    def sacrifice_ratio(self, delta_u: float = 1.0) -> float:
        """Output cost to reduce inflation by 1pp."""
        if self._coefs is None:
            return 5.0
        slope = abs(self._coefs[0])
        return 1.0 / slope if slope > 0 else np.inf

    def nairu_estimate(self, unemployment: np.ndarray, inflation: np.ndarray) -> float:
        """Estimate NAIRU via grid search on residual variance."""
        best_nairu, best_var = self.nairu, np.inf
        for candidate in np.arange(3.0, 7.0, 0.1):
            self.nairu = candidate
            self.fit(unemployment, inflation)
            resid = inflation - self.predict(unemployment)
            var = float(resid.var())
            if var < best_var:
                best_var = var
                best_nairu = candidate
        self.nairu = best_nairu
        self.fit(unemployment, inflation)
        return best_nairu

from __future__ import annotations

import math

import numpy as np


from sklearn.linear_model import Ridge


class Engine:
    """Linear forecaster with transparent per-feature contributions."""

    FEATURES = ["lag1", "lag5", "trend", "vol", "macro", "flow"]

    def __init__(self, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(700, len(self.FEATURES)))
        w = np.array([0.5, 0.3, 0.4, -0.2, 0.25, 0.35])
        y = X @ w + rng.normal(0, 0.25, size=700)
        self.model = Ridge(alpha=0.5).fit(X, y)
        self.bias = float(self.model.intercept_)

    def _vec(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(f, 0.0)) for f in self.FEATURES])

    def explain(self, features: dict) -> dict:
        x = self._vec(features)
        contrib = self.model.coef_ * x
        pred = float(self.bias + contrib.sum())
        attributions = [{"feature": self.FEATURES[i],
                         "contribution": round(float(contrib[i]), 4)}
                        for i in range(len(self.FEATURES))]
        attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)
        return {"primitive": "forecast", "forecast": round(pred, 4),
                "baseline": round(self.bias, 4),
                "summary": f"Forecast {pred:+.3f}, led by {attributions[0]['feature']}.",
                "attributions": attributions}

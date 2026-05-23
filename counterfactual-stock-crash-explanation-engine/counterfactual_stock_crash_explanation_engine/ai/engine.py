from __future__ import annotations

import math

import numpy as np


from sklearn.linear_model import LogisticRegression


class Engine:
    """Fits a crash classifier and searches a minimal counterfactual that
    flips the prediction below the decision threshold."""

    FEATURES = ["leverage", "liquidity", "volatility", "sentiment",
                "rate_shock", "concentration"]

    def __init__(self, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(800, len(self.FEATURES)))
        w = np.array([1.4, -1.2, 1.1, -0.8, 1.0, 0.9])
        logits = X @ w + rng.normal(0, 0.4, size=800)
        y = (logits > 0).astype(int)
        self.model = LogisticRegression(max_iter=500).fit(X, y)

    def _vec(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(f, 0.0)) for f in self.FEATURES])

    def _proba(self, x: np.ndarray) -> float:
        return float(self.model.predict_proba(x.reshape(1, -1))[0, 1])

    def explain(self, features: dict, target: float = 0.4) -> dict:
        x = self._vec(features)
        base = self._proba(x)
        cf = x.copy()
        steps = 0
        coef = self.model.coef_[0]
        while self._proba(cf) > target and steps < 200:
            grad = coef * self._proba(cf) * (1 - self._proba(cf))
            j = int(np.argmax(np.abs(grad)))
            cf[j] -= 0.05 * np.sign(grad[j])
            steps += 1
        delta = {self.FEATURES[i]: round(float(cf[i] - x[i]), 4)
                 for i in range(len(self.FEATURES)) if abs(cf[i] - x[i]) > 1e-6}
        distance = float(np.linalg.norm(cf - x))
        return {"primitive": "counterfactual", "score": round(base, 4),
                "flipped": self._proba(cf) <= target,
                "counterfactual_delta": delta, "distance": round(distance, 4),
                "summary": f"Crash probability {base:.0%}; minimal aversion needs "
                           f"{len(delta)} factor change(s)."}

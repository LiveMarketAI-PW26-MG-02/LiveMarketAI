from __future__ import annotations

import math

import numpy as np


from sklearn.linear_model import Ridge


class Engine:
    """Attention-style attribution: softmax over |weight*value| of a fitted
    linear factor model gives normalised feature importance."""

    FEATURES = ["momentum", "value", "carry", "quality", "size", "low_vol"]

    def __init__(self, seed: int = 5) -> None:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(700, len(self.FEATURES)))
        w = np.array([0.9, 0.6, -0.4, 0.5, -0.3, 0.7])
        y = X @ w + rng.normal(0, 0.3, size=700)
        self.model = Ridge(alpha=1.0).fit(X, y)

    def _vec(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(f, 0.0)) for f in self.FEATURES])

    def explain(self, features: dict) -> dict:
        x = self._vec(features)
        signed = self.model.coef_ * x
        logits = np.abs(signed)
        attn = np.exp(logits - logits.max())
        attn = attn / (attn.sum() + 1e-9)
        pred = float(self.model.predict(x.reshape(1, -1))[0])
        attributions = [{"feature": self.FEATURES[i], "attention": round(float(attn[i]), 4),
                         "contribution": round(float(signed[i]), 4)}
                        for i in range(len(self.FEATURES))]
        attributions.sort(key=lambda a: a["attention"], reverse=True)
        return {"primitive": "attention", "prediction": round(pred, 4),
                "summary": f"Decision dominated by {attributions[0]['feature']}.",
                "attributions": attributions}

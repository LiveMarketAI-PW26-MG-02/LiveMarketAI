from __future__ import annotations

import math

import numpy as np


from sklearn.ensemble import IsolationForest


class Engine:
    """Unsupervised anomaly detector with ablation-based attribution."""

    FEATURES = ["volume_zscore", "spread", "cancel_ratio", "order_imbalance",
                "price_jump", "trade_intensity"]

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        normal = rng.normal(0.0, 1.0, size=(600, len(self.FEATURES)))
        self.medians = np.median(normal, axis=0)
        self.model = IsolationForest(n_estimators=150, random_state=seed)
        self.model.fit(normal)

    def _vec(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(f, 0.0)) for f in self.FEATURES])

    def _score(self, x: np.ndarray) -> float:
        raw = -float(self.model.score_samples(x.reshape(1, -1))[0])
        return 1.0 / (1.0 + math.exp(-raw))

    def explain(self, features: dict) -> dict:
        x = self._vec(features)
        score = self._score(x)
        attributions = []
        for i, name in enumerate(self.FEATURES):
            ablated = x.copy()
            ablated[i] = self.medians[i]
            contribution = score - self._score(ablated)
            attributions.append({"feature": name, "value": float(x[i]),
                                  "contribution": round(float(contribution), 4)})
        attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)
        top = attributions[0]["feature"] if attributions else "n/a"
        return {"primitive": "anomaly", "score": round(score, 4),
                "flagged": score >= 0.6,
                "summary": f"Anomaly score {score:.0%}; primary driver: {top}.",
                "attributions": attributions}

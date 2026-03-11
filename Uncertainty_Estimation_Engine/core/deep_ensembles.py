"""
Deep Ensembles uncertainty estimation.
Trains multiple models and aggregates predictions.
"""
import numpy as np
from typing import List, Tuple
from .base_estimator import BaseUncertaintyEstimator


class DeepEnsemble(BaseUncertaintyEstimator):
    """Combines multiple base models for robust uncertainty estimation."""

    def __init__(self, estimators: List[BaseUncertaintyEstimator], name: str = "deep_ensemble"):
        super().__init__(name=name)
        self.estimators = estimators
        self.n_estimators = len(estimators)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "DeepEnsemble":
        for i, est in enumerate(self.estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            est.fit(X[indices], y[indices], **kwargs)
        self._is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        all_preds, all_ep, all_al = [], [], []
        for est in self.estimators:
            p, ep, al = est.predict_with_uncertainty(X)
            all_preds.append(p)
            all_ep.append(ep)
            all_al.append(al)
        preds = np.stack(all_preds)
        mean_pred = preds.mean(axis=0)
        epistemic = preds.var(axis=0)
        aleatoric = np.stack(all_al).mean(axis=0)
        return mean_pred, epistemic, aleatoric

    def diversity_score(self, X: np.ndarray) -> float:
        """Average pairwise disagreement between ensemble members."""
        preds = [e.predict(X) for e in self.estimators]
        preds = np.stack(preds)
        n = self.n_estimators
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.mean((preds[i] - preds[j]) ** 2)
                count += 1
        return total / count if count > 0 else 0.0

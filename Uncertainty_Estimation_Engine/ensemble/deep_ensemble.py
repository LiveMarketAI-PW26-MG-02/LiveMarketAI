"""Deep ensemble uncertainty estimation."""
import numpy as np
from typing import List
from core.base_estimator import BaseUncertaintyEstimator


class DeepEnsembleEstimator(BaseUncertaintyEstimator):
    def __init__(self, estimators: List[BaseUncertaintyEstimator], name="deep_ensemble"):
        super().__init__(name=name)
        self.estimators = estimators

    def fit(self, X, y, **kwargs):
        for est in self.estimators:
            idx = np.random.choice(len(X), size=len(X), replace=True)
            est.fit(X[idx], y[idx], **kwargs)
        self._is_fitted = True
        return self

    def predict_with_uncertainty(self, X):
        self._check_is_fitted()
        preds, ep, al = zip(*[e.predict_with_uncertainty(X) for e in self.estimators])
        preds_arr = np.stack(preds)
        mean_pred = preds_arr.mean(axis=0)
        epistemic = preds_arr.var(axis=0)
        aleatoric = np.stack(al).mean(axis=0)
        return mean_pred, epistemic, aleatoric

    def diversity(self, X):
        preds = [e.predict(X) for e in self.estimators]
        preds = np.stack(preds)
        n = len(self.estimators)
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(np.mean((preds[i]-preds[j])**2))
        return float(np.mean(diffs)) if diffs else 0.0

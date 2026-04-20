"""Bootstrap ensemble for uncertainty estimation."""
import numpy as np
from typing import List
from core.base_estimator import BaseUncertaintyEstimator


class BootstrapEnsemble(BaseUncertaintyEstimator):
    """Trains ensemble members on bootstrap resamples."""

    def __init__(self, base_estimator_class, n_estimators=10, **est_kwargs):
        super().__init__(name="bootstrap_ensemble")
        self.base_estimator_class = base_estimator_class
        self.n_estimators = n_estimators
        self.est_kwargs = est_kwargs
        self.estimators: List[BaseUncertaintyEstimator] = []

    def fit(self, X, y, **kwargs):
        self.estimators = []
        for i in range(self.n_estimators):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            est = self.base_estimator_class(**self.est_kwargs)
            est.fit(X[idx], y[idx], **kwargs)
            self.estimators.append(est)
        self._is_fitted = True
        return self

    def predict_with_uncertainty(self, X):
        self._check_is_fitted()
        preds = np.stack([e.predict(X) for e in self.estimators])
        return preds.mean(axis=0), preds.var(axis=0), np.zeros(X.shape[0])

    def oob_score(self):
        return None  # placeholder for out-of-bag scoring

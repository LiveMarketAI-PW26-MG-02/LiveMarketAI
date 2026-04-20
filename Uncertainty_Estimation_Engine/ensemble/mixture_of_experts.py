"""Mixture of Experts model for uncertainty estimation."""
import numpy as np
from typing import List
from core.base_estimator import BaseUncertaintyEstimator


class MixtureOfExperts(BaseUncertaintyEstimator):
    """Gating-network-based mixture of expert models."""

    def __init__(self, experts: List[BaseUncertaintyEstimator], name="moe"):
        super().__init__(name=name)
        self.experts = experts
        self.n_experts = len(experts)
        self._gate_weights: np.ndarray = None

    def fit(self, X, y, **kwargs):
        for expert in self.experts:
            expert.fit(X, y, **kwargs)
        self._gate_weights = np.ones(self.n_experts) / self.n_experts
        self._is_fitted = True
        return self

    def _compute_gates(self, X):
        # Uniform gating (override with learned gating network)
        n = len(X)
        return np.tile(self._gate_weights, (n, 1))

    def predict_with_uncertainty(self, X):
        self._check_is_fitted()
        gates = self._compute_gates(X)
        preds, epis, aleas = [], [], []
        for expert in self.experts:
            p, ep, al = expert.predict_with_uncertainty(X)
            preds.append(p)
            epis.append(ep)
            aleas.append(al)
        preds_arr = np.stack(preds, axis=1)
        epis_arr = np.stack(epis, axis=1)
        aleas_arr = np.stack(aleas, axis=1)
        mean_pred = (gates * preds_arr).sum(axis=1)
        epistemic = (gates * epis_arr).sum(axis=1)
        aleatoric = (gates * aleas_arr).sum(axis=1)
        return mean_pred, epistemic, aleatoric

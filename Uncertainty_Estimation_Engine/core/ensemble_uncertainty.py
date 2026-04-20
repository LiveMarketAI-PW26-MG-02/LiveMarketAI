"""
Ensemble Uncertainty Estimation
=================================
Combines predictions from multiple base models to decompose
total predictive uncertainty into epistemic and aleatoric components.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EnsembleUncertainty:
    """
    Deep Ensemble uncertainty estimator.

    Aggregates an ensemble of model predictions to compute:
    - Predictive mean
    - Epistemic uncertainty  (variance of means)
    - Aleatoric uncertainty  (mean of variances)
    - Total predictive uncertainty
    """

    AGGREGATION_METHODS = ("mean", "trimmed_mean", "median", "weighted_mean")

    def __init__(
        self,
        aggregation: str = "mean",
        trim_fraction: float = 0.1,
        weights: Optional[np.ndarray] = None,
    ):
        if aggregation not in self.AGGREGATION_METHODS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATION_METHODS}")
        self.aggregation = aggregation
        self.trim_fraction = trim_fraction
        self.weights = weights
        self._ensemble_preds: Optional[np.ndarray] = None
        self._ensemble_vars: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def collect_predictions(
        self,
        models: List[Callable],
        X: np.ndarray,
        return_variance: bool = False,
    ) -> "EnsembleUncertainty":
        """
        Run forward pass for each model in the ensemble.

        Parameters
        ----------
        models : list of callables
            Each callable takes X and returns either y_pred or (y_pred, y_var).
        X : ndarray
        return_variance : bool
            If True, models must return (mean, variance) tuples.
        """
        logger.info("Collecting predictions from %d ensemble members …", len(models))
        means, variances = [], []
        for i, m in enumerate(models):
            out = m(X)
            if return_variance:
                mu, var = out
            else:
                mu = out
                var = np.zeros_like(mu)
            means.append(np.asarray(mu))
            variances.append(np.asarray(var))

        self._ensemble_preds = np.stack(means, axis=0)   # (M, n)
        self._ensemble_vars = np.stack(variances, axis=0) # (M, n)
        logger.info("Collected ensemble predictions. Shape: %s", self._ensemble_preds.shape)
        return self

    def predict(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Aggregate ensemble predictions and decompose uncertainty.

        Returns
        -------
        mean : ndarray, shape (n,)
        uncertainty_dict : dict with keys
            'epistemic', 'aleatoric', 'total', 'std_dev'
        """
        self._check_collected()
        mean = self._aggregate(self._ensemble_preds)
        epistemic = self._epistemic_var()
        aleatoric = self._aleatoric_var()
        total = epistemic + aleatoric
        uncertainty = {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "std_dev": np.sqrt(total),
            "disagreement": self._pairwise_disagreement(),
        }
        return mean, uncertainty

    # ------------------------------------------------------------------
    # Diversity metrics
    # ------------------------------------------------------------------

    def ensemble_diversity(self) -> Dict[str, float]:
        """Compute diversity metrics across ensemble members."""
        self._check_collected()
        P = self._ensemble_preds
        M = P.shape[0]
        # Ambiguity decomposition
        mean_pred = P.mean(axis=0)
        ambiguity = np.mean((P - mean_pred) ** 2)
        # Q-statistic (pairwise)
        q_stats = []
        for i in range(M):
            for j in range(i + 1, M):
                corr = np.corrcoef(P[i], P[j])[0, 1]
                q_stats.append(corr)
        return {
            "ambiguity": float(ambiguity),
            "mean_pairwise_correlation": float(np.mean(q_stats)) if q_stats else 0.0,
            "mean_member_std": float(P.std(axis=1).mean()),
            "n_members": M,
        }

    def rank_members(self) -> np.ndarray:
        """Rank ensemble members by their contribution to total variance."""
        self._check_collected()
        ensemble_mean = self._ensemble_preds.mean(axis=0)
        contributions = np.array(
            [np.mean((self._ensemble_preds[i] - ensemble_mean) ** 2)
             for i in range(len(self._ensemble_preds))]
        )
        return np.argsort(-contributions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        if self.aggregation == "mean":
            return preds.mean(axis=0)
        elif self.aggregation == "trimmed_mean":
            k = max(1, int(self.trim_fraction * len(preds)))
            trimmed = np.sort(preds, axis=0)[k:-k]
            return trimmed.mean(axis=0)
        elif self.aggregation == "median":
            return np.median(preds, axis=0)
        elif self.aggregation == "weighted_mean":
            w = self.weights if self.weights is not None else np.ones(len(preds))
            w = w / w.sum()
            return (preds * w[:, None]).sum(axis=0)
        return preds.mean(axis=0)

    def _epistemic_var(self) -> np.ndarray:
        """Variance of predictive means across ensemble members."""
        return self._ensemble_preds.var(axis=0)

    def _aleatoric_var(self) -> np.ndarray:
        """Mean of individual model variances."""
        return self._ensemble_vars.mean(axis=0)

    def _pairwise_disagreement(self) -> np.ndarray:
        """Mean squared pairwise disagreement per data point."""
        P = self._ensemble_preds
        M = P.shape[0]
        total = np.zeros(P.shape[1])
        count = 0
        for i in range(M):
            for j in range(i + 1, M):
                total += (P[i] - P[j]) ** 2
                count += 1
        return total / max(count, 1)

    def _check_collected(self) -> None:
        if self._ensemble_preds is None:
            raise RuntimeError("Call collect_predictions() first.")

    def __repr__(self) -> str:
        return (
            f"EnsembleUncertainty(aggregation='{self.aggregation}', "
            f"trim_fraction={self.trim_fraction})"
        )

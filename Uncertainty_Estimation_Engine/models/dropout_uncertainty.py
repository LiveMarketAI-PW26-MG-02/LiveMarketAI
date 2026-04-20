"""
Monte Carlo Dropout Uncertainty Model
=======================================
Wraps a stochastic neural network (with dropout layers active at test time)
to estimate predictive uncertainty via repeated forward passes.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class DropoutUncertaintyModel:
    """
    MC Dropout wrapper for probabilistic deep learning.

    The user supplies a model_fn that, when called with (X, training=True),
    returns predictions with dropout active (enabling stochastic variates).
    """

    def __init__(
        self,
        model_fn: Callable,
        n_forward_passes: int = 100,
        task: str = "regression",
        random_state: Optional[int] = None,
    ):
        assert task in ("regression", "classification"), \
            "task must be 'regression' or 'classification'"
        self.model_fn = model_fn
        self.n_forward_passes = n_forward_passes
        self.task = task
        self.rng = np.random.default_rng(random_state)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run T stochastic forward passes and decompose uncertainty.

        Returns
        -------
        mean_prediction : ndarray
        uncertainty : dict with keys 'epistemic', 'aleatoric', 'total', 'entropy'
        """
        logger.info("Running %d MC dropout forward passes …", self.n_forward_passes)
        preds = self._gather_predictions(X)

        if self.task == "regression":
            return self._regression_uncertainty(preds)
        else:
            return self._classification_uncertainty(preds)

    def _gather_predictions(self, X: np.ndarray) -> np.ndarray:
        """Collect stochastic predictions, shape (T, n) or (T, n, K)."""
        results: List[np.ndarray] = []
        for _ in range(self.n_forward_passes):
            out = self.model_fn(X, training=True)
            results.append(np.asarray(out))
        return np.stack(results, axis=0)

    def _regression_uncertainty(
        self, preds: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """For regression: preds shape (T, n)."""
        mean = preds.mean(axis=0)
        epistemic_var = preds.var(axis=0)
        aleatoric_var = np.zeros_like(mean)  # needs heteroscedastic model for non-zero
        total_var = epistemic_var + aleatoric_var
        return mean, {
            "epistemic": np.sqrt(epistemic_var),
            "aleatoric": np.sqrt(aleatoric_var),
            "total": np.sqrt(total_var),
            "entropy": np.full(len(mean), np.nan),
        }

    def _classification_uncertainty(
        self, preds: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        For classification: preds shape (T, n, K).
        Uncertainty = predictive entropy − expected per-sample entropy.
        """
        eps = 1e-15
        mean_probs = preds.mean(axis=0)                         # (n, K)
        pred_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)
        per_pass_entropy = -np.sum(preds * np.log(preds + eps), axis=-1)   # (T, n)
        expected_entropy = per_pass_entropy.mean(axis=0)                   # (n,)
        mutual_info = pred_entropy - expected_entropy

        return mean_probs.argmax(axis=-1), {
            "epistemic": mutual_info,
            "aleatoric": expected_entropy,
            "total": pred_entropy,
            "entropy": pred_entropy,
        }

    def calibrate_temperature(
        self, logits: np.ndarray, y_true: np.ndarray, temps: Optional[np.ndarray] = None
    ) -> float:
        """
        Grid-search optimal temperature scaling factor T* that minimises NLL.
        """
        if temps is None:
            temps = np.linspace(0.1, 5.0, 50)

        best_T, best_nll = 1.0, np.inf
        for T in temps:
            scaled = logits / T
            scaled -= scaled.max(axis=-1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
            nll = -log_probs[np.arange(len(y_true)), y_true.astype(int)].mean()
            if nll < best_nll:
                best_nll, best_T = nll, T

        logger.info("Optimal temperature: %.3f (NLL=%.4f)", best_T, best_nll)
        return float(best_T)

    def __repr__(self) -> str:
        return (
            f"DropoutUncertaintyModel(n_forward_passes={self.n_forward_passes}, "
            f"task='{self.task}')"
        )

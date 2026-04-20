"""
Deep Ensemble Model
====================
Trains and aggregates M independently-initialised models to provide
well-calibrated uncertainty estimates without Bayesian approximations.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """
    Lakshminarayanan-style Deep Ensemble.

    Each member outputs (mean, log_var) and is trained with NLL loss.
    At inference, mixture-of-Gaussians aggregation yields calibrated UQ.
    """

    def __init__(
        self,
        model_builder: Callable[[], object],
        n_members: int = 5,
        adversarial_training: bool = False,
        adversarial_eps: float = 0.01,
    ):
        self.model_builder = model_builder
        self.n_members = n_members
        self.adversarial_training = adversarial_training
        self.adversarial_eps = adversarial_eps
        self.members: List = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **fit_kwargs,
    ) -> "DeepEnsemble":
        """
        Build and train each ensemble member independently.

        Parameters
        ----------
        train_fn : callable
            train_fn(model, X, y, **kwargs) → trained model
        """
        logger.info("Training %d ensemble members …", self.n_members)
        self.members = []
        for i in range(self.n_members):
            logger.info("  Training member %d / %d …", i + 1, self.n_members)
            model = self.model_builder()
            trained = train_fn(model, X_train, y_train, **fit_kwargs)
            self.members.append(trained)
        self._trained = True
        logger.info("All %d members trained.", self.n_members)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Mixture-of-Gaussians predictive distribution.

        Returns
        -------
        mean : ndarray, shape (n,)
        uncertainty : dict
        """
        self._check_trained()
        means, log_vars = [], []
        for m in self.members:
            out = m.predict(X)
            if isinstance(out, tuple):
                mu, lv = out
            else:
                mu, lv = out, np.zeros_like(out)
            means.append(np.asarray(mu))
            log_vars.append(np.asarray(lv))

        means_arr = np.stack(means, axis=0)                    # (M, n)
        vars_arr = np.exp(np.stack(log_vars, axis=0))          # (M, n)

        mixture_mean = means_arr.mean(axis=0)
        epistemic_var = means_arr.var(axis=0)
        aleatoric_var = vars_arr.mean(axis=0)
        total_var = epistemic_var + aleatoric_var

        return mixture_mean, {
            "epistemic_std": np.sqrt(epistemic_var),
            "aleatoric_std": np.sqrt(aleatoric_var),
            "total_std": np.sqrt(total_var),
            "member_means": means_arr,
            "member_vars": vars_arr,
        }

    def predict_quantiles(
        self, X: np.ndarray, quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    ) -> Dict[float, np.ndarray]:
        """Approximate quantiles from the mixture distribution."""
        self._check_trained()
        means, stds = [], []
        for m in self.members:
            out = m.predict(X)
            mu, lv = out if isinstance(out, tuple) else (out, np.zeros(len(X)))
            means.append(np.asarray(mu))
            stds.append(np.sqrt(np.exp(np.asarray(lv)) + 1e-9))

        means_arr = np.stack(means, axis=0)
        stds_arr = np.stack(stds, axis=0)
        # Sample from each component
        rng = np.random.default_rng(0)
        samples = rng.normal(means_arr, stds_arr)   # (M, n)
        return {q: np.quantile(samples, q, axis=0) for q in quantiles}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def member_agreement(self, X: np.ndarray) -> float:
        """Fraction of inputs where all members agree (within 1 std)."""
        self._check_trained()
        preds = np.stack([np.asarray(m.predict(X)) if not isinstance(m.predict(X), tuple)
                          else np.asarray(m.predict(X)[0]) for m in self.members], axis=0)
        std = preds.std(axis=0)
        mean = preds.mean(axis=0)
        return float(np.mean(std < np.abs(mean) * 0.1 + 1e-9))

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Call fit() before predict().")

    def __repr__(self) -> str:
        return (
            f"DeepEnsemble(n_members={self.n_members}, "
            f"adversarial_training={self.adversarial_training})"
        )

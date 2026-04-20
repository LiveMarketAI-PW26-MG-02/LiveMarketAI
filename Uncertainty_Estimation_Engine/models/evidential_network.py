"""
Evidential Deep Learning Network
==================================
Implements evidential regression and classification based on Normal-Inverse-Gamma
and Dirichlet priors respectively (Amini et al. 2020, Sensoy et al. 2018).
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EvidentialNetwork:
    """
    Evidential uncertainty estimator for regression and classification.

    For regression, the network outputs (gamma, nu, alpha, beta) of the
    Normal-Inverse-Gamma (NIG) distribution.

    For classification, outputs (alpha_1, ..., alpha_K) of Dirichlet.
    """

    def __init__(self, task: str = "regression", min_evidence: float = 1e-6):
        assert task in ("regression", "classification")
        self.task = task
        self.min_evidence = min_evidence

    # ------------------------------------------------------------------
    # Regression (NIG)
    # ------------------------------------------------------------------

    def nig_predict(
        self,
        gamma: np.ndarray,
        nu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute predictive mean and decomposed uncertainty from NIG params.

        Returns
        -------
        dict with mean, epistemic_var, aleatoric_var, total_var
        """
        nu = np.maximum(nu, self.min_evidence)
        alpha = np.maximum(alpha, 1.0 + self.min_evidence)

        mean = gamma
        aleatoric_var = beta / (alpha - 1)          # expected variance
        epistemic_var = beta / (nu * (alpha - 1))   # variance of mean

        return {
            "mean": mean,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "total_var": epistemic_var + aleatoric_var,
            "epistemic_std": np.sqrt(epistemic_var),
            "aleatoric_std": np.sqrt(aleatoric_var),
        }

    def nig_loss(
        self,
        y: np.ndarray,
        gamma: np.ndarray,
        nu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        coeff: float = 1.0,
    ) -> float:
        """
        NIG negative log-likelihood loss with evidence regularisation.
        """
        nu = np.maximum(nu, self.min_evidence)
        alpha = np.maximum(alpha, 1.0 + self.min_evidence)
        beta = np.maximum(beta, self.min_evidence)

        omega = 2 * beta * (1 + nu)
        nll = (
            0.5 * np.log(np.pi / nu)
            - alpha * np.log(omega)
            + (alpha + 0.5) * np.log(nu * (y - gamma) ** 2 + omega)
            + np.log(self._gamma_fn(alpha) / self._gamma_fn(alpha + 0.5))
        )
        evidence = 2 * nu + alpha
        reg = np.abs(y - gamma) * evidence
        return float(np.mean(nll + coeff * reg))

    # ------------------------------------------------------------------
    # Classification (Dirichlet)
    # ------------------------------------------------------------------

    def dirichlet_predict(
        self, alpha: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute class probabilities and uncertainty from Dirichlet params.

        Parameters
        ----------
        alpha : ndarray, shape (n, K)
        """
        alpha = np.maximum(alpha, self.min_evidence)
        S = alpha.sum(axis=-1, keepdims=True)         # total evidence
        probs = alpha / S
        # Uncertainty = K / S  (vacuity)
        K = alpha.shape[-1]
        vacuity = K / S.squeeze()
        # Dissonance
        dissonance = self._dissonance(alpha)

        return {
            "probs": probs,
            "predicted_class": probs.argmax(axis=-1),
            "vacuity": vacuity,
            "dissonance": dissonance,
            "total_evidence": S.squeeze(),
            "epistemic_uncertainty": vacuity,
        }

    def dirichlet_loss(
        self,
        y_onehot: np.ndarray,
        alpha: np.ndarray,
        epoch: int = 1,
        annealing_steps: int = 10,
    ) -> float:
        """
        Type II maximum likelihood (Dirichlet NLL) + KL regularisation.
        """
        alpha = np.maximum(alpha, self.min_evidence)
        S = alpha.sum(axis=-1, keepdims=True)
        K = alpha.shape[-1]

        # NLL
        nll = np.sum(y_onehot * (np.log(S) - np.log(alpha)), axis=-1)

        # KL term (annealed)
        coeff = min(1.0, epoch / annealing_steps)
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        S_tilde = alpha_tilde.sum(axis=-1, keepdims=True)
        kl = (
            np.log(self._beta_fn(alpha_tilde))
            - np.log(self._beta_fn(np.ones_like(alpha_tilde)))
            + np.sum((alpha_tilde - 1) * (
                np.log(alpha_tilde) - np.log(S_tilde)
            ), axis=-1)
        )
        return float(np.mean(nll + coeff * kl))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gamma_fn(x: np.ndarray) -> np.ndarray:
        """Stirling approximation for Γ(x)."""
        x = np.maximum(x, 1e-9)
        return np.sqrt(2 * np.pi / x) * (x / np.e) ** x

    @staticmethod
    def _beta_fn(alpha: np.ndarray) -> np.ndarray:
        """Multivariate Beta function B(α)."""
        return np.prod(EvidentialNetwork._gamma_fn(alpha), axis=-1) / \
               EvidentialNetwork._gamma_fn(alpha.sum(axis=-1))

    @staticmethod
    def _dissonance(alpha: np.ndarray) -> np.ndarray:
        """Per-sample belief dissonance (conflict between classes)."""
        S = alpha.sum(axis=-1, keepdims=True)
        b = alpha / S
        K = alpha.shape[-1]
        diss = np.zeros(len(alpha))
        for i in range(K):
            for j in range(K):
                if i != j:
                    sim = 1 - np.abs(b[:, i] - b[:, j]) / (b[:, i] + b[:, j] + 1e-9)
                    diss += b[:, i] * sim * b[:, j]
        return diss

    def __repr__(self) -> str:
        return f"EvidentialNetwork(task='{self.task}')"

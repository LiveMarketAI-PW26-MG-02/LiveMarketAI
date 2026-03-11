"""
Bayesian Uncertainty Estimator
================================
Implements Bayesian inference-based uncertainty estimation using
prior/posterior distributions, likelihood models, and MCMC sampling.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BayesianEstimator:
    """
    Full Bayesian uncertainty estimator supporting conjugate priors,
    MCMC-based posterior sampling, and credible interval computation.
    """

    SUPPORTED_PRIORS = ("normal", "beta", "gamma", "dirichlet", "uniform")

    def __init__(
        self,
        prior: str = "normal",
        n_samples: int = 5000,
        burn_in: int = 1000,
        random_state: Optional[int] = None,
    ):
        if prior not in self.SUPPORTED_PRIORS:
            raise ValueError(f"Prior must be one of {self.SUPPORTED_PRIORS}")
        self.prior = prior
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)
        self._posterior_samples: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianEstimator":
        """Fit the Bayesian model and draw posterior samples via MCMC."""
        logger.info("Fitting BayesianEstimator with %d samples …", len(y))
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        self._posterior_samples = self._mcmc_sample(X, y)
        self._fitted = True
        logger.info("MCMC sampling complete. Effective samples: %d", len(self._posterior_samples))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return predictive mean and standard deviation for each input.

        Returns
        -------
        mean : ndarray, shape (n,)
        std  : ndarray, shape (n,)
        """
        self._check_fitted()
        X = np.atleast_2d(X)
        preds = self._posterior_samples @ X.T          # (samples, n)
        return preds.mean(axis=0), preds.std(axis=0)

    def credible_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (1-alpha) credible intervals."""
        self._check_fitted()
        X = np.atleast_2d(X)
        preds = self._posterior_samples @ X.T
        lower = np.percentile(preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Return epistemic (model) uncertainty as variance of posterior means."""
        self._check_fitted()
        X = np.atleast_2d(X)
        preds = self._posterior_samples @ X.T
        return preds.var(axis=0)

    def aleatoric_uncertainty(self, X: np.ndarray, noise_var: float = 1.0) -> np.ndarray:
        """Return constant aleatoric (data) noise variance."""
        return np.full(X.shape[0], noise_var)

    def total_uncertainty(self, X: np.ndarray, noise_var: float = 1.0) -> np.ndarray:
        """Total uncertainty = epistemic + aleatoric."""
        return self.epistemic_uncertainty(X) + self.aleatoric_uncertainty(X, noise_var)

    def get_posterior_summary(self) -> Dict[str, Any]:
        """Return summary statistics of posterior samples."""
        self._check_fitted()
        s = self._posterior_samples
        return {
            "mean": s.mean(axis=0).tolist(),
            "std": s.std(axis=0).tolist(),
            "median": np.median(s, axis=0).tolist(),
            "q05": np.percentile(s, 5, axis=0).tolist(),
            "q95": np.percentile(s, 95, axis=0).tolist(),
            "n_samples": len(s),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mcmc_sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Random-walk Metropolis–Hastings sampler for linear regression."""
        n_features = X.shape[1]
        samples: List[np.ndarray] = []
        theta = self.rng.standard_normal(n_features)
        log_post_current = self._log_posterior(theta, X, y)
        proposal_std = 0.1

        for step in range(self.n_samples + self.burn_in):
            theta_prop = theta + self.rng.normal(0, proposal_std, size=n_features)
            log_post_prop = self._log_posterior(theta_prop, X, y)
            log_accept = log_post_prop - log_post_current

            if np.log(self.rng.uniform()) < log_accept:
                theta = theta_prop
                log_post_current = log_post_prop

            if step >= self.burn_in:
                samples.append(theta.copy())

        return np.array(samples)

    def _log_posterior(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Log-posterior = log-likelihood + log-prior."""
        residuals = y - X @ theta
        log_lik = -0.5 * np.sum(residuals ** 2)
        log_prior = self._log_prior(theta)
        return log_lik + log_prior

    def _log_prior(self, theta: np.ndarray) -> float:
        if self.prior == "normal":
            return -0.5 * np.sum(theta ** 2)
        elif self.prior == "uniform":
            return 0.0
        elif self.prior == "gamma":
            return float(np.sum(stats.gamma.logpdf(np.abs(theta) + 1e-8, a=2)))
        elif self.prior == "beta":
            clipped = np.clip(theta, 0.01, 0.99)
            return float(np.sum(stats.beta.logpdf(clipped, a=2, b=2)))
        return 0.0

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def __repr__(self) -> str:
        return (
            f"BayesianEstimator(prior='{self.prior}', "
            f"n_samples={self.n_samples}, burn_in={self.burn_in})"
        )

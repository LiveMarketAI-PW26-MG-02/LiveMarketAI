"""
Variational Inference for Bayesian neural networks.
Minimizes KL divergence between approximate and true posterior.
"""
import numpy as np
from typing import Tuple, Callable
import logging
logger = logging.getLogger(__name__)


class ELBO:
    @staticmethod
    def gaussian_kl(mu_q, sigma_q, mu_p=0.0, sigma_p=1.0):
        return float(0.5 * np.sum(
            np.log(sigma_p**2 / sigma_q**2)
            + (sigma_q**2 + (mu_q - mu_p)**2) / sigma_p**2 - 1))

    @staticmethod
    def log_likelihood_gaussian(y, pred, sigma=1.0):
        n = len(y)
        return float(-0.5*n*np.log(2*np.pi*sigma**2) - 0.5*np.sum((y-pred)**2)/sigma**2)

    def compute(self, y, pred, mu_q, sigma_q, n_data):
        ll = self.log_likelihood_gaussian(y, pred)
        kl = self.gaussian_kl(mu_q, sigma_q)
        return ll - kl / n_data


class VariationalInference:
    """Mean-field variational inference. Optimises ELBO via gradient ascent."""

    def __init__(self, n_iter=1000, lr=0.01, n_samples=10):
        self.n_iter = n_iter
        self.lr = lr
        self.n_samples = n_samples
        self.elbo_computer = ELBO()
        self._elbo_history = []

    def fit(self, log_joint, init_params, n_params):
        mu = init_params.copy()
        log_sigma = np.zeros_like(mu) - 1.0
        for i in range(self.n_iter):
            sigma = np.exp(log_sigma)
            elbo_vals, grad_mu, grad_ls = [], np.zeros_like(mu), np.zeros_like(log_sigma)
            for _ in range(self.n_samples):
                eps = np.random.randn(n_params)
                params = mu + sigma * eps
                lj = log_joint(params)
                kl = ELBO.gaussian_kl(mu, sigma)
                elbo_vals.append(lj - kl)
                grad_mu += (params - mu) / sigma**2
                grad_ls += eps * sigma - 1.0
            mu += self.lr * grad_mu / self.n_samples
            log_sigma += self.lr * grad_ls / self.n_samples
            self._elbo_history.append(float(np.mean(elbo_vals)))
        return mu, np.exp(log_sigma)

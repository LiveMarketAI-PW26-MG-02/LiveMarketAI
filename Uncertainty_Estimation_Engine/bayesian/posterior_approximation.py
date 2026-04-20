"""Posterior approximation: mean-field and Laplace."""
import numpy as np
from scipy.optimize import minimize


class MeanFieldApproximation:
    def __init__(self, n_params):
        self.mu = np.zeros(n_params)
        self.log_sigma = np.full(n_params, -1.0)

    @property
    def sigma(self): return np.exp(self.log_sigma)

    def sample(self, n=1):
        return self.mu + self.sigma * np.random.randn(n, len(self.mu))

    def entropy(self):
        return float(0.5*np.sum(np.log(2*np.pi*np.e*self.sigma**2)))

    def kl_to_standard_normal(self):
        return float(0.5*np.sum(self.mu**2 + self.sigma**2 - 2*self.log_sigma - 1))


class LaplaceApproximation:
    def __init__(self):
        self.map_estimate = None
        self.covariance = None

    def fit(self, log_posterior, init):
        res = minimize(lambda x: -log_posterior(x), x0=init, method="L-BFGS-B")
        self.map_estimate = res.x
        n, eps = len(init), 1e-5
        H = np.zeros((n, n))
        f0 = -log_posterior(self.map_estimate)
        for i in range(n):
            xp = self.map_estimate.copy(); xp[i] += eps
            xm = self.map_estimate.copy(); xm[i] -= eps
            H[i,i] = (-log_posterior(xp) - 2*f0 + (-log_posterior(xm))) / eps**2
        self.covariance = np.linalg.pinv(H + 1e-6*np.eye(n))
        return self

    def sample(self, n=100):
        return np.random.multivariate_normal(self.map_estimate, self.covariance, size=n)

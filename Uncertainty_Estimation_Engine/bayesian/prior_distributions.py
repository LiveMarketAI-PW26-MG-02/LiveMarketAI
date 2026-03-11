"""Prior distributions for Bayesian models."""
import numpy as np
from abc import ABC, abstractmethod


class BasePrior(ABC):
    @abstractmethod
    def log_prob(self, x): ...
    @abstractmethod
    def sample(self, shape): ...


class GaussianPrior(BasePrior):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu, self.sigma = mu, sigma

    def log_prob(self, x):
        return float(-0.5*np.sum(((x-self.mu)/self.sigma)**2)
                     - len(x)*np.log(self.sigma*np.sqrt(2*np.pi)))

    def sample(self, shape):
        return self.mu + self.sigma * np.random.randn(*shape)


class LaplacePrior(BasePrior):
    def __init__(self, mu=0.0, b=1.0):
        self.mu, self.b = mu, b

    def log_prob(self, x):
        return float(-np.sum(np.abs(x-self.mu))/self.b - len(x)*np.log(2*self.b))

    def sample(self, shape):
        return np.random.laplace(self.mu, self.b, shape)


class HorseshoePrior(BasePrior):
    """Horseshoe prior for sparse Bayesian regression."""
    def __init__(self, tau=1.0):
        self.tau = tau

    def log_prob(self, x):
        lam_sq = np.abs(np.random.cauchy(size=x.shape))**2
        sigma_sq = lam_sq * self.tau**2
        return float(-0.5*np.sum(x**2/sigma_sq + np.log(sigma_sq)))

    def sample(self, shape):
        lam = np.abs(np.random.standard_cauchy(shape))
        return np.random.randn(*shape) * lam * self.tau

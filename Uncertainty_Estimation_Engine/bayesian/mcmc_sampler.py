"""
MCMC sampling: Metropolis-Hastings and base class.
"""
import numpy as np
from typing import Callable, Tuple, Optional
import logging
logger = logging.getLogger(__name__)


class MCMCSampler:
    def __init__(self, n_samples=1000, burn_in=200):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self._samples = None
        self._acceptance_rate = 0.0

    def sample(self, log_prob, init): raise NotImplementedError

    @property
    def samples(self): return self._samples

    def posterior_mean(self):
        return self._samples.mean(axis=0)

    def posterior_std(self):
        return self._samples.std(axis=0)

    def credible_interval(self, alpha=0.05):
        lo = np.percentile(self._samples, 100*alpha/2, axis=0)
        hi = np.percentile(self._samples, 100*(1-alpha/2), axis=0)
        return lo, hi


class MetropolisHastings(MCMCSampler):
    def __init__(self, n_samples=1000, burn_in=200, step_size=0.1):
        super().__init__(n_samples, burn_in)
        self.step_size = step_size

    def sample(self, log_prob: Callable, init: np.ndarray) -> np.ndarray:
        dim = len(init)
        current = init.copy()
        current_lp = log_prob(current)
        total = self.n_samples + self.burn_in
        chain = np.zeros((total, dim))
        n_accepted = 0
        for i in range(total):
            prop = current + self.step_size * np.random.randn(dim)
            prop_lp = log_prob(prop)
            if np.log(np.random.rand()) < prop_lp - current_lp:
                current, current_lp = prop, prop_lp
                n_accepted += 1
            chain[i] = current
        self._acceptance_rate = n_accepted / total
        self._samples = chain[self.burn_in:]
        logger.info("MH acceptance: %.1f%%", self._acceptance_rate * 100)
        return self._samples

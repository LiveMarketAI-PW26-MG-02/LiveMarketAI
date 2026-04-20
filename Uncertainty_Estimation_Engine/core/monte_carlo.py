"""
Monte Carlo Dropout-based Uncertainty Estimation.
Enables uncertainty by keeping dropout active during inference.
"""

import numpy as np
from typing import Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloEstimator:
    """Estimates uncertainty via Monte Carlo sampling."""

    def __init__(self, n_samples: int = 100, seed: Optional[int] = None):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def estimate(self, forward_fn: Callable, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_preds, all_vars = [], []
        for _ in range(self.n_samples):
            out = forward_fn(X)
            if isinstance(out, tuple):
                pred, var = out
                all_vars.append(var)
            else:
                pred = out
            all_preds.append(pred)
        preds_array = np.stack(all_preds, axis=0)
        mean_pred = preds_array.mean(axis=0)
        epistemic = preds_array.var(axis=0)
        aleatoric = np.stack(all_vars).mean(axis=0) if all_vars else np.zeros_like(epistemic)
        return mean_pred, epistemic, aleatoric

    def sample_predictions(self, forward_fn: Callable, X: np.ndarray) -> np.ndarray:
        return np.stack([forward_fn(X) for _ in range(self.n_samples)], axis=0)

    def entropy_from_samples(self, samples: np.ndarray, n_bins: int = 20) -> np.ndarray:
        n_mc, n_data = samples.shape
        entropies = np.zeros(n_data)
        for i in range(n_data):
            hist, _ = np.histogram(samples[:, i], bins=n_bins, density=True)
            hist = hist[hist > 0]
            entropies[i] = -np.sum(hist * np.log(hist + 1e-10)) / n_bins
        return entropies

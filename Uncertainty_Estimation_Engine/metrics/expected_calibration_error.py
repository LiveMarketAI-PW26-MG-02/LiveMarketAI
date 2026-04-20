"""Expected Calibration Error with adaptive binning."""
import numpy as np


class ECEComputer:
    def __init__(self, adaptive: bool = False, n_bins: int = 10):
        self.adaptive = adaptive
        self.n_bins = n_bins

    def compute(self, probs: np.ndarray, y: np.ndarray) -> float:
        if self.adaptive:
            return self._adaptive_ece(probs, y)
        return self._fixed_ece(probs, y)

    def _fixed_ece(self, probs, y):
        n = len(y)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (probs >= lo) & (probs < hi)
            if m.sum():
                ece += m.sum() * abs(y[m].mean() - probs[m].mean()) / n
        return float(ece)

    def _adaptive_ece(self, probs, y):
        n = len(y)
        order = np.argsort(probs)
        bins = np.array_split(order, self.n_bins)
        ece = 0.0
        for b in bins:
            if len(b):
                ece += len(b) * abs(y[b].mean() - probs[b].mean()) / n
        return float(ece)

"""Evaluate calibration quality of probabilistic predictions."""
import numpy as np
from typing import Tuple


class CalibrationEvaluator:
    """Computes ECE, MCE, and reliability diagram data."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def expected_calibration_error(self, probs: np.ndarray, y: np.ndarray) -> float:
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(y)
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc = y[mask].mean()
            conf = probs[mask].mean()
            ece += mask.sum() * abs(acc - conf) / n
        return float(ece)

    def maximum_calibration_error(self, probs: np.ndarray, y: np.ndarray) -> float:
        bins = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0
        for i in range(self.n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() == 0:
                continue
            mce = max(mce, abs(y[mask].mean() - probs[mask].mean()))
        return float(mce)

    def reliability_data(self, probs, y):
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []
        for i in range(self.n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            bin_counts.append(int(mask.sum()))
            if mask.sum():
                bin_accs.append(float(y[mask].mean()))
                bin_confs.append(float(probs[mask].mean()))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bins[i]+bins[i+1])/2)
        return {"accuracies": bin_accs, "confidences": bin_confs, "counts": bin_counts}

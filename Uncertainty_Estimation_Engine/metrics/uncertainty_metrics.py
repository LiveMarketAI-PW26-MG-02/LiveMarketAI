"""Comprehensive uncertainty quality metrics."""
import numpy as np
from typing import Dict


class UncertaintyMetrics:
    """Collects scalar quality measures for uncertainty estimates."""

    @staticmethod
    def sharpness(uncertainty: np.ndarray) -> float:
        """Mean uncertainty – lower is sharper."""
        return float(np.mean(uncertainty))

    @staticmethod
    def dispersion(uncertainty: np.ndarray) -> float:
        return float(np.std(uncertainty))

    @staticmethod
    def nll_gaussian(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Gaussian NLL – measures calibration of predictive distribution."""
        return float(np.mean(0.5 * np.log(2*np.pi*std**2) + (y-mean)**2 / (2*std**2)))

    @staticmethod
    def crps(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Continuous Ranked Probability Score (analytical Gaussian)."""
        from scipy.stats import norm
        z = (y - mean) / std
        return float(np.mean(std * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))))

    @staticmethod
    def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.mean((y >= lower) & (y <= upper)))

    @staticmethod
    def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.mean(upper - lower))

    def evaluate_all(self, y, mean, std, lower=None, upper=None) -> Dict[str, float]:
        out = {
            "sharpness": self.sharpness(std),
            "dispersion": self.dispersion(std),
            "nll": self.nll_gaussian(y, mean, std),
            "crps": self.crps(y, mean, std),
        }
        if lower is not None and upper is not None:
            out["coverage"] = self.coverage(y, lower, upper)
            out["interval_width"] = self.interval_width(lower, upper)
        return out

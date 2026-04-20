"""
Uncertainty Metrics
====================
Comprehensive suite of metrics for evaluating the quality of uncertainty estimates,
including calibration, sharpness, resolution, and proper scoring rules.
"""

import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class UncertaintyMetrics:
    """
    Evaluate probabilistic predictions via calibration, sharpness,
    proper scoring rules, and information-theoretic metrics.
    """

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 15,
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) and Maximum CE (MCE).

        Parameters
        ----------
        confidences : ndarray, shape (n,)   predicted probabilities
        accuracies  : ndarray, shape (n,)   binary correctness labels
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_data = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            prop = mask.sum() / len(confidences)
            gap = abs(acc - conf)
            ece += prop * gap
            mce = max(mce, gap)
            bin_data.append({"lo": lo, "hi": hi, "accuracy": acc, "confidence": conf, "n": int(mask.sum())})

        return {"ece": float(ece), "mce": float(mce), "bin_data": bin_data}

    @staticmethod
    def reliability_diagram_data(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """Return data needed to draw a reliability diagram."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_confs, bin_accs, bin_counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_confs.append(float(confidences[mask].mean()))
            bin_accs.append(float(accuracies[mask].mean()))
            bin_counts.append(int(mask.sum()))
        return {"bin_confidences": bin_confs, "bin_accuracies": bin_accs, "bin_counts": bin_counts}

    # ------------------------------------------------------------------
    # Proper scoring rules
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier score (lower is better)."""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def log_loss(
        y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15
    ) -> float:
        """Binary cross-entropy log loss."""
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    @staticmethod
    def crps(
        y_true: np.ndarray,
        forecast_samples: np.ndarray,
    ) -> float:
        """
        Continuous Ranked Probability Score (CRPS) — energy score form.

        Parameters
        ----------
        forecast_samples : ndarray, shape (M, n)
            M ensemble members, n observations.
        """
        M = forecast_samples.shape[0]
        term1 = np.mean(np.abs(forecast_samples - y_true[None, :]), axis=0)
        pair_diffs = 0.0
        for i in range(M):
            for j in range(i + 1, M):
                pair_diffs += np.abs(forecast_samples[i] - forecast_samples[j])
        term2 = pair_diffs / (M * (M - 1) / 2)
        return float(np.mean(term1 - 0.5 * term2))

    # ------------------------------------------------------------------
    # Sharpness / Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def sharpness(std_preds: np.ndarray) -> Dict[str, float]:
        """Measures how narrow predictive distributions are."""
        return {
            "mean_std": float(std_preds.mean()),
            "median_std": float(np.median(std_preds)),
            "sharpness_score": float(1.0 / (std_preds.mean() + 1e-9)),
        }

    @staticmethod
    def interval_score(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.1,
    ) -> float:
        """
        Interval Score: penalizes width + out-of-interval misses.
        Lower is better.
        """
        width = upper - lower
        below = np.maximum(lower - y_true, 0)
        above = np.maximum(y_true - upper, 0)
        return float(np.mean(width + (2 / alpha) * below + (2 / alpha) * above))

    # ------------------------------------------------------------------
    # Information-theoretic
    # ------------------------------------------------------------------

    @staticmethod
    def predictive_entropy(probs: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        """Shannon entropy of predictive distribution."""
        probs = np.clip(probs, eps, 1.0)
        return -np.sum(probs * np.log(probs), axis=-1)

    @staticmethod
    def mutual_information(
        mc_probs: np.ndarray, eps: float = 1e-15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutual Information = H[y|x] − E_θ[H[y|x,θ]]
        mc_probs : ndarray, shape (T, n, K)
        Returns (mutual_information, expected_entropy)
        """
        mean_probs = mc_probs.mean(axis=0)                         # (n, K)
        predictive_H = UncertaintyMetrics.predictive_entropy(mean_probs)
        expected_H = np.mean(
            [UncertaintyMetrics.predictive_entropy(mc_probs[t]) for t in range(mc_probs.shape[0])],
            axis=0,
        )
        mi = predictive_H - expected_H
        return mi, expected_H

    @staticmethod
    def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Total variation distance between two discrete distributions."""
        return 0.5 * float(np.sum(np.abs(p - q)))

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    @staticmethod
    def full_report(
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ) -> Dict:
        """Generate a comprehensive uncertainty evaluation report."""
        z = 1.645  # 90% CI
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

        report = {
            "rmse": float(np.sqrt(np.mean((y_true - y_pred_mean) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred_mean))),
            "interval_coverage": coverage,
            "mean_interval_width": float(np.mean(upper - lower)),
            "interval_score": UncertaintyMetrics.interval_score(y_true, lower, upper, alpha),
            **UncertaintyMetrics.sharpness(y_pred_std),
        }
        if y_pred_proba is not None:
            report["brier_score"] = UncertaintyMetrics.brier_score(y_true, y_pred_proba)
            report["log_loss"] = UncertaintyMetrics.log_loss(y_true, y_pred_proba)
        return report

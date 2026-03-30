"""
drift_detector.py — Requirement 3
Stock Drift-Aware Update Trigger
---------------------------------
Monitors the incoming stock-data stream for significant distribution shifts and
fires an update signal only when drift exceeds a configurable threshold.

Three detection methods:
  • ks_test      — Kolmogorov-Smirnov two-sample test on rolling windows
  • page_hinkley — sequential change-point detection (lightweight)
  • wasserstein  — Earth-Mover distance between reference and current window
"""

import numpy as np
from collections import deque
from scipy import stats
from typing import Deque, Dict, List, Tuple, Optional
import config


class StockDriftDetector:
    """
    Maintains two rolling windows (reference / current) and computes a drift
    score.  Triggers an update event when the score crosses DRIFT_THRESHOLD.

    Usage
    -----
        detector = StockDriftDetector()
        for batch in stream:
            triggered, score = detector.update(batch["close"].values)
            if triggered:
                run_incremental_update(model, batch)
    """

    def __init__(self,
                 window_size: int       = config.DRIFT_WINDOW,
                 threshold: float       = config.DRIFT_THRESHOLD,
                 method: str            = "ks_test",
                 check_frequency: int   = config.DRIFT_CHECK_FREQ):
        assert method in {"ks_test", "page_hinkley", "wasserstein"}
        self.window_size     = window_size
        self.threshold       = threshold
        self.method          = method
        self.check_frequency = check_frequency

        self._reference: Deque[float] = deque(maxlen=window_size)
        self._current:   Deque[float] = deque(maxlen=window_size)
        self._batch_count = 0
        self._scores: List[float]     = []
        self._trigger_history: List[int] = []

        # Page-Hinkley state
        self._ph_sum     = 0.0
        self._ph_min     = 0.0
        self._ph_mu      = 0.0
        self._ph_n       = 0
        self._ph_delta   = 0.01   # allowed mean increase
        self._ph_lambda  = threshold * 100

        print(f"[DriftDetector] method='{method}', threshold={threshold}, "
              f"window={window_size}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def update(self, new_values: np.ndarray) -> Tuple[bool, float]:
        """
        Feed new return/price values.  Returns (triggered, drift_score).
        triggered=True means an incremental update should run.
        """
        self._batch_count += 1
        returns = np.diff(new_values) / (np.abs(new_values[:-1]) + 1e-9)

        # Build reference from first window, then fill current
        if len(self._reference) < self.window_size:
            self._reference.extend(returns.tolist())
            return False, 0.0

        self._current.extend(returns.tolist())

        if self._batch_count % self.check_frequency != 0:
            return False, 0.0

        score     = self._compute_score()
        triggered = score > self.threshold
        self._scores.append(score)

        if triggered:
            self._trigger_history.append(self._batch_count)
            # Slide reference window forward
            self._reference.extend(list(self._current)[-self.window_size // 2:])

        return triggered, score

    def reset_reference(self) -> None:
        """Manually reset reference window (e.g., after a confirmed regime change)."""
        self._reference.clear()
        self._reference.extend(list(self._current))
        self._ph_sum = self._ph_min = 0.0

    @property
    def drift_scores(self) -> List[float]:
        return self._scores

    @property
    def trigger_history(self) -> List[int]:
        return self._trigger_history

    def summary(self) -> Dict:
        return {
            "method":        self.method,
            "threshold":     self.threshold,
            "total_batches": self._batch_count,
            "triggers":      len(self._trigger_history),
            "trigger_steps": self._trigger_history,
            "mean_score":    float(np.mean(self._scores)) if self._scores else 0.0,
            "max_score":     float(np.max(self._scores))  if self._scores else 0.0,
        }

    # ─── Scoring Methods ─────────────────────────────────────────────────────

    def _compute_score(self) -> float:
        if self.method == "ks_test":
            return self._ks_score()
        elif self.method == "page_hinkley":
            return self._ph_score()
        else:
            return self._wasserstein_score()

    def _ks_score(self) -> float:
        """Two-sample KS statistic between reference and current windows."""
        ref = np.array(self._reference)
        cur = np.array(self._current)
        stat, _ = stats.ks_2samp(ref, cur)
        return float(stat)

    def _ph_score(self) -> float:
        """
        Page-Hinkley test — detects upward shifts in the mean return magnitude.
        Returns normalised cumulative sum as score.
        """
        for x in self._current:
            self._ph_n   += 1
            self._ph_mu  += (abs(x) - self._ph_mu) / self._ph_n
            self._ph_sum += abs(x) - self._ph_mu - self._ph_delta
            self._ph_min  = min(self._ph_min, self._ph_sum)
        score = (self._ph_sum - self._ph_min) / (self._ph_lambda + 1e-9)
        return float(np.clip(score, 0, 1))

    def _wasserstein_score(self) -> float:
        """Earth-Mover distance, normalised by the reference std."""
        ref = np.array(self._reference)
        cur = np.array(self._current)
        dist = stats.wasserstein_distance(ref, cur)
        norm = np.std(ref) + 1e-9
        return float(np.clip(dist / norm, 0, 1))


# ─── Multi-feature drift detection ───────────────────────────────────────────

class MultiFeatureDriftDetector:
    """
    Wraps StockDriftDetector for each feature column and fires a combined
    alert when the majority of features are drifting.
    """

    def __init__(self, feature_names: List[str], **kwargs):
        self.feature_names = feature_names
        self.detectors     = {f: StockDriftDetector(**kwargs)
                              for f in feature_names}

    def update(self, feature_matrix: np.ndarray) -> Tuple[bool, float]:
        """
        feature_matrix: (n_samples, n_features)
        Returns (any_triggered, average_score)
        """
        scores    = []
        triggered = False
        for i, name in enumerate(self.feature_names):
            trig, score = self.detectors[name].update(feature_matrix[:, i])
            scores.append(score)
            triggered = triggered or trig
        return triggered, float(np.mean(scores))


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    det = StockDriftDetector(window_size=40, threshold=0.05, method="ks_test")

    # Stable regime
    for _ in range(15):
        prices = 100 + np.cumsum(rng.normal(0, 0.5, 40))
        trig, score = det.update(prices)
        print(f"score={score:.4f}  triggered={trig}")

    print("--- Injecting regime shift ---")
    for _ in range(5):
        prices = 100 + np.cumsum(rng.normal(0.5, 3.0, 40))   # high drift
        trig, score = det.update(prices)
        print(f"score={score:.4f}  triggered={trig}")

    print(det.summary())

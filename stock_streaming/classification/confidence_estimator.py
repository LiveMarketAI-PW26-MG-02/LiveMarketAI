"""
confidence_estimator.py — Tracks and dynamically adjusts prediction confidence
as new data enters the streaming window.  Confidence evolves via EMA blending
with a calibration correction derived from recent prediction accuracy.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple
from core.config import CFG


class CalibrationTracker:
    """
    Rolling calibration: measures empirical accuracy over recent predictions
    and returns a correction multiplier for raw model confidence.
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._records: deque = deque(maxlen=window)  # (predicted_label, actual_label)

    def record(self, predicted: str, actual: str) -> None:
        self._records.append(predicted == actual)

    def calibration_factor(self) -> float:
        """Returns a factor in (0, 1.2] — higher when model is accurate."""
        if len(self._records) < 10:
            return 1.0
        empirical_accuracy = sum(self._records) / len(self._records)
        # Platt-style linear mapping: acc=0.5 → 0.7, acc=1.0 → 1.2
        factor = 0.4 + 1.6 * empirical_accuracy
        return min(1.2, max(0.3, factor))


class ConfidenceEstimator:
    """
    Maintains a smoothed, calibrated confidence score for each label class.

    How it works:
        1. Raw model confidence is received per prediction.
        2. An EMA smooths the confidence signal over time.
        3. A calibration factor adjusts upward/downward based on accuracy.
        4. Entropy of the probability distribution modulates final confidence.
        5. Window-fill ratio boosts confidence as more data enters the window.
    """

    def __init__(self, alpha: Optional[float] = None):
        self._alpha = alpha or CFG.classifier.confidence_ema_alpha
        self._ema: Dict[str, float] = {l: 0.5 for l in CFG.classifier.labels}
        self._calibration = CalibrationTracker()
        self._last_label: Optional[str] = None
        self._history: deque = deque(maxlen=200)
        self._updates = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, label: str, raw_confidence: float,
               proba_dict: Optional[Dict[str, float]] = None,
               window_fill_pct: float = 100.0) -> Tuple[str, float]:
        """
        Update internal state and return (final_label, final_confidence).

        Parameters
        ----------
        label            : Highest-probability label from the model.
        raw_confidence   : Corresponding probability.
        proba_dict       : Full probability distribution (optional).
        window_fill_pct  : How full is the current window (0–100)?
        """
        # 1. EMA smooth per label
        for l, raw in (proba_dict or {label: raw_confidence}).items():
            if l in self._ema:
                self._ema[l] = self._alpha * raw + (1 - self._alpha) * self._ema[l]

        # Choose label by smoothed probability
        smoothed_label = max(self._ema, key=self._ema.get)
        smoothed_conf  = self._ema[smoothed_label]

        # 2. Entropy penalty — high uncertainty → lower confidence
        entropy_penalty = self._entropy_penalty(proba_dict)

        # 3. Calibration factor from empirical accuracy
        cal_factor = self._calibration.calibration_factor()

        # 4. Window-fill boost: confidence scales up as window fills
        fill_boost = 0.7 + 0.3 * min(1.0, window_fill_pct / 100.0)

        # 5. Compose
        final_conf = min(0.99, smoothed_conf * (1 - entropy_penalty) * cal_factor * fill_boost)
        final_conf = max(0.01, final_conf)

        self._last_label = smoothed_label
        self._history.append((smoothed_label, round(final_conf, 4)))
        self._updates += 1
        return smoothed_label, final_conf

    def record_outcome(self, predicted: str, actual: str) -> None:
        """Feed the actual label outcome back to calibrate future estimates."""
        self._calibration.record(predicted, actual)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def confidence_history(self) -> List[float]:
        return [c for _, c in self._history]

    def label_history(self) -> List[str]:
        return [l for l, _ in self._history]

    def current_ema_distribution(self) -> Dict[str, float]:
        return dict(self._ema)

    def diagnostics(self) -> Dict:
        hist = self.confidence_history()
        return {
            "updates":           self._updates,
            "mean_confidence":   round(sum(hist) / len(hist), 4) if hist else 0.0,
            "last_label":        self._last_label,
            "calibration_factor": round(self._calibration.calibration_factor(), 3),
            "ema_dist":          {k: round(v, 4) for k, v in self._ema.items()},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_penalty(proba_dict: Optional[Dict[str, float]]) -> float:
        """
        Returns a penalty in [0, 1].  Max entropy (uniform) → penalty ≈ 0.4.
        """
        if not proba_dict:
            return 0.0
        probs = [p for p in proba_dict.values() if p > 0]
        n = len(probs)
        if n <= 1:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(n)
        normalised = entropy / (max_entropy + 1e-12)
        return 0.4 * normalised  # scale: at most 40% confidence penalty


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(0)
    labels = CFG.classifier.labels
    est = ConfidenceEstimator(alpha=0.3)

    for i in range(150):
        raw_label = rng.choice(labels)
        raw_conf  = rng.uniform(0.3, 0.9)
        fill      = min(100.0, i * 2.0)
        out_label, out_conf = est.update(raw_label, raw_conf, window_fill_pct=fill)
        if i % 20 == 0:
            print(f"step={i:3d} raw={raw_label:<12} out={out_label:<12} conf={out_conf:.3f}")

    print(est.diagnostics())

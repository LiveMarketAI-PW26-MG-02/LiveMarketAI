"""
base_classifier.py — Abstract base class and shared utilities for all classifiers
in the stock streaming system.  Concrete classifiers extend BaseClassifier and
implement predict_raw().
"""

import time
import abc
from typing import Dict, List, Optional, Tuple
from core.config import CFG


# Label constants match CFG.classifier.labels
LABELS = CFG.classifier.labels


class PredictionResult:
    """Structured output from a single classifier call."""

    def __init__(self, label: str, confidence: float,
                 probabilities: Optional[Dict[str, float]] = None,
                 elapsed_ms: float = 0.0):
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities or {l: 0.0 for l in LABELS}
        self.elapsed_ms = elapsed_ms
        self.timed_out = False

    @property
    def is_bullish(self) -> bool:
        return self.label in ("STRONG_BUY", "BUY")

    @property
    def is_bearish(self) -> bool:
        return self.label in ("SELL", "STRONG_SELL")

    def __str__(self) -> str:
        return (f"PredictionResult(label={self.label}, "
                f"confidence={self.confidence:.3f}, "
                f"elapsed={self.elapsed_ms:.2f}ms)")


class BaseClassifier(abc.ABC):
    """
    Abstract base for streaming and batch classifiers.

    Subclasses implement:
        fit(X, y)              — train / update the model
        predict_raw(features)  — return (label, confidence, proba_dict)
    """

    def __init__(self):
        self._fitted = False
        self._predict_count = 0
        self._timeout_count = 0
        self._total_latency_ms = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, X: List[Dict[str, float]], y: List[str]) -> None:
        """Train or incrementally update the model."""

    @abc.abstractmethod
    def predict_raw(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Return (label, confidence, {label: probability}).
        Must be implemented without any I/O or blocking calls.
        """

    # ------------------------------------------------------------------
    # Public predict with latency budget
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float],
                deadline: Optional[float] = None) -> Tuple[str, float]:
        """
        Calls predict_raw() and returns (label, confidence).
        Enforces `deadline` (perf_counter timestamp); if exceeded,
        falls back to the fastest available heuristic.
        """
        t0 = time.perf_counter()

        if not self._fitted:
            # Bootstrap heuristic: use z_score if model not ready
            label, conf = self._heuristic_predict(features)
        else:
            if deadline and time.perf_counter() >= deadline:
                label, conf = self._heuristic_predict(features)
                self._timeout_count += 1
            else:
                label, conf, _ = self.predict_raw(features)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += elapsed_ms
        self._predict_count += 1
        return label, conf

    # ------------------------------------------------------------------
    # Shared heuristic (used before model is trained or on timeout)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_predict(features: Dict[str, float]) -> Tuple[str, float]:
        """Rule-based fallback: momentum + z-score + RSI."""
        m5    = features.get("momentum_5", 0.0)
        z     = features.get("z_score", 0.0)
        rsi   = features.get("rsi", 50.0)
        score = m5 * 0.5 + (rsi - 50) / 100.0 * 0.3 + z * 0.2

        if score >  0.04:  return "STRONG_BUY",  min(0.9, 0.5 + score * 5)
        if score >  0.01:  return "BUY",          min(0.75, 0.5 + score * 5)
        if score < -0.04:  return "STRONG_SELL",  min(0.9, 0.5 - score * 5)
        if score < -0.01:  return "SELL",         min(0.75, 0.5 - score * 5)
        return "HOLD", 0.5

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict:
        avg_lat = self._total_latency_ms / max(1, self._predict_count)
        return {
            "fitted":            self._fitted,
            "predict_count":     self._predict_count,
            "timeout_count":     self._timeout_count,
            "avg_latency_ms":    round(avg_lat, 3),
        }

    # ------------------------------------------------------------------
    # Helpers shared across subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def features_to_vector(features: Dict[str, float]) -> List[float]:
        """Stable feature ordering for numpy / sklearn."""
        keys = sorted(features.keys())
        return [features[k] for k in keys]

    @staticmethod
    def label_to_int(label: str) -> int:
        return LABELS.index(label) if label in LABELS else 2  # default HOLD

    @staticmethod
    def int_to_label(idx: int) -> str:
        return LABELS[idx] if 0 <= idx < len(LABELS) else "HOLD"

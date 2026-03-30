"""
state_aware_predictor.py — Wraps any BaseClassifier and blends the model's raw
output with a state prior derived from recent historical predictions.
Prevents rapid signal flipping and stabilises predictions during noisy regimes.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple
from classification.base_classifier import BaseClassifier, LABELS
from core.config import CFG


class TransitionMatrix:
    """
    Learns the empirical label-to-label transition probabilities from the
    stream of past predictions and uses them as a prior for the next step.
    """

    def __init__(self, labels: List[str], smoothing: float = 1.0):
        self._labels  = labels
        self._n       = len(labels)
        self._idx     = {l: i for i, l in enumerate(labels)}
        # Laplace-smoothed count matrix
        self._counts  = [[smoothing] * self._n for _ in range(self._n)]
        self._last    = None

    def observe(self, label: str) -> None:
        if self._last is not None and label in self._idx:
            i = self._idx[self._last]
            j = self._idx[label]
            self._counts[i][j] += 1
        self._last = label

    def transition_prior(self, from_label: str) -> Dict[str, float]:
        """P(next_label | from_label) as a normalised probability dict."""
        if from_label not in self._idx:
            return {l: 1.0 / self._n for l in self._labels}
        row = self._counts[self._idx[from_label]]
        total = sum(row)
        return {l: row[self._idx[l]] / total for l in self._labels}


class StateAwarePredictor:
    """
    Decorator around any BaseClassifier that conditions predictions on the
    recent prediction state.

    The blending formula:
        P_final(label) = (1 - beta) * P_model(label) + beta * P_transition(label)

    where beta is modulated by:
        - How recently the label changed (change_recency_factor)
        - Current volatility regime (high vol → lower beta, trust model more)
    """

    def __init__(self, classifier: BaseClassifier,
                 memory_length: Optional[int] = None,
                 base_beta: float = 0.25):
        self._clf     = classifier
        self._mem_len = memory_length or CFG.classifier.state_memory_length
        self._base_beta = base_beta
        self._history: deque = deque(maxlen=self._mem_len)
        self._transition = TransitionMatrix(LABELS)
        self._last_label: Optional[str] = None
        self._label_change_count = 0
        self._total_calls = 0
        self._flip_rate_buf: deque = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float],
                deadline: Optional[float] = None) -> Tuple[str, float]:
        """
        Blended prediction incorporating state prior.
        Returns (final_label, final_confidence).
        """
        self._total_calls += 1

        # 1. Raw model prediction
        raw_label, raw_conf, proba_dict = self._raw_predict(features, deadline)

        # 2. State prior from transition matrix
        if self._last_label:
            prior = self._transition.transition_prior(self._last_label)
        else:
            prior = {l: 1.0 / len(LABELS) for l in LABELS}

        # 3. Modulate beta
        beta = self._compute_beta(features)

        # 4. Blend probabilities
        model_proba = self._ensure_full_proba(proba_dict, raw_label, raw_conf)
        blended = {}
        for l in LABELS:
            blended[l] = (1 - beta) * model_proba.get(l, 0.0) + beta * prior.get(l, 0.0)

        # Normalise
        total = sum(blended.values()) + 1e-12
        blended = {l: v / total for l, v in blended.items()}

        final_label = max(blended, key=blended.get)
        final_conf  = blended[final_label]

        # 5. Track state
        self._update_state(final_label)

        return final_label, round(final_conf, 4)

    # ------------------------------------------------------------------
    # Delegate fit and update
    # ------------------------------------------------------------------

    def fit(self, X: List[Dict[str, float]], y: List[str]) -> None:
        self._clf.fit(X, y)

    def update(self, features: Dict[str, float], label: str) -> None:
        if hasattr(self._clf, "update"):
            self._clf.update(features, label)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def flip_rate(self) -> float:
        """Fraction of consecutive calls where the label changed."""
        if len(self._flip_rate_buf) < 2:
            return 0.0
        flips = sum(1 for i in range(1, len(self._flip_rate_buf))
                    if self._flip_rate_buf[i] != self._flip_rate_buf[i-1])
        return flips / (len(self._flip_rate_buf) - 1)

    def diagnostics(self) -> Dict:
        d = self._clf.diagnostics()
        d.update({
            "state_total_calls":     self._total_calls,
            "label_change_count":    self._label_change_count,
            "current_flip_rate":     round(self.flip_rate(), 4),
            "last_label":            self._last_label,
            "memory_length":         self._mem_len,
            "base_beta":             self._base_beta,
        })
        return d

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_predict(self, features, deadline):
        if not hasattr(self._clf, "predict_raw") or not self._clf._fitted:
            label, conf = self._clf.predict(features, deadline=deadline)
            return label, conf, {}
        label, conf, proba = self._clf.predict_raw(features)
        return label, conf, proba

    def _compute_beta(self, features: Dict[str, float]) -> float:
        vol = features.get("volatility", 0.01)
        # High volatility → trust model more → lower beta
        vol_factor = math.exp(-vol * 50)
        # Recent flip-rate → higher flip rate → increase beta (more smoothing)
        flip_boost = 0.5 * self.flip_rate()
        beta = self._base_beta * vol_factor + flip_boost
        return min(0.6, max(0.05, beta))

    def _update_state(self, label: str) -> None:
        self._transition.observe(label)
        self._flip_rate_buf.append(label)
        if label != self._last_label:
            self._label_change_count += 1
        self._last_label = label
        self._history.append(label)

    @staticmethod
    def _ensure_full_proba(proba_dict: Dict, label: str, conf: float) -> Dict[str, float]:
        if proba_dict:
            return proba_dict
        p = {l: 0.0 for l in LABELS}
        remaining = 1.0 - conf
        others = [l for l in LABELS if l != label]
        for l in others:
            p[l] = remaining / len(others) if others else 0.0
        p[label] = conf
        return p


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from classification.streaming_classifier import StreamingClassifier
    import random

    clf  = StreamingClassifier()
    pred = StateAwarePredictor(clf)
    rng  = random.Random(99)

    fake_feats = lambda: {
        "volatility": rng.uniform(0, 0.02),
        "momentum_5": rng.uniform(-0.01, 0.01),
        "z_score":    rng.uniform(-2, 2),
        "rsi":        rng.uniform(20, 80),
        "momentum_10": 0.0, "momentum_20": 0.0,
        "macd_signal": 0.0, "vol_ratio": 1.0,
        "return_std": 0.002, "skewness": 0.0,
        "drawdown": -0.01, "range_pct": 0.02,
        "price": 150.0, "ema_fast": 150.0, "ema_slow": 150.0, "ticks_seen": 100.0,
    }

    for i in range(80):
        f = fake_feats()
        label, conf = pred.predict(f)
        if i % 10 == 0:
            print(f"[{i:3d}] {label:<12} conf={conf:.3f}  flip_rate={pred.flip_rate():.3f}")

    print(pred.diagnostics())

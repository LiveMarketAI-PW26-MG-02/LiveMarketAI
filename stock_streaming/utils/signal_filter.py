"""
signal_filter.py — Post-processing filters applied to raw classification outputs
to remove noise, enforce minimum hold durations, and generate actionable
trading signals from the streaming prediction stream.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple
from core.config import CFG


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

class MinimumHoldFilter:
    """
    Suppresses label changes unless the new label has persisted for at least
    `min_hold` consecutive predictions.  Prevents acting on ephemeral spikes.
    """

    def __init__(self, min_hold: int = 3):
        self._min_hold    = min_hold
        self._current     = "HOLD"
        self._candidate   = "HOLD"
        self._hold_count  = 0
        self._changes     = 0

    def apply(self, label: str) -> str:
        if label == self._current:
            self._candidate  = label
            self._hold_count = 0
            return self._current

        if label == self._candidate:
            self._hold_count += 1
            if self._hold_count >= self._min_hold:
                self._current    = label
                self._hold_count = 0
                self._changes   += 1
        else:
            self._candidate  = label
            self._hold_count = 1

        return self._current

    @property
    def total_changes(self) -> int:
        return self._changes


class ConfidenceGatingFilter:
    """
    Only passes through predictions whose confidence exceeds `threshold`.
    Below threshold: the last high-confidence label is repeated.
    """

    def __init__(self, threshold: float = 0.55):
        self._threshold = threshold
        self._last_valid = "HOLD"
        self._suppressed = 0
        self._passed     = 0

    def apply(self, label: str, confidence: float) -> Tuple[str, float, bool]:
        """Returns (output_label, output_confidence, was_passed)."""
        if confidence >= self._threshold:
            self._last_valid = label
            self._passed    += 1
            return label, confidence, True
        else:
            self._suppressed += 1
            return self._last_valid, self._threshold, False

    @property
    def pass_rate(self) -> float:
        total = self._passed + self._suppressed
        return self._passed / total if total else 1.0


class TrendConsistencyFilter:
    """
    Converts a noisy label stream into a smoothed directional signal
    by requiring that a configurable fraction of recent labels agree.
    """

    def __init__(self, window: int = 8, required_fraction: float = 0.6):
        self._window   = window
        self._req_frac = required_fraction
        self._buf: deque = deque(maxlen=window)
        self._last_output = "HOLD"

    def apply(self, label: str) -> str:
        self._buf.append(label)
        if len(self._buf) < self._window // 2:
            return self._last_output

        counts: Dict[str, int] = {}
        for l in self._buf:
            counts[l] = counts.get(l, 0) + 1

        dominant = max(counts, key=counts.get)
        if counts[dominant] / len(self._buf) >= self._req_frac:
            self._last_output = dominant

        return self._last_output


class MomentumAlignmentFilter:
    """
    Checks that the predicted direction aligns with recent price momentum
    (derived from the raw feature vector).  Misaligned predictions are
    downgraded to HOLD.
    """

    def __init__(self, momentum_threshold: float = 0.0005):
        self._thr = momentum_threshold
        self._rejected = 0
        self._accepted = 0

    def apply(self, label: str, features: Dict[str, float]) -> str:
        mom = features.get("momentum_5", 0.0)

        bull = label in ("STRONG_BUY", "BUY")
        bear = label in ("SELL", "STRONG_SELL")

        if bull and mom < -self._thr:
            self._rejected += 1
            return "HOLD"
        if bear and mom >  self._thr:
            self._rejected += 1
            return "HOLD"

        self._accepted += 1
        return label

    @property
    def rejection_rate(self) -> float:
        total = self._accepted + self._rejected
        return self._rejected / total if total else 0.0


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------

class SignalFilterPipeline:
    """
    Chains MinimumHold → ConfidenceGating → TrendConsistency → MomentumAlignment
    into a single call.  Returns a clean, actionable trading signal.
    """

    def __init__(self,
                 min_hold: int           = 3,
                 conf_threshold: float   = 0.55,
                 trend_window: int       = 8,
                 trend_fraction: float   = 0.6,
                 momentum_thr: float     = 0.0005):
        self._hold_filter    = MinimumHoldFilter(min_hold=min_hold)
        self._conf_gate      = ConfidenceGatingFilter(threshold=conf_threshold)
        self._trend_filter   = TrendConsistencyFilter(window=trend_window,
                                                      required_fraction=trend_fraction)
        self._momentum_filter = MomentumAlignmentFilter(momentum_threshold=momentum_thr)
        self._total_in       = 0
        self._history: deque = deque(maxlen=1000)

    def apply(self, label: str, confidence: float,
              features: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Process one prediction through the full filter chain.
        Returns (final_label, adjusted_confidence).
        """
        self._total_in += 1

        # Stage 1: Confidence gate
        gated_label, gated_conf, passed = self._conf_gate.apply(label, confidence)

        # Stage 2: Minimum hold
        held_label = self._hold_filter.apply(gated_label)

        # Stage 3: Trend consistency
        trend_label = self._trend_filter.apply(held_label)

        # Stage 4: Momentum alignment (if features available)
        if features:
            final_label = self._momentum_filter.apply(trend_label, features)
        else:
            final_label = trend_label

        # Confidence correction: reduce by filter rejection multiplier
        conf_adj = gated_conf * (0.7 if final_label == "HOLD" and label != "HOLD" else 1.0)

        self._history.append((final_label, round(conf_adj, 4)))
        return final_label, round(conf_adj, 4)

    def diagnostics(self) -> Dict:
        return {
            "total_inputs":        self._total_in,
            "label_changes":       self._hold_filter.total_changes,
            "conf_pass_rate":      round(self._conf_gate.pass_rate, 4),
            "momentum_rejection":  round(self._momentum_filter.rejection_rate, 4),
            "output_history_len":  len(self._history),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng    = random.Random(7)
    labels = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    pipe   = SignalFilterPipeline()

    prev = None
    changes_in  = 0
    changes_out = 0

    for i in range(200):
        raw_label = rng.choice(labels)
        conf      = rng.uniform(0.3, 0.95)
        feats     = {"momentum_5": rng.uniform(-0.01, 0.01)}
        out_label, out_conf = pipe.apply(raw_label, conf, feats)

        if prev:
            if raw_label != prev[0]:  changes_in  += 1
            if out_label != prev[1]:  changes_out += 1
        prev = (raw_label, out_label)

    print(f"Label flips IN:  {changes_in}")
    print(f"Label flips OUT: {changes_out}  (reduction = {1 - changes_out/(changes_in+1):.1%})")
    print(pipe.diagnostics())

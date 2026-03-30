"""
temporal_smoother.py — Aggregates recent predictions from the streaming
classifier to reduce noise and produce stable final signals.  Supports
majority vote, confidence-weighted majority, and EMA-based smoothing.
"""

import math
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple
from core.config import CFG


class TemporalSmoother:
    """
    Maintains a rolling buffer of (label, confidence) pairs and applies
    one of three smoothing strategies to produce a stable output signal.

    Strategies:
        "majority"           — Unweighted majority vote over the buffer.
        "weighted_majority"  — Votes weighted by confidence scores.
        "ema"                — Exponential moving average on confidence
                               probability vectors; highest EMA label wins.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 method: Optional[str] = None,
                 ema_alpha: Optional[float] = None,
                 noise_threshold: Optional[float] = None):
        cfg  = CFG.smoothing
        self._ws        = window_size      or cfg.window_size
        self._method    = method           or cfg.method
        self._alpha     = ema_alpha        or cfg.ema_alpha
        self._noise_thr = noise_threshold  or cfg.noise_threshold
        self._labels    = CFG.classifier.labels

        self._buf: deque = deque(maxlen=self._ws)
        self._ema: Dict[str, float] = {l: 1.0 / len(self._labels) for l in self._labels}
        self._output_history: deque = deque(maxlen=500)
        self._flip_log: deque       = deque(maxlen=100)
        self._last_output: Optional[str] = None
        self._total_inputs  = 0
        self._total_outputs = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def smooth(self, label: str, confidence: float) -> Tuple[str, float]:
        """
        Ingest one prediction and return the smoothed (label, confidence).
        Low-confidence predictions are treated as noise and down-weighted.
        """
        self._total_inputs += 1

        # Down-weight noisy predictions
        effective_conf = confidence if confidence >= self._noise_thr else confidence * 0.3
        self._buf.append((label, effective_conf))

        if self._method == "majority":
            out_label, out_conf = self._majority_vote()
        elif self._method == "weighted_majority":
            out_label, out_conf = self._weighted_majority()
        elif self._method == "ema":
            out_label, out_conf = self._ema_smooth(label, effective_conf)
        else:
            out_label, out_conf = label, confidence

        self._total_outputs += 1
        self._output_history.append((out_label, round(out_conf, 4)))
        if self._last_output and out_label != self._last_output:
            self._flip_log.append(self._total_outputs)
        self._last_output = out_label

        return out_label, round(out_conf, 4)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _majority_vote(self) -> Tuple[str, float]:
        if not self._buf:
            return "HOLD", 0.5
        counts = Counter(l for l, _ in self._buf)
        winner = counts.most_common(1)[0][0]
        conf   = counts[winner] / len(self._buf)
        return winner, conf

    def _weighted_majority(self) -> Tuple[str, float]:
        if not self._buf:
            return "HOLD", 0.5
        weights: Dict[str, float] = {l: 0.0 for l in self._labels}
        total_w = 0.0
        # More recent predictions get higher weight (linear ramp)
        n = len(self._buf)
        buf_list = list(self._buf)
        for i, (label, conf) in enumerate(buf_list):
            w = (i + 1) / n * conf   # recency × confidence
            weights[label] += w
            total_w += w
        if total_w < 1e-9:
            return "HOLD", 0.5
        winner = max(weights, key=weights.get)
        conf   = weights[winner] / total_w
        return winner, conf

    def _ema_smooth(self, label: str, conf: float) -> Tuple[str, float]:
        # Update EMA for each label based on current vote
        for l in self._labels:
            vote = conf if l == label else 0.0
            self._ema[l] = self._alpha * vote + (1 - self._alpha) * self._ema[l]
        # Normalise
        total = sum(self._ema.values()) + 1e-12
        normed = {l: v / total for l, v in self._ema.items()}
        winner = max(normed, key=normed.get)
        return winner, normed[winner]

    # ------------------------------------------------------------------
    # Responsiveness metrics
    # ------------------------------------------------------------------

    def flip_frequency(self) -> float:
        """Mean number of outputs between label changes (higher = more stable)."""
        if len(self._flip_log) < 2:
            return float(self._total_outputs)
        gaps = [self._flip_log[i] - self._flip_log[i-1]
                for i in range(1, len(self._flip_log))]
        return sum(gaps) / len(gaps)

    def buffer_label_distribution(self) -> Dict[str, float]:
        """Fraction of each label in the current rolling buffer."""
        if not self._buf:
            return {}
        counts: Dict[str, int] = Counter(l for l, _ in self._buf)
        n = len(self._buf)
        return {l: round(counts.get(l, 0) / n, 3) for l in self._labels}

    def output_confidence_stats(self) -> Dict:
        confs = [c for _, c in self._output_history]
        if not confs:
            return {}
        return {
            "mean":   round(sum(confs) / len(confs), 4),
            "min":    round(min(confs), 4),
            "max":    round(max(confs), 4),
            "stdev":  round(math.sqrt(sum((c - sum(confs)/len(confs))**2
                                          for c in confs) / len(confs)), 4),
        }

    def diagnostics(self) -> Dict:
        return {
            "method":          self._method,
            "buffer_size":     self._ws,
            "buffer_fill":     len(self._buf),
            "total_inputs":    self._total_inputs,
            "total_outputs":   self._total_outputs,
            "flip_frequency":  round(self.flip_frequency(), 2),
            "last_output":     self._last_output,
            "label_dist":      self.buffer_label_distribution(),
            "conf_stats":      self.output_confidence_stats(),
        }


# ---------------------------------------------------------------------------
# Smoke test: compare strategies
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(42)

    labels   = CFG.classifier.labels
    # Noisy stream: mostly BUY with occasional SELL spikes
    stream   = (["BUY"] * 7 + ["SELL"] * 3) * 20

    for method in ("majority", "weighted_majority", "ema"):
        smoother = TemporalSmoother(window_size=10, method=method)
        flips = 0
        prev  = None
        for raw_label in stream:
            conf = rng.uniform(0.4, 0.95)
            out_label, out_conf = smoother.smooth(raw_label, conf)
            if prev and out_label != prev:
                flips += 1
            prev = out_label
        print(f"method={method:<20} flips={flips:3d}  "
              f"flip_freq={smoother.flip_frequency():.2f}")
        print(f"  conf_stats={smoother.output_confidence_stats()}")

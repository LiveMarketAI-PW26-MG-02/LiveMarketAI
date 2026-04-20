"""
metrics_collector.py — Thread-safe collector of prediction metrics for use
across all evaluation and benchmarking modules.
"""

import math
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
from core.config import CFG


class MetricsCollector:
    """
    Collects per-prediction metrics and provides aggregated statistics.

    Metrics tracked per prediction:
        - label
        - confidence
        - latency_ms
        - optional ground_truth label (for accuracy if available)
    """

    def __init__(self, name: str, buffer_size: int = 5000):
        self.name   = name
        self._lock  = threading.Lock()
        self._labels:     deque = deque(maxlen=buffer_size)
        self._confidences: deque = deque(maxlen=buffer_size)
        self._latencies:  deque = deque(maxlen=buffer_size)
        self._gtruth:     deque = deque(maxlen=buffer_size)
        self._n    = 0
        self._flips = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def record(self, label: str, confidence: float, latency_ms: float,
               ground_truth: Optional[str] = None) -> None:
        with self._lock:
            if self._labels and label != self._labels[-1]:
                self._flips += 1
            self._labels.append(label)
            self._confidences.append(confidence)
            self._latencies.append(latency_ms)
            self._gtruth.append(ground_truth)
            self._n += 1

    def record_event(self, event) -> None:
        """Convenience wrapper accepting a PredictionEvent object."""
        self.record(
            label=event.label,
            confidence=event.confidence,
            latency_ms=event.latency_ms,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        with self._lock:
            labels  = list(self._labels)
            confs   = list(self._confidences)
            lats    = list(self._latencies)
            gt      = list(self._gtruth)
            n       = self._n
            flips   = self._flips

        if not labels:
            return {"name": self.name, "n_predictions": 0}

        # Latency stats
        sorted_lats = sorted(lats)
        def pct(p):
            idx = max(0, int(math.ceil(p / 100 * len(sorted_lats))) - 1)
            return sorted_lats[idx]

        mean_lat = sum(lats) / len(lats)
        sla_ok   = sum(1 for l in lats if l <= CFG.stream.inference_timeout_ms)

        # Confidence stats
        mean_conf = sum(confs) / len(confs)

        # Label distribution
        label_dist: Dict[str, int] = {}
        for l in labels:
            label_dist[l] = label_dist.get(l, 0) + 1
        label_pct = {l: round(c / n, 4) for l, c in label_dist.items()}

        # Accuracy (if ground truth available)
        gt_avail = [i for i in range(len(gt)) if gt[i] is not None]
        accuracy = None
        if gt_avail:
            correct = sum(1 for i in gt_avail if labels[i] == gt[i])
            accuracy = correct / len(gt_avail)

        # Flip rate
        flip_rate = flips / max(1, n - 1)

        return {
            "name":             self.name,
            "n_predictions":    n,
            "mean_latency_ms":  round(mean_lat, 4),
            "p50_latency_ms":   round(pct(50), 4),
            "p90_latency_ms":   round(pct(90), 4),
            "p99_latency_ms":   round(pct(99), 4),
            "max_latency_ms":   round(max(lats), 4),
            "sla_compliance":   round(sla_ok / n, 4),
            "mean_confidence":  round(mean_conf, 4),
            "min_confidence":   round(min(confs), 4),
            "max_confidence":   round(max(confs), 4),
            "label_flips":      flips,
            "flip_rate":        round(flip_rate, 4),
            "label_distribution": label_pct,
            "accuracy":         round(accuracy, 4) if accuracy is not None else None,
        }

    # ------------------------------------------------------------------
    # Window-based analysis
    # ------------------------------------------------------------------

    def rolling_accuracy(self, window: int = 100) -> Optional[float]:
        """Accuracy over the last `window` labelled predictions."""
        with self._lock:
            labels = list(self._labels)[-window:]
            gt     = list(self._gtruth)[-window:]
        pairs = [(l, g) for l, g in zip(labels, gt) if g is not None]
        if not pairs:
            return None
        return sum(1 for l, g in pairs if l == g) / len(pairs)

    def rolling_flip_rate(self, window: int = 50) -> float:
        """Flip rate over the last `window` predictions."""
        with self._lock:
            labels = list(self._labels)[-window:]
        if len(labels) < 2:
            return 0.0
        flips = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        return flips / (len(labels) - 1)

    def confidence_trend(self, window: int = 50) -> str:
        """Is confidence going UP, DOWN, or STABLE over the last window?"""
        with self._lock:
            confs = list(self._confidences)[-window:]
        if len(confs) < 4:
            return "STABLE"
        half = len(confs) // 2
        first_half = sum(confs[:half]) / half
        second_half = sum(confs[half:]) / (len(confs) - half)
        delta = second_half - first_half
        if delta >  0.02: return "UP"
        if delta < -0.02: return "DOWN"
        return "STABLE"

    def reset(self) -> None:
        with self._lock:
            self._labels.clear()
            self._confidences.clear()
            self._latencies.clear()
            self._gtruth.clear()
            self._n = 0
            self._flips = 0

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n── MetricsCollector: {s['name']} ──")
        for k, v in s.items():
            if k != "name":
                print(f"  {k:<25}: {v}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(0)
    mc  = MetricsCollector("Test")

    labels = ["BUY", "HOLD", "SELL", "BUY", "BUY", "SELL"]
    for i in range(200):
        lbl  = rng.choice(labels)
        conf = rng.uniform(0.4, 0.95)
        lat  = rng.uniform(0.5, 15.0)
        gt   = rng.choice(labels) if i % 5 == 0 else None
        mc.record(lbl, conf, lat, ground_truth=gt)

    mc.print_summary()
    print(f"\nRolling flip rate (last 50): {mc.rolling_flip_rate(50):.4f}")
    print(f"Confidence trend          : {mc.confidence_trend()}")

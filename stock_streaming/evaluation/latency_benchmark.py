"""
latency_benchmark.py — Measures end-to-end prediction latency of both the
streaming and batch classifiers under simulated real-time conditions.
Reports percentile distributions and compares against the inference timeout SLA.
"""

import time
import math
import statistics
from typing import Callable, Dict, List, Optional, Tuple
from core.config import CFG


class LatencyRecord:
    """Single timing observation."""
    __slots__ = ("tick_index", "latency_ms", "timed_out", "label", "confidence")

    def __init__(self, tick_index: int, latency_ms: float,
                 timed_out: bool, label: str, confidence: float):
        self.tick_index  = tick_index
        self.latency_ms  = latency_ms
        self.timed_out   = timed_out
        self.label       = label
        self.confidence  = confidence


class LatencyBenchmark:
    """
    Runs a predict_fn over a prepared feature list and collects precise
    latency measurements.  Reports SLA compliance and percentile breakdown.
    """

    SLA_MS = CFG.stream.inference_timeout_ms

    def __init__(self, name: str):
        self.name = name
        self._records: List[LatencyRecord] = []

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    def run(self, predict_fn: Callable[[Dict], Tuple[str, float]],
            feature_list: List[Dict[str, float]],
            warm_up: int = 10) -> "LatencyBenchmark":
        """
        Parameters
        ----------
        predict_fn   : callable(features) → (label, confidence)
        feature_list : list of feature dicts to run through predict_fn
        warm_up      : First N calls are discarded (JIT / cache warm-up)
        """
        print(f"[LatencyBenchmark:{self.name}] Running {len(feature_list)} samples "
              f"(warm_up={warm_up}) …")
        for i, feats in enumerate(feature_list):
            deadline = time.perf_counter() + self.SLA_MS / 1000.0
            t0 = time.perf_counter()
            label, conf = predict_fn(feats)
            elapsed = (time.perf_counter() - t0) * 1000.0
            timed_out = time.perf_counter() > deadline

            if i >= warm_up:
                self._records.append(LatencyRecord(
                    tick_index=i,
                    latency_ms=elapsed,
                    timed_out=timed_out,
                    label=label,
                    confidence=conf,
                ))
        print(f"[LatencyBenchmark:{self.name}] Done. "
              f"Recorded {len(self._records)} measurements.")
        return self

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def percentile(self, p: float) -> float:
        """Return the p-th percentile latency in ms."""
        if not self._records:
            return 0.0
        sorted_lats = sorted(r.latency_ms for r in self._records)
        idx = max(0, int(math.ceil(p / 100 * len(sorted_lats))) - 1)
        return sorted_lats[idx]

    def sla_compliance(self) -> float:
        """Fraction of predictions that met the SLA."""
        if not self._records:
            return 1.0
        met = sum(1 for r in self._records if r.latency_ms <= self.SLA_MS)
        return met / len(self._records)

    def summary(self) -> Dict:
        if not self._records:
            return {}
        lats = [r.latency_ms for r in self._records]
        return {
            "name":              self.name,
            "n_predictions":     len(self._records),
            "mean_ms":           round(statistics.mean(lats), 3),
            "median_ms":         round(statistics.median(lats), 3),
            "stdev_ms":          round(statistics.stdev(lats), 3) if len(lats) > 1 else 0.0,
            "p90_ms":            round(self.percentile(90), 3),
            "p95_ms":            round(self.percentile(95), 3),
            "p99_ms":            round(self.percentile(99), 3),
            "max_ms":            round(max(lats), 3),
            "min_ms":            round(min(lats), 3),
            "sla_ms":            self.SLA_MS,
            "sla_compliance":    round(self.sla_compliance(), 4),
            "timeout_count":     sum(1 for r in self._records if r.timed_out),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*55}")
        print(f"  Latency Benchmark — {s['name']}")
        print(f"{'='*55}")
        print(f"  Samples         : {s['n_predictions']}")
        print(f"  Mean            : {s['mean_ms']:.3f} ms")
        print(f"  Median (p50)    : {s['median_ms']:.3f} ms")
        print(f"  p90             : {s['p90_ms']:.3f} ms")
        print(f"  p95             : {s['p95_ms']:.3f} ms")
        print(f"  p99             : {s['p99_ms']:.3f} ms")
        print(f"  Max             : {s['max_ms']:.3f} ms")
        print(f"  SLA ({s['sla_ms']:.0f} ms)    : {s['sla_compliance']*100:.1f}% compliant")
        print(f"  Timeouts        : {s['timeout_count']}")
        print(f"{'='*55}")

    def compare(self, other: "LatencyBenchmark") -> Dict:
        """Return improvement statistics of self vs other."""
        s = self.summary()
        o = other.summary()
        improvements = {}
        for key in ("mean_ms", "p90_ms", "p99_ms"):
            if o.get(key, 0) > 0:
                pct = (o[key] - s[key]) / o[key] * 100
                improvements[f"{key}_improvement_%"] = round(pct, 2)
        improvements["sla_compliance_delta"] = round(
            s.get("sla_compliance", 0) - o.get("sla_compliance", 0), 4)
        return improvements


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from classification.batch_classifier import BatchClassifier, generate_synthetic_training_data
    from classification.base_classifier import BaseClassifier

    rng = random.Random(0)
    X, y = generate_synthetic_training_data(n_samples=500)

    # Batch classifier
    batch_clf = BatchClassifier(n_estimators=50)
    batch_clf.fit(X[:400], y[:400])

    bench = LatencyBenchmark("BatchClassifier")
    bench.run(lambda f: batch_clf.predict(f), X[400:], warm_up=5)
    bench.print_summary()

    # Heuristic baseline
    heuristic = lambda f: BaseClassifier._heuristic_predict(f)
    bench2 = LatencyBenchmark("Heuristic")
    bench2.run(heuristic, X[400:], warm_up=5)
    bench2.print_summary()

    print("\nHeuristic vs Batch improvements:")
    print(bench2.compare(bench))

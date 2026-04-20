"""
batch_vs_streaming.py — Head-to-head comparison of batch and streaming
classifiers across latency, label stability, and adaptability metrics
when processing the same simulated stock price stream.
"""

import time
import random
import math
from typing import Dict, List, Optional, Tuple
from core.config import CFG
from core.data_generator import StockDataGenerator
from core.feature_extractor import FeatureExtractor
from core.window_manager import WindowManager
from classification.batch_classifier import BatchClassifier, generate_synthetic_training_data
from classification.streaming_classifier import StreamingClassifier
from classification.base_classifier import LABELS
from evaluation.metrics_collector import MetricsCollector


class BatchVsStreamingEval:
    """
    Runs both classifiers on an identical synthetic tick stream and compares:
        - Prediction latency (mean, p99)
        - Label flip frequency (stability)
        - Accuracy on labeled windows (adaptability)
        - Confidence evolution over time
    """

    def __init__(self, n_ticks: int = 1000, seed: Optional[int] = None):
        self._n_ticks = n_ticks
        self._seed    = seed or CFG.evaluation.random_seed
        self._results: Dict = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        print("=" * 60)
        print("  Batch vs Streaming Classification Evaluation")
        print("=" * 60)

        # 1. Pre-train the batch classifier
        print("\n[1/4] Training batch classifier on synthetic history …")
        X_train, y_train = generate_synthetic_training_data(n_samples=1500, seed=self._seed)
        batch_clf = BatchClassifier(n_estimators=100)
        batch_clf.fit(X_train, y_train)

        # 2. Initialise streaming classifier (starts with no training)
        streaming_clf = StreamingClassifier(min_fit_samples=30)

        # 3. Generate shared evaluation stream
        print(f"\n[2/4] Generating {self._n_ticks}-tick evaluation stream …")
        gen = StockDataGenerator(seed=self._seed + 1)
        ticks = gen.generate_batch(self._n_ticks)

        # 4. Run both classifiers on the stream
        print("\n[3/4] Running classifiers …")
        batch_metrics    = MetricsCollector("Batch")
        stream_metrics   = MetricsCollector("Streaming")

        fx_batch   = FeatureExtractor(window_size=60)
        fx_stream  = FeatureExtractor(window_size=60)

        for i, tick in enumerate(ticks):
            feats_b = self._step(tick, fx_batch)
            feats_s = self._step(tick, fx_stream)
            if feats_b is None:
                continue

            # Batch predict
            t0 = time.perf_counter()
            b_label, b_conf = batch_clf.predict(feats_b)
            b_lat = (time.perf_counter() - t0) * 1000
            batch_metrics.record(b_label, b_conf, b_lat)

            # Streaming predict + update
            t0 = time.perf_counter()
            s_label, s_conf = streaming_clf.predict(feats_s)
            s_lat = (time.perf_counter() - t0) * 1000
            stream_metrics.record(s_label, s_conf, s_lat)

            # Self-supervised label
            pseudo_label = "BUY" if feats_s["momentum_5"] > 0 else "SELL"
            streaming_clf.update(feats_s, pseudo_label)

        # 5. Collect and display results
        print("\n[4/4] Results:\n")
        b_summary = batch_metrics.summary()
        s_summary = stream_metrics.summary()
        self._print_comparison(b_summary, s_summary)

        self._results = {
            "batch":     b_summary,
            "streaming": s_summary,
            "deltas":    self._compute_deltas(b_summary, s_summary),
        }
        return self._results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step(tick, fx: FeatureExtractor) -> Optional[Dict]:
        fx.update(tick)
        return fx.extract()

    @staticmethod
    def _print_comparison(b: Dict, s: Dict) -> None:
        rows = [
            ("Predictions",        b["n_predictions"],    s["n_predictions"],   ""),
            ("Mean latency (ms)",  b["mean_latency_ms"],  s["mean_latency_ms"], "ms"),
            ("P99 latency (ms)",   b["p99_latency_ms"],   s["p99_latency_ms"],  "ms"),
            ("Label flips",        b["label_flips"],       s["label_flips"],     ""),
            ("Flip rate",          b["flip_rate"],         s["flip_rate"],       ""),
            ("Mean confidence",    b["mean_confidence"],   s["mean_confidence"], ""),
            ("SLA compliance",     b["sla_compliance"],    s["sla_compliance"],  ""),
        ]
        hdr = f"{'Metric':<25} {'Batch':>12} {'Streaming':>12}"
        print(hdr)
        print("-" * len(hdr))
        for name, bv, sv, unit in rows:
            bv_str = f"{bv:.4f}{unit}" if isinstance(bv, float) else str(bv)
            sv_str = f"{sv:.4f}{unit}" if isinstance(sv, float) else str(sv)
            print(f"  {name:<23} {bv_str:>12} {sv_str:>12}")

    @staticmethod
    def _compute_deltas(b: Dict, s: Dict) -> Dict:
        deltas = {}
        for key in ("mean_latency_ms", "p99_latency_ms", "flip_rate", "mean_confidence"):
            if key in b and key in s and b[key] != 0:
                deltas[f"{key}_delta_%"] = round(
                    (s[key] - b[key]) / (abs(b[key]) + 1e-9) * 100, 2)
        return deltas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    eval_runner = BatchVsStreamingEval(n_ticks=800)
    results = eval_runner.run()
    print("\nKey deltas (streaming vs batch):")
    for k, v in results["deltas"].items():
        direction = "↑" if v > 0 else "↓"
        print(f"  {k}: {direction} {abs(v):.2f}%")

"""
main.py — Entry point for the Stock Streaming Window-Based Classification System.

Run modes:
    python main.py --mode stream      Live streaming demo (default)
    python main.py --mode batch       Batch vs streaming comparison
    python main.py --mode responsiveness  Window responsiveness analysis
    python main.py --mode full        All modes sequentially
"""

import argparse
import sys
import time

# ── Core ────────────────────────────────────────────────────────────────────
from core.config import CFG, print_config
from core.data_generator import StockDataGenerator
from core.feature_extractor import FeatureExtractor
from core.window_manager import WindowManager

# ── Classification ───────────────────────────────────────────────────────────
from classification.streaming_classifier import StreamingClassifier
from classification.batch_classifier import BatchClassifier, generate_synthetic_training_data
from classification.confidence_estimator import ConfidenceEstimator
from classification.state_aware_predictor import StateAwarePredictor

# ── Windowing ────────────────────────────────────────────────────────────────
from windowing.volatility_detector import VolatilityDetector
from windowing.window_resizer import WindowResizer
from windowing.temporal_smoother import TemporalSmoother
from windowing.overlapping_window import OverlappingWindowScheduler
from windowing.dynamic_window import DynamicWindowResizer

# ── Evaluation ───────────────────────────────────────────────────────────────
from evaluation.batch_vs_streaming import BatchVsStreamingEval
from evaluation.latency_benchmark import LatencyBenchmark
from evaluation.metrics_collector import MetricsCollector
from evaluation.responsiveness_analyzer import WindowResponsivenessAnalyzer
from evaluation.report_generator import ReportGenerator

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.logger import get_logger, flush as flush_logs
from utils.data_store import DataStore
from utils.signal_filter import SignalFilterPipeline
from utils.visualization import make_display
from utils.helpers import Timer, ensure_dir

log = get_logger("main")


# ═══════════════════════════════════════════════════════════════════════════
# Mode: STREAM — live streaming demo
# ═══════════════════════════════════════════════════════════════════════════

def run_stream(duration_seconds: int = 15, seed: int = 42) -> None:
    print("\n" + "=" * 60)
    print("  MODE: Live Streaming Classification Demo")
    print(f"  Duration: {duration_seconds}s  |  Tick interval: {CFG.stream.tick_interval_ms}ms")
    print("=" * 60 + "\n")

    # Build pipeline components
    gen          = StockDataGenerator(symbol="SYNTH", seed=seed)
    wm           = WindowManager()
    fx           = FeatureExtractor(window_size=CFG.window.base_window_size)
    vol_det      = VolatilityDetector()
    resizer      = WindowResizer(vol_detector=vol_det)
    smoother     = TemporalSmoother(method="weighted_majority")
    sig_filter   = SignalFilterPipeline()
    conf_est     = ConfidenceEstimator()
    base_clf     = StreamingClassifier()
    predictor    = StateAwarePredictor(base_clf)
    store        = DataStore()
    metrics      = MetricsCollector("streaming_demo")
    display      = make_display()

    store.start()
    gen.start()
    warm_up_done = False
    warm_up_n    = CFG.stream.warm_up_ticks

    print("[main] Warming up … (first predictions will appear shortly)\n")

    deadline = time.time() + duration_seconds
    tick_count = 0

    try:
        for tick in gen.iter_ticks():
            if time.time() > deadline:
                break

            tick_count += 1

            # 1. Feature extraction (incremental)
            fx.update(tick)
            feats = fx.extract()
            if feats is None:
                continue

            # 2. Volatility / window adaptation
            new_ws = resizer.update(feats["volatility"], tick.volume, price=tick.price)
            if new_ws:
                wm.resize(new_ws)
                fx.resize(new_ws)
                store.windows.record_resize(wm.current_window_size, new_ws, "vol_driven")
                log.info("Window resized", old=wm.current_window_size, new=new_ws)

            # 3. Sliding window
            snapshot = wm.ingest(tick)
            if snapshot is None:
                continue

            # 4. Warm-up guard
            if not warm_up_done:
                warm_up_n -= 1
                if warm_up_n <= 0:
                    warm_up_done = True
                    print("[main] Warm-up complete — emitting predictions\n")
                continue

            # 5. State-aware prediction
            with Timer("predict") as t:
                label, conf = predictor.predict(feats)

            # 6. Online update (self-supervised pseudo-label from momentum)
            pseudo_lbl = "BUY" if feats["momentum_5"] > 0 else "SELL"
            predictor.update(feats, pseudo_lbl)

            # 7. Temporal smoothing
            sm_label, sm_conf = smoother.smooth(label, conf)

            # 8. Confidence estimation
            adj_label, adj_conf = conf_est.update(
                sm_label, sm_conf,
                window_fill_pct=wm.buffer_fill_pct
            )

            # 9. Signal filtering
            final_label, final_conf = sig_filter.apply(adj_label, adj_conf, feats)

            # 10. Store & display — wrap in a simple namespace
            class _Evt:
                pass
            evt = _Evt()
            evt.tick_index         = tick.tick_index
            evt.symbol             = tick.symbol
            evt.label              = final_label
            evt.smoothed_label     = final_label
            evt.confidence         = final_conf
            evt.smoothed_confidence = final_conf
            evt.latency_ms         = t.ms
            evt.window_size        = wm.current_window_size

            store.predictions.push(evt)
            store.features.push(tick.tick_index, feats)
            metrics.record(final_label, final_conf, t.ms)
            display.update(evt)

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
    finally:
        gen.stop()

    print("\n\n[main] Streaming complete.")
    print(f"  Ticks processed    : {tick_count}")
    print(f"  Predictions made   : {len(store.predictions)}")
    print(f"  Window resizes     : {store.windows.total_resizes}")
    summary = metrics.summary()
    print(f"  Mean latency       : {summary.get('mean_latency_ms', 0):.3f} ms")
    print(f"  SLA compliance     : {summary.get('sla_compliance', 0)*100:.1f}%")
    print(f"  Label flip rate    : {summary.get('flip_rate', 0):.4f}")
    print(f"  Final predictor    : {predictor.diagnostics()}")


# ═══════════════════════════════════════════════════════════════════════════
# Mode: BATCH — batch vs streaming comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_batch_comparison(n_ticks: int = 600) -> dict:
    evaluator = BatchVsStreamingEval(n_ticks=n_ticks)
    return evaluator.run()


# ═══════════════════════════════════════════════════════════════════════════
# Mode: RESPONSIVENESS — window size trade-off analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_responsiveness() -> dict:
    analyzer = WindowResponsivenessAnalyzer(seed=42)
    results  = analyzer.run()
    best     = analyzer.best_window()
    print(f"\n★  Recommended window size: {best}\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Mode: FULL — run everything and generate report
# ═══════════════════════════════════════════════════════════════════════════

def run_full() -> None:
    rg = ReportGenerator()

    print("\n[1/3] Batch vs Streaming Comparison …")
    comp_results = run_batch_comparison(n_ticks=500)
    rg.add_comparison_section(comp_results["batch"], comp_results["streaming"])

    print("\n[2/3] Latency Benchmark …")
    X, y = generate_synthetic_training_data(n_samples=300)
    clf  = BatchClassifier(n_estimators=30)
    clf.fit(X[:250], y[:250])
    bench = LatencyBenchmark("BatchClassifier_full")
    bench.run(lambda f: clf.predict(f), X[250:], warm_up=5)
    bench.print_summary()
    rg.add_latency_section(bench.summary())

    print("\n[3/3] Window Responsiveness Analysis …")
    resp_results = run_responsiveness()
    rg.add_responsiveness_section(resp_results)

    ensure_dir("reports")
    rg.save("full_evaluation", output_dir="reports")
    print("\n[main] Full evaluation complete. Reports saved to ./reports/")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stock Streaming Window-Based Classification System")
    p.add_argument("--mode", choices=["stream", "batch", "responsiveness", "full"],
                   default="stream",
                   help="Execution mode (default: stream)")
    p.add_argument("--duration", type=int, default=15,
                   help="Stream demo duration in seconds (default: 15)")
    p.add_argument("--ticks", type=int, default=600,
                   help="Number of ticks for batch comparison (default: 600)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--config", action="store_true",
                   help="Print active configuration and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        print_config()
        return

    if args.mode == "stream":
        run_stream(duration_seconds=args.duration, seed=args.seed)
    elif args.mode == "batch":
        run_batch_comparison(n_ticks=args.ticks)
    elif args.mode == "responsiveness":
        run_responsiveness()
    elif args.mode == "full":
        run_full()

    flush_logs()


if __name__ == "__main__":
    main()

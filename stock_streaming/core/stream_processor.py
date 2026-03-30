"""
stream_processor.py — Top-level orchestrator that wires together the generator,
window manager, feature extractor, classifier, and temporal smoother into
a low-latency streaming inference loop.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from core.config import CFG
from core.data_generator import StockDataGenerator, StockTick
from core.window_manager import WindowManager, WindowSnapshot
from core.feature_extractor import FeatureExtractor


@dataclass
class PredictionEvent:
    """Result emitted by the streaming pipeline for each classification."""
    window_id: int
    tick_index: int
    timestamp: float
    symbol: str
    label: str
    confidence: float
    smoothed_label: str
    smoothed_confidence: float
    latency_ms: float
    window_size: int
    raw_features: Dict[str, float] = field(default_factory=dict)


class StreamProcessor:
    """
    Orchestrates the end-to-end streaming classification loop.

    Pipeline per tick:
        tick → WindowManager → (on window fire) FeatureExtractor.extract()
             → Classifier.predict() → TemporalSmoother.smooth()
             → PredictionEvent → callback(s)
    """

    def __init__(self,
                 classifier,
                 smoother,
                 volatility_detector,
                 symbol: str = "SYNTH",
                 seed: Optional[int] = None):
        self._classifier   = classifier
        self._smoother     = smoother
        self._vol_detector = volatility_detector
        self._symbol       = symbol

        self._gen    = StockDataGenerator(symbol=symbol, seed=seed)
        self._wm     = WindowManager()
        self._fx     = FeatureExtractor(window_size=CFG.window.base_window_size)
        self._callbacks: List[Callable[[PredictionEvent], None]] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._warm_up_remaining = CFG.stream.warm_up_ticks
        self._total_predictions = 0
        self._latencies: List[float] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_callback(self, fn: Callable[[PredictionEvent], None]) -> None:
        """Register a function to be called whenever a prediction is ready."""
        self._callbacks.append(fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._gen.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._gen.stop()
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        for tick in self._gen.iter_ticks():
            if not self._running:
                break
            self._process_tick(tick)

    def _process_tick(self, tick: StockTick) -> None:
        t0 = time.perf_counter()

        # Always update incremental features
        self._fx.update(tick)

        # Possibly adapt volatility
        feats = self._fx.extract()
        if feats:
            vol = feats.get("volatility", 0.0)
            self._vol_detector.observe(vol, tick.volume)
            new_ws = self._vol_detector.recommended_window_size()
            resized = self._wm.resize(new_ws)
            if resized:
                self._fx.resize(new_ws)

        # Feed into window manager
        snapshot = self._wm.ingest(tick)
        if snapshot is None:
            return  # Not time for a prediction yet

        # Warm-up guard
        if self._warm_up_remaining > 0:
            self._warm_up_remaining -= 1
            return

        # Feature extraction
        feats = self._fx.extract()
        if feats is None:
            return

        # Classify with hard latency budget
        deadline = time.perf_counter() + CFG.stream.inference_timeout_ms / 1000.0
        label, confidence = self._classifier.predict(feats, deadline=deadline)

        # Temporal smoothing
        s_label, s_conf = self._smoother.smooth(label, confidence)

        # Latency
        latency_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_predictions += 1

        event = PredictionEvent(
            window_id=snapshot.window_id,
            tick_index=tick.tick_index,
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            label=label,
            confidence=round(confidence, 4),
            smoothed_label=s_label,
            smoothed_confidence=round(s_conf, 4),
            latency_ms=round(latency_ms, 3),
            window_size=snapshot.window_size,
            raw_features=feats,
        )
        for cb in self._callbacks:
            cb(event)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        with self._lock:
            lats = list(self._latencies)
        if not lats:
            return {"total_predictions": 0}
        lats_sorted = sorted(lats)
        n = len(lats_sorted)
        def pct(p): return lats_sorted[int(p / 100 * n) - 1]
        return {
            "total_predictions": self._total_predictions,
            "latency_p50_ms":   round(pct(50), 3),
            "latency_p90_ms":   round(pct(90), 3),
            "latency_p99_ms":   round(pct(99), 3),
            "latency_mean_ms":  round(sum(lats) / n, 3),
            "window_size":      self._wm.current_window_size,
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal stubs so this file can run standalone
    class _DummyClassifier:
        def predict(self, feats, deadline=None):
            import random; return random.choice(["BUY", "HOLD", "SELL"]), random.random()

    class _DummySmoother:
        def smooth(self, label, conf):
            return label, conf

    class _DummyVol:
        def observe(self, v, vol): pass
        def recommended_window_size(self): return CFG.window.base_window_size

    proc = StreamProcessor(_DummyClassifier(), _DummySmoother(), _DummyVol(), seed=7)
    seen = []
    proc.add_callback(lambda e: seen.append(e))
    proc.start()
    time.sleep(3)
    proc.stop()
    print(f"Predictions received: {len(seen)}")
    print(proc.stats())
